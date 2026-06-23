import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


C0 = 0.28209479177387814


def load_splats(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "splats" not in checkpoint:
        raise ValueError(f"{path} does not contain a 'splats' state dictionary")
    splats = checkpoint["splats"]
    required = {"means", "scales"}
    missing = sorted(required - set(splats))
    if missing:
        raise ValueError(f"Checkpoint is missing: {', '.join(missing)}")
    return splats


def gaussian_weights(splats, mode):
    count = len(splats["means"])
    if mode == "density":
        return torch.ones(count, dtype=torch.float32)

    if "sh0" not in splats:
        raise ValueError(f"Weight mode {mode!r} requires checkpoint field 'sh0'")
    echo = (splats["sh0"].float().reshape(count, -1).mean(dim=1) * C0 + 0.5)
    echo = echo.clamp_min(0.0)
    if mode == "echo":
        return echo

    if "transmittances" not in splats:
        raise ValueError(
            f"Weight mode {mode!r} requires checkpoint field 'transmittances'"
        )
    attenuation = 1.0 - torch.sigmoid(
        splats["transmittances"].float().reshape(count, -1).mean(dim=1)
    )
    if mode == "attenuation":
        return attenuation
    return echo * attenuation


def scatter_trilinear(volume, points, weights):
    resolution = torch.tensor(volume.shape, dtype=torch.long)
    base = torch.floor(points).long()
    fraction = points - base.float()
    flat = volume.reshape(-1)

    for dz in (0, 1):
        wz = fraction[:, 0] if dz else 1.0 - fraction[:, 0]
        for dy in (0, 1):
            wy = fraction[:, 1] if dy else 1.0 - fraction[:, 1]
            for dx in (0, 1):
                wx = fraction[:, 2] if dx else 1.0 - fraction[:, 2]
                indices = base + torch.tensor([dz, dy, dx])
                valid = ((indices >= 0) & (indices < resolution)).all(dim=1)
                if not valid.any():
                    continue
                selected = indices[valid]
                linear = (
                    selected[:, 0] * resolution[1] * resolution[2]
                    + selected[:, 1] * resolution[2]
                    + selected[:, 2]
                )
                flat.index_add_(
                    0,
                    linear,
                    weights[valid] * wz[valid] * wy[valid] * wx[valid],
                )


def gaussian_kernel_1d(sigma, device):
    sigma = max(float(sigma), 0.35)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device).float()
    kernel = torch.exp(-0.5 * (coordinates / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur_3d(volume, sigma_zyx):
    result = volume[None, None]
    for axis, sigma in enumerate(sigma_zyx):
        kernel = gaussian_kernel_1d(sigma, volume.device)
        radius = kernel.numel() // 2
        if axis == 0:
            weight = kernel[:, None, None][None, None]
            padding = (radius, 0, 0)
        elif axis == 1:
            weight = kernel[None, :, None][None, None]
            padding = (0, radius, 0)
        else:
            weight = kernel[None, None, :][None, None]
            padding = (0, 0, radius)
        result = F.conv3d(result, weight, padding=padding)
    return result[0, 0]


def build_volume(splats, args):
    means = splats["means"].detach().float()
    scales = torch.exp(splats["scales"].detach().float())
    weights = gaussian_weights(splats, args.weight_mode)

    finite = (
        torch.isfinite(means).all(dim=1)
        & torch.isfinite(scales).all(dim=1)
        & torch.isfinite(weights)
        & (weights > float(args.min_weight))
    )
    means = means[finite]
    scales = scales[finite]
    weights = weights[finite]
    if len(means) == 0:
        raise ValueError("No finite Gaussians remain after weight filtering")

    lower = torch.quantile(means, float(args.crop_quantile), dim=0)
    upper = torch.quantile(means, 1.0 - float(args.crop_quantile), dim=0)
    learned_padding = scales.max(dim=1).values.quantile(0.9) * 3.0
    padding = max(float(learned_padding), float(args.padding))
    lower -= padding
    upper += padding

    resolution_xyz = torch.tensor(args.resolution, dtype=torch.long)
    resolution_zyx = resolution_xyz.flip(0)
    spacing_xyz = (upper - lower) / (resolution_xyz.float() - 1.0)
    spacing_zyx = spacing_xyz.flip(0)
    points_xyz = (means - lower) / spacing_xyz
    points_zyx = points_xyz[:, [2, 1, 0]]

    size_measure = scales.prod(dim=1).pow(1.0 / 3.0)
    boundaries = torch.quantile(
        size_measure,
        torch.linspace(0.0, 1.0, int(args.scale_buckets) + 1),
    )
    output = torch.zeros(tuple(resolution_zyx.tolist()), dtype=torch.float32)

    for bucket in range(int(args.scale_buckets)):
        if bucket == int(args.scale_buckets) - 1:
            selected = (
                (size_measure >= boundaries[bucket])
                & (size_measure <= boundaries[bucket + 1])
            )
        else:
            selected = (
                (size_measure >= boundaries[bucket])
                & (size_measure < boundaries[bucket + 1])
            )
        if not selected.any():
            continue
        bucket_volume = torch.zeros_like(output)
        scatter_trilinear(
            bucket_volume,
            points_zyx[selected],
            weights[selected],
        )
        sigma_world = float(size_measure[selected].median())
        sigma_zyx = (sigma_world / spacing_zyx).clamp(0.35, args.max_sigma_voxels)
        output += gaussian_blur_3d(bucket_volume, sigma_zyx.tolist())

    maximum = float(output.max())
    if maximum <= 0.0:
        raise ValueError("The generated scalar volume is empty")
    output /= maximum
    print(
        f"volume: shape={tuple(output.shape)}, gaussians={len(means)}, "
        f"bounds_min={lower.tolist()}, bounds_max={upper.tolist()}, "
        f"spacing_xyz={spacing_xyz.tolist()}"
    )
    return output.numpy(), lower.numpy(), spacing_xyz.numpy()


def write_binary_ply(path, vertices, faces, normals):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
        ]
    )
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    for index, name in enumerate(("x", "y", "z")):
        vertex_data[name] = vertices[:, index]
    for index, name in enumerate(("nx", "ny", "nz")):
        vertex_data[name] = normals[:, index]

    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data["count"] = 3
    face_data["indices"] = faces.astype(np.int32, copy=False)

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertex_data.tofile(stream)
        face_data.tofile(stream)


def checkpoint_to_mesh(args):
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise RuntimeError(
            "Mesh extraction requires scikit-image. Install it with "
            "'conda install -c conda-forge scikit-image'."
        ) from error

    splats = load_splats(args.checkpoint)
    volume, lower_xyz, spacing_xyz = build_volume(splats, args)
    if args.save_volume:
        volume_path = Path(args.output).with_suffix(".npy")
        np.save(volume_path, volume)
        print(f"saved normalized scalar volume {volume_path}")

    level = float(args.level)
    if not 0.0 < level < 1.0:
        raise ValueError("--level must be between 0 and 1")
    vertices_zyx, faces, normals_zyx, _ = marching_cubes(
        volume,
        level=level,
        spacing=tuple(spacing_xyz[::-1]),
        allow_degenerate=False,
    )
    vertices_xyz = vertices_zyx[:, ::-1] + lower_xyz
    normals_xyz = normals_zyx[:, ::-1]
    write_binary_ply(args.output, vertices_xyz, faces, normals_xyz)
    print(
        f"saved mesh {args.output}: vertices={len(vertices_xyz)}, "
        f"faces={len(faces)}, level={level}"
    )


def parse_resolution(value):
    values = [int(part) for part in value.lower().replace("x", ",").split(",")]
    if len(values) == 1:
        values *= 3
    if len(values) != 3 or any(item < 16 for item in values):
        raise argparse.ArgumentTypeError(
            "resolution must be one integer or X,Y,Z, with each value >= 16"
        )
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an UltraG-Ray checkpoint into an isosurface mesh."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=[256, 256, 256],
        help="Voxel resolution as one value or X,Y,Z (default: 256).",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["density", "echo", "attenuation", "echo_attenuation"],
        default="echo",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.08,
        help="Isosurface threshold after volume normalization (default: 0.08).",
    )
    parser.add_argument("--scale-buckets", type=int, default=4)
    parser.add_argument("--max-sigma-voxels", type=float, default=8.0)
    parser.add_argument("--crop-quantile", type=float, default=0.002)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--min-weight", type=float, default=1e-5)
    parser.add_argument("--save-volume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    checkpoint_to_mesh(parse_args())
