import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from load_data import TrackedUltrasoundDataset
from ultragray_cuda_backend import load_ultragray_rasterizer
from vanilla_gaussian_splatting import SourcePoseCorrection, VanillaGaussianSplatting


CHECKPOINT_PATH = "outputs/vanilla_gaussians.pt"
OUTPUT_PATH = "outputs/arbitrary_slice.png"
DEVICE = "cuda"


def image_tensor_to_pil(image):
    image = image.detach().float().cpu()
    if image.ndim == 4:
        image = image.squeeze(0)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    image = image.clamp(0.0, 1.0)

    if image.shape[0] == 1:
        array = (image.squeeze(0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    array = (image[:3].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def adjust_display_intensity(image, gain=1.0, gamma=1.0, normalize=False, percentile=None):
    image = image.detach().float()
    if normalize:
        if percentile is not None:
            high = torch.quantile(image.reshape(-1), float(percentile) / 100.0)
            low = image.amin()
            image = (image - low) / (high - low).clamp_min(1e-8)
        else:
            image = image - image.amin()
            image = image / image.amax().clamp_min(1e-8)
    image = image * float(gain)
    if gamma != 1.0:
        image = image.clamp_min(0.0).pow(1.0 / float(gamma))
    return image.clamp(0.0, 1.0)


def checkpoint_value(checkpoint, name, default=None):
    args = checkpoint.get("args", {})
    return args.get(name, default)


def load_model(checkpoint, device):
    model = VanillaGaussianSplatting(
        checkpoint["num_gaussians"],
        channels=checkpoint.get("channels", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model.to(device).eval()


def is_native_checkpoint(checkpoint):
    return "splats" in checkpoint and "model_state_dict" not in checkpoint


def load_native_metadata(checkpoint_path, explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(Path(checkpoint_path).with_suffix(".json"))
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            print(f"loaded native metadata: {path}")
            return metadata
    return {}


def native_image_size(args, metadata):
    if args.height is not None and args.width is not None:
        return int(args.height), int(args.width)
    if args.image_dir:
        frames = np.load(args.image_dir, mmap_mode="r")
        if frames.ndim not in (3, 4):
            raise ValueError(
                f"Expected image NPY shape N,H,W or N,H,W,C, got {frames.shape}"
            )
        return int(frames.shape[1]), int(frames.shape[2])
    height = metadata.get("height")
    width = metadata.get("width")
    if height is None or width is None:
        raise ValueError(
            "Native checkpoint rendering needs --image-dir, both --height and "
            "--width, or a metadata JSON containing image dimensions."
        )
    return int(height), int(width)


def native_raw_pose(args):
    if args.pose_values:
        pose = np.asarray(args.pose_values, dtype=np.float32).reshape(4, 4)
    elif args.pose_matrix:
        pose = load_matrix_file(args.pose_matrix, args.pose_array_index)
    elif args.poses:
        poses = np.load(args.poses)
        if args.slice_index < 0 or args.slice_index >= len(poses):
            raise IndexError(f"slice_index must be in [0, {len(poses) - 1}]")
        pose = np.asarray(poses[args.slice_index], dtype=np.float32)
    else:
        pose = transform_from_parameters(args.translation_mm, args.rotation_deg)
    return offset_pose(pose, args.offset_mm, args.offset_rotation_deg)


def native_scene_center(args, metadata, translation_scale, opening_width, far_plane):
    if args.native_scene_center is not None:
        return np.asarray(args.native_scene_center, dtype=np.float32)
    if "scene_center" in metadata:
        return np.asarray(metadata["scene_center"], dtype=np.float32)
    if not args.poses:
        raise ValueError(
            "Could not recover native scene centering. Provide --poses, "
            "--native-metadata, or --native-scene-center X Y Z."
        )

    poses = np.asarray(np.load(args.poses), dtype=np.float32).copy()
    poses[:, :3, 3] *= float(translation_scale)
    corners = np.asarray(
        [
            [-0.5 * opening_width, 0.0, 0.0],
            [0.5 * opening_width, 0.0, 0.0],
            [0.5 * opening_width, 0.0, far_plane],
            [-0.5 * opening_width, 0.0, far_plane],
        ],
        dtype=np.float32,
    )
    world = (
        np.einsum("nij,kj->nki", poses[:, :3, :3], corners)
        + poses[:, None, :3, 3]
    )
    return world.reshape(-1, 3).mean(axis=0)


def render_native_checkpoint(checkpoint, args, device):
    if device.type != "cuda":
        raise RuntimeError(
            "Native UltraG-Ray checkpoint rendering requires a CUDA device."
        )
    metadata = load_native_metadata(args.checkpoint, args.native_metadata)
    height, width = native_image_size(args, metadata)
    far_plane = float(metadata.get("far_plane", args.ultrasound_far_plane))
    opening_width = float(
        metadata.get("opening_width", args.ultrasound_opening_width)
    )
    translation_scale = float(
        metadata.get(
            "pose_translation_scale",
            args.native_pose_translation_scale,
        )
    )
    scene_center = native_scene_center(
        args,
        metadata,
        translation_scale,
        opening_width,
        far_plane,
    )
    pose = native_raw_pose(args)
    pose[:3, 3] *= translation_scale
    pose[:3, 3] -= scene_center
    camera_to_world = torch.as_tensor(
        pose,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    splats = {
        name: value.detach().to(device)
        for name, value in checkpoint["splats"].items()
    }
    intensities = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    native_shadowing = args.shadowing is not False
    transmittances = torch.sigmoid(splats["transmittances"])
    if not native_shadowing:
        transmittances = torch.ones_like(transmittances)
    sh_degree = (
        int(args.native_sh_degree)
        if args.native_sh_degree is not None
        else max(int(round(math.sqrt(intensities.shape[1]) - 1)), 0)
    )
    rasterizer = load_ultragray_rasterizer(args.ultragray_repo_path)
    render, _, _, _, _ = rasterizer(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        transmittances=transmittances,
        intensities=intensities,
        viewmats=torch.linalg.inv(camera_to_world),
        width=width,
        height=height,
        near_plane=0.0,
        far_plane=far_plane,
        opening_angle=None,
        opening_width=opening_width,
        tile_size_x=int(args.cuda_tile_size_x),
        tile_size_y=int(args.cuda_tile_size_y),
        sh_degree=sh_degree,
    )
    print(
        "native render geometry: "
        f"image={height}x{width}, opening_width={opening_width}, "
        f"far_plane={far_plane}, shadowing={native_shadowing}, "
        f"scene_center={scene_center.tolist()}"
    )
    return render[0].permute(2, 0, 1), pose, height, width


def load_pose_correction(checkpoint, device):
    state_dict = checkpoint.get("pose_correction_state_dict")
    metadata = checkpoint.get("pose_correction")
    if not state_dict or not metadata or metadata.get("mode") == "none":
        return None

    raw_translation = state_dict.get("raw_translation")
    if raw_translation is None:
        return None

    correction = SourcePoseCorrection(
        source_count=int(raw_translation.shape[0]),
        max_translation_mm=float(metadata.get("max_translation_mm", 2.0)),
        max_rotation_deg=float(metadata.get("max_rotation_deg", 2.0)),
    )
    correction.load_state_dict(state_dict)
    return correction.to(device).eval()


def load_matrix_file(path, pose_array_index=0):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            matrix = np.asarray(json.load(handle), dtype=np.float32)
    else:
        try:
            matrix = np.loadtxt(path, delimiter=",", dtype=np.float32)
        except ValueError:
            matrix = np.loadtxt(path, dtype=np.float32)

    if matrix.shape == (4, 4):
        return matrix.astype(np.float32)

    if matrix.ndim == 3 and matrix.shape[1:] == (4, 4):
        if pose_array_index < 0 or pose_array_index >= matrix.shape[0]:
            raise IndexError(f"pose_array_index must be in [0, {matrix.shape[0] - 1}]")
        return matrix[pose_array_index].astype(np.float32)

    raise ValueError(f"Expected a 4x4 pose matrix or Nx4x4 pose array, got shape {matrix.shape}")


def load_ascii_ply_points(path, max_points=50000, seed=1234):
    path = Path(path)
    vertex_count = None
    header_lines = 0
    properties = []
    in_vertex = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
                in_vertex = True
                continue
            if stripped.startswith("element ") and not stripped.startswith("element vertex"):
                in_vertex = False
            if in_vertex and stripped.startswith("property"):
                properties.append(stripped.split()[-1])
            if stripped == "end_header":
                break

    if vertex_count is None:
        raise ValueError(f"Could not find vertex count in PLY header: {path}")

    data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]

    columns = {name: index for index, name in enumerate(properties)}
    points = data[:, [columns["x"], columns["y"], columns["z"]]]
    colors = None
    if {"red", "green", "blue"}.issubset(columns):
        colors = data[:, [columns["red"], columns["green"], columns["blue"]]] / 255.0

    opacity = None
    if "opacity" in columns:
        opacity = data[:, columns["opacity"]]

    if max_points and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
        if opacity is not None:
            opacity = opacity[keep]

    return points, colors, opacity


def rotation_matrix_xyz(rotation_deg):
    rx, ry, rz = np.deg2rad(np.asarray(rotation_deg, dtype=np.float32))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float32,
    )
    rot_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rot_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return rot_z @ rot_y @ rot_x


def transform_from_parameters(translation_mm, rotation_deg):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation_matrix_xyz(rotation_deg)
    pose[:3, 3] = np.asarray(translation_mm, dtype=np.float32)
    return pose


def offset_pose(pose, offset_mm, offset_rotation_deg):
    delta = transform_from_parameters(offset_mm, offset_rotation_deg)
    return pose @ delta


def load_dataset_pose(args, height, width, grayscale, checkpoint):
    dataset = TrackedUltrasoundDataset(
        image_dir=args.image_dir,
        poses_path=args.poses,
        image_size=(height, width),
        grayscale=grayscale,
        low_intensity_threshold=checkpoint_value(checkpoint, "low_intensity_threshold", None),
    )
    if args.slice_index < 0 or args.slice_index >= len(dataset):
        raise IndexError(f"slice_index must be in [0, {len(dataset) - 1}]")
    sample = dataset[args.slice_index]
    return sample["pose"].numpy().astype(np.float32), dataset.calibration.to_renderer_kwargs()


def resolve_pose(args, height, width, grayscale, checkpoint):
    dataset_calibration = None
    if args.pose_values:
        pose = np.asarray(args.pose_values, dtype=np.float32).reshape(4, 4)
    elif args.pose_matrix:
        pose = load_matrix_file(args.pose_matrix, args.pose_array_index)
    elif args.image_dir:
        pose, dataset_calibration = load_dataset_pose(args, height, width, grayscale, checkpoint)
    else:
        pose = transform_from_parameters(args.translation_mm, args.rotation_deg)

    pose = offset_pose(pose, args.offset_mm, args.offset_rotation_deg)
    return pose, dataset_calibration


def resolve_calibration(checkpoint, dataset_calibration=None):
    fallback = {
        "image_t_probe": np.eye(4, dtype=np.float32),
        "image_plane_origin_px": (0.0, 0.0),
        "pixel_to_mm": 1.0,
        "image_scale": 1.0,
    }
    if dataset_calibration:
        fallback.update(dataset_calibration)

    calibration = checkpoint.get("calibration", {})
    image_t_probe = calibration.get("image_t_probe", fallback["image_t_probe"])
    image_origin = calibration.get("image_plane_origin_px", fallback["image_plane_origin_px"])
    pixel_to_mm = calibration.get("pixel_to_mm", fallback["pixel_to_mm"])
    image_scale = calibration.get("image_scale", fallback["image_scale"])
    return image_t_probe, image_origin, pixel_to_mm, image_scale


def slice_corners_world(
    pose,
    height,
    width,
    image_t_probe,
    image_plane_origin_px,
    pixel_to_mm,
    image_scale,
    pixel_spacing=(1.0, 1.0),
):
    origin_px = np.asarray(image_plane_origin_px, dtype=np.float32)
    calibrated_spacing = float(pixel_to_mm) * float(image_scale)
    if calibrated_spacing > 0.0:
        spacing_x = calibrated_spacing
        spacing_y = calibrated_spacing
    else:
        spacing_x, spacing_y = pixel_spacing

    pixel_corners = np.asarray(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )
    image_points = np.ones((4, 4), dtype=np.float32)
    image_points[:, 0] = (pixel_corners[:, 0] - origin_px[0]) * spacing_x
    image_points[:, 1] = (pixel_corners[:, 1] - origin_px[1]) * spacing_y
    image_points[:, 2] = 0.0

    image_t_probe = np.asarray(image_t_probe, dtype=np.float32)
    probe_t_image = np.linalg.inv(image_t_probe)
    probe_points = (probe_t_image @ image_points.T).T
    world_points = (np.asarray(pose, dtype=np.float32) @ probe_points.T).T[:, :3]
    return world_points


def infer_scene_ply_path(args):
    if args.scene_ply:
        return Path(args.scene_ply)
    checkpoint_path = Path(args.checkpoint)
    inferred = checkpoint_path.with_suffix(".ply")
    if inferred.exists():
        return inferred
    raise FileNotFoundError(
        f"Could not infer scene PLY from checkpoint path. Expected {inferred}; pass --scene-ply explicitly."
    )


def render_scene_preview(
    args,
    checkpoint,
    pose,
    height,
    width,
    image_t_probe,
    image_origin,
    pixel_to_mm,
    image_scale,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ply_path = infer_scene_ply_path(args)
    points, colors, opacity = load_ascii_ply_points(
        ply_path,
        max_points=args.scene_max_points,
        seed=args.scene_seed,
    )

    if opacity is not None:
        threshold = float(args.scene_opacity_threshold)
        keep = opacity >= threshold
        if keep.any():
            points = points[keep]
            if colors is not None:
                colors = colors[keep]

    corners = slice_corners_world(
        pose,
        height,
        width,
        image_t_probe,
        image_origin,
        pixel_to_mm,
        image_scale,
        pixel_spacing=(
            checkpoint_value(checkpoint, "pixel_spacing_x", 1.0),
            checkpoint_value(checkpoint, "pixel_spacing_y", 1.0),
        ),
    )
    center = corners.mean(axis=0)
    edge_a = corners[1] - corners[0]
    edge_b = corners[3] - corners[0]
    normal = np.cross(edge_a, edge_b)
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    arrow_length = max(np.linalg.norm(edge_a), np.linalg.norm(edge_b)) * 0.35

    scene_output = Path(args.scene_output) if args.scene_output else Path(args.output).with_name(
        f"{Path(args.output).stem}_scene.png"
    )
    scene_output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    point_colors = colors if colors is not None else np.full((len(points), 3), 0.35)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=point_colors,
        s=float(args.scene_point_size),
        alpha=float(args.scene_point_alpha),
        depthshade=False,
    )

    plane = Poly3DCollection(
        [corners],
        facecolors=(0.0, 0.85, 0.95, 0.16),
        edgecolors=(0.0, 0.75, 0.85, 1.0),
        linewidths=2.5,
    )
    ax.add_collection3d(plane)
    closed = np.vstack([corners, corners[0]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="cyan", linewidth=3.0)
    ax.quiver(
        center[0],
        center[1],
        center[2],
        normal[0],
        normal[1],
        normal[2],
        length=arrow_length,
        color="magenta",
        linewidth=2.5,
        normalize=True,
    )
    ax.scatter([center[0]], [center[1]], [center[2]], color="magenta", s=30)

    all_points = np.vstack([points, corners])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center_bounds = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    radius = max(radius, 1.0)
    ax.set_xlim(center_bounds[0] - radius, center_bounds[0] + radius)
    ax.set_ylim(center_bounds[1] - radius, center_bounds[1] + radius)
    ax.set_zlim(center_bounds[2] - radius, center_bounds[2] + radius)
    ax.view_init(elev=args.scene_elev, azim=args.scene_azim)
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.set_title("Rendered Slice Plane in Gaussian Point Cloud")
    fig.tight_layout()
    fig.savefig(scene_output, dpi=args.scene_dpi)
    plt.close(fig)
    print(f"saved scene: {scene_output}")


def render_native_scene_preview(
    args,
    checkpoint,
    camera_to_world,
    opening_width,
    far_plane,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    means = checkpoint["splats"]["means"].detach().float().cpu().numpy()
    sh0 = checkpoint["splats"].get("sh0")
    colors = None
    if sh0 is not None:
        intensity = (
            sh0.detach().float().reshape(len(means), -1).mean(dim=1)
            * 0.28209479177387814
            + 0.5
        ).clamp(0.0, 1.0).cpu().numpy()
        colors = np.repeat(intensity[:, None], 3, axis=1)

    if args.scene_max_points and len(means) > int(args.scene_max_points):
        rng = np.random.default_rng(args.scene_seed)
        keep = rng.choice(
            len(means),
            size=int(args.scene_max_points),
            replace=False,
        )
        means = means[keep]
        if colors is not None:
            colors = colors[keep]

    local_corners = np.asarray(
        [
            [-0.5 * opening_width, 0.0, 0.0, 1.0],
            [0.5 * opening_width, 0.0, 0.0, 1.0],
            [0.5 * opening_width, 0.0, far_plane, 1.0],
            [-0.5 * opening_width, 0.0, far_plane, 1.0],
        ],
        dtype=np.float32,
    )
    corners = (
        np.asarray(camera_to_world, dtype=np.float32) @ local_corners.T
    ).T[:, :3]
    center = corners.mean(axis=0)
    normal = np.asarray(camera_to_world, dtype=np.float32)[:3, 1]
    normal /= max(np.linalg.norm(normal), 1e-8)
    arrow_length = max(float(opening_width), float(far_plane)) * 0.35

    scene_output = (
        Path(args.scene_output)
        if args.scene_output
        else Path(args.output).with_name(f"{Path(args.output).stem}_scene.png")
    )
    scene_output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    point_colors = colors if colors is not None else np.full((len(means), 3), 0.35)
    ax.scatter(
        means[:, 0],
        means[:, 1],
        means[:, 2],
        c=point_colors,
        s=float(args.scene_point_size),
        alpha=float(args.scene_point_alpha),
        depthshade=False,
    )
    plane = Poly3DCollection(
        [corners],
        facecolors=(0.0, 0.85, 0.95, 0.16),
        edgecolors=(0.0, 0.75, 0.85, 1.0),
        linewidths=2.5,
    )
    ax.add_collection3d(plane)
    closed = np.vstack([corners, corners[0]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="cyan", linewidth=3.0)
    ax.quiver(
        center[0],
        center[1],
        center[2],
        normal[0],
        normal[1],
        normal[2],
        length=arrow_length,
        color="magenta",
        linewidth=2.5,
        normalize=True,
    )

    all_points = np.vstack([means, corners])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    bounds_center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins) * 0.55), 0.1)
    ax.set_xlim(bounds_center[0] - radius, bounds_center[0] + radius)
    ax.set_ylim(bounds_center[1] - radius, bounds_center[1] + radius)
    ax.set_zlim(bounds_center[2] - radius, bounds_center[2] + radius)
    ax.view_init(elev=args.scene_elev, azim=args.scene_azim)
    ax.set_xlabel("X cm")
    ax.set_ylabel("Y cm")
    ax.set_zlabel("Z cm")
    ax.set_title("Rendered Slice Plane in Native UltraG-Ray Gaussians")
    fig.tight_layout()
    fig.savefig(scene_output, dpi=args.scene_dpi)
    plt.close(fig)
    print(f"saved scene: {scene_output}")


def render(args):
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    requested_device = args.device
    device = torch.device(requested_device if torch.cuda.is_available() or requested_device == "cpu" else "cpu")

    if is_native_checkpoint(checkpoint):
        with torch.no_grad():
            pred, pose_np, _, _ = render_native_checkpoint(
                checkpoint,
                args,
                device,
            )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        display_pred = adjust_display_intensity(
            pred,
            gain=args.display_gain,
            gamma=args.display_gamma,
            normalize=args.display_normalize,
            percentile=args.display_percentile,
        )
        image_tensor_to_pil(display_pred).save(output_path)
        if args.save_npy:
            np.save(output_path.with_suffix(".npy"), pred.cpu().numpy())
        if args.save_pose:
            np.save(
                output_path.with_name(f"{output_path.stem}_pose.npy"),
                pose_np,
            )
        if args.scene:
            metadata = load_native_metadata(
                args.checkpoint,
                args.native_metadata,
            )
            render_native_scene_preview(
                args,
                checkpoint,
                pose_np,
                float(
                    metadata.get(
                        "opening_width",
                        args.ultrasound_opening_width,
                    )
                ),
                float(
                    metadata.get(
                        "far_plane",
                        args.ultrasound_far_plane,
                    )
                ),
            )
        print(f"saved image: {output_path}")
        print(f"pose:\n{pose_np}")
        return

    model = load_model(checkpoint, device)
    pose_correction = load_pose_correction(checkpoint, device)

    height = args.height or checkpoint_value(checkpoint, "height", 128)
    width = args.width or checkpoint_value(checkpoint, "width", 128)
    grayscale = bool(checkpoint_value(checkpoint, "grayscale", True))
    pose_np, dataset_calibration = resolve_pose(args, height, width, grayscale, checkpoint)
    pose = torch.as_tensor(pose_np, dtype=torch.float32, device=device)

    if args.apply_pose_correction:
        if pose_correction is None:
            raise ValueError("Checkpoint does not contain a usable source pose correction module.")
        if args.source_index is None:
            raise ValueError("--source-index is required with --apply-pose-correction.")
        source_index = torch.tensor(args.source_index, dtype=torch.long, device=device)
        pose = pose_correction(pose, source_index)

    image_t_probe, image_origin, pixel_to_mm, image_scale = resolve_calibration(
        checkpoint,
        dataset_calibration=dataset_calibration,
    )
    resolved_pixel_to_mm = args.pixel_to_mm if args.pixel_to_mm is not None else pixel_to_mm
    resolved_image_scale = args.image_scale if args.image_scale is not None else image_scale

    with torch.no_grad():
        pred = model(
            pose,
            height,
            width,
            pixel_spacing=(
                checkpoint_value(checkpoint, "pixel_spacing_x", 1.0),
                checkpoint_value(checkpoint, "pixel_spacing_y", 1.0),
            ),
            slice_thickness=args.slice_thickness
            if args.slice_thickness is not None
            else checkpoint_value(checkpoint, "slice_thickness", 1.0),
            shadowing=bool(args.shadowing),
            shadow_strength=checkpoint_value(checkpoint, "shadow_strength", 1.0),
            image_t_probe=image_t_probe,
            image_plane_origin_px=tuple(image_origin),
            pixel_to_mm=resolved_pixel_to_mm,
            image_scale=resolved_image_scale,
            covariance_mode=checkpoint_value(checkpoint, "covariance_mode", "ultrasound_psf"),
            min_scale_mm=checkpoint_value(checkpoint, "min_scale_mm", 0.05),
            max_scale_mm=checkpoint_value(checkpoint, "max_scale_mm", 10.0),
            lateral_depth_slope=checkpoint_value(checkpoint, "lateral_depth_slope", 0.0),
            elevational_depth_slope=checkpoint_value(checkpoint, "elevational_depth_slope", 0.0),
            render_chunk_size=args.render_chunk_size,
            primitive_mode=checkpoint_value(checkpoint, "primitive_mode", "volume"),
            acoustic_rendering=args.acoustic_rendering,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_pred = adjust_display_intensity(
        pred,
        gain=args.display_gain,
        gamma=args.display_gamma,
        normalize=args.display_normalize,
        percentile=args.display_percentile,
    )
    image_tensor_to_pil(display_pred).save(output_path)

    if args.save_npy:
        npy_path = output_path.with_suffix(".npy")
        np.save(npy_path, pred.detach().float().cpu().numpy())
        print(f"saved array: {npy_path}")

    if args.save_pose:
        pose_path = output_path.with_name(f"{output_path.stem}_pose.npy")
        np.save(pose_path, pose.detach().float().cpu().numpy())
        print(f"saved pose:  {pose_path}")

    if args.scene:
        render_scene_preview(
            args,
            checkpoint,
            pose.detach().float().cpu().numpy(),
            height,
            width,
            image_t_probe,
            image_origin,
            resolved_pixel_to_mm,
            resolved_image_scale,
        )

    print(f"saved image: {output_path}")
    print(f"pose:\n{pose.detach().float().cpu().numpy()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an arbitrary ultrasound slice from a trained Gaussian checkpoint."
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--render-chunk-size", type=int, default=8)
    parser.add_argument("--native-metadata", default=None)
    parser.add_argument("--native-scene-center", type=float, nargs=3, default=None)
    parser.add_argument("--native-pose-translation-scale", type=float, default=0.1)
    parser.add_argument("--ultrasound-far-plane", type=float, default=9.0)
    parser.add_argument("--ultrasound-opening-width", type=float, default=5.13)
    parser.add_argument("--ultragray-repo-path", default="../UltraG-Ray")
    parser.add_argument("--cuda-tile-size-x", type=int, default=4)
    parser.add_argument("--cuda-tile-size-y", type=int, default=128)
    parser.add_argument("--native-sh-degree", type=int, default=None)

    pose_group = parser.add_argument_group("probe plane pose")
    pose_group.add_argument("--pose-matrix", default=None, help="Path to a 4x4 or Nx4x4 pose matrix in npy/json/csv/txt.")
    pose_group.add_argument(
        "--pose-values",
        type=float,
        nargs=16,
        default=None,
        help="Inline row-major 4x4 pose matrix values.",
    )
    pose_group.add_argument("--pose-array-index", type=int, default=0)
    pose_group.add_argument("--translation-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    pose_group.add_argument("--rotation-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    pose_group.add_argument("--offset-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    pose_group.add_argument("--offset-rotation-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))

    dataset_group = parser.add_argument_group("optional dataset pose source")
    dataset_group.add_argument("--image-dir", default=None, help="Use a dataset slice pose as the base plane.")
    dataset_group.add_argument("--poses", default=None)
    dataset_group.add_argument("--slice-index", type=int, default=0)

    correction_group = parser.add_argument_group("optional learned source correction")
    correction_group.add_argument("--apply-pose-correction", action="store_true")
    correction_group.add_argument("--source-index", type=int, default=None)

    render_group = parser.add_argument_group("render settings")
    render_group.add_argument("--slice-thickness", type=float, default=None)
    render_group.add_argument("--pixel-to-mm", type=float, default=None)
    render_group.add_argument("--image-scale", type=float, default=None)
    render_group.add_argument(
        "--shadowing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Control learned attenuation. Native UltraG-Ray checkpoints "
            "default to enabled; older checkpoints default to disabled."
        ),
    )
    render_group.add_argument("--acoustic-rendering", action=argparse.BooleanOptionalAction, default=True)
    render_group.add_argument("--save-npy", action="store_true")
    render_group.add_argument("--save-pose", action="store_true")
    render_group.add_argument("--display-gain", type=float, default=1.0)
    render_group.add_argument("--display-gamma", type=float, default=1.0)
    render_group.add_argument("--display-normalize", action="store_true")
    render_group.add_argument(
        "--display-percentile",
        type=float,
        default=None,
        help="When normalizing, map this intensity percentile to white instead of using the absolute max.",
    )

    scene_group = parser.add_argument_group("optional 3D scene screenshot")
    scene_group.add_argument("--scene", action="store_true", help="Save a 3D screenshot of the slice plane in a PLY cloud.")
    scene_group.add_argument("--scene-ply", default=None, help="PLY cloud to draw behind the slice plane.")
    scene_group.add_argument("--scene-output", default=None)
    scene_group.add_argument("--scene-max-points", type=int, default=50000)
    scene_group.add_argument("--scene-opacity-threshold", type=float, default=0.0)
    scene_group.add_argument("--scene-point-size", type=float, default=1.0)
    scene_group.add_argument("--scene-point-alpha", type=float, default=0.22)
    scene_group.add_argument("--scene-seed", type=int, default=1234)
    scene_group.add_argument("--scene-elev", type=float, default=24.0)
    scene_group.add_argument("--scene-azim", type=float, default=-55.0)
    scene_group.add_argument("--scene-dpi", type=int, default=160)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
