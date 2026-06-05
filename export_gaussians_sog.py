import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_PATH = "outputs/vanilla_gaussians.pt"
OUTPUT_PATH = "outputs/gaussians.sog"
TEMP_PLY_PATH = "outputs/gaussians_sog_source.ply"
OPACITY_THRESHOLD = 0.0
SPLAT_TRANSFORM = "splat-transform"
SPLAT_TRANSFORM_GPU = None  # Example: "0" or "cpu".
KEEP_TEMP_PLY = False
SCALE_X = 1.0
SCALE_Y = 1.0
SCALE_Z = 1.0

SH_C0 = 0.28209479177387814


def tensor_to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def color_to_sh_dc(colors):
    colors = np.clip(colors, 0.0, 1.0)
    if colors.shape[1] == 1:
        colors = np.repeat(colors, 3, axis=1)
    return (colors[:, :3] - 0.5) / SH_C0


def normals_to_quaternions(normals):
    normals = np.asarray(normals, dtype=np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norms, 1e-8, None)

    source = np.zeros_like(normals)
    source[:, 2] = 1.0
    cross = np.cross(source, normals)
    dot = np.sum(source * normals, axis=1, keepdims=True)
    quat = np.concatenate([1.0 + dot, cross], axis=1)

    opposite = dot[:, 0] < -0.999999
    if np.any(opposite):
        quat[opposite] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / np.clip(quat_norm, 1e-8, None)


def write_standard_3dgs_ply(path, means, colors, opacities, scales, quats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(means)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    vertices = np.zeros(len(means), dtype=dtype)
    vertices["x"] = means[:, 0]
    vertices["y"] = means[:, 1]
    vertices["z"] = means[:, 2]
    vertices["f_dc_0"] = colors[:, 0]
    vertices["f_dc_1"] = colors[:, 1]
    vertices["f_dc_2"] = colors[:, 2]
    vertices["opacity"] = opacities
    vertices["scale_0"] = np.log(np.clip(scales[:, 0], 1e-8, None))
    vertices["scale_1"] = np.log(np.clip(scales[:, 1], 1e-8, None))
    vertices["scale_2"] = np.log(np.clip(scales[:, 2], 1e-8, None))
    vertices["rot_0"] = quats[:, 0]
    vertices["rot_1"] = quats[:, 1]
    vertices["rot_2"] = quats[:, 2]
    vertices["rot_3"] = quats[:, 3]

    with path.open("wb") as file:
        file.write(("\n".join(header) + "\n").encode("ascii"))
        vertices.tofile(file)


def checkpoint_to_standard_ply(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state_dict"]

    means = tensor_to_numpy(state["means"])
    scales = tensor_to_numpy(torch.exp(state["log_scales"]))
    scale_multiplier = np.asarray(
        [args.scale_x, args.scale_y, args.scale_z],
        dtype=np.float32,
    )
    if np.any(scale_multiplier <= 0.0):
        raise ValueError("--scale-x, --scale-y, and --scale-z must be greater than 0.")
    scales = scales * scale_multiplier
    colors = color_to_sh_dc(tensor_to_numpy(state["colors"]))
    opacities = tensor_to_numpy(state["logit_opacities"]).reshape(-1)

    if "disk_normals" in state:
        normals = tensor_to_numpy(torch.nn.functional.normalize(state["disk_normals"].float(), dim=-1, eps=1e-8))
    else:
        normals = np.zeros_like(means)
        normals[:, 2] = 1.0
    quats = normals_to_quaternions(normals)

    visible_opacity = 1.0 / (1.0 + np.exp(-opacities))
    keep = visible_opacity >= args.opacity_threshold

    write_standard_3dgs_ply(
        args.temp_ply,
        means[keep],
        colors[keep],
        opacities[keep],
        scales[keep],
        quats[keep],
    )
    return int(np.count_nonzero(keep))


def convert_ply_to_sog(args):
    executable = shutil.which(args.splat_transform)
    if executable is None:
        raise RuntimeError(
            f"Could not find {args.splat_transform!r}. Install PlayCanvas SplatTransform with:\n"
            "npm install -g @playcanvas/splat-transform\n"
            "Then rerun this script."
        )

    command = [executable]
    if args.gpu is not None:
        command.extend(["-g", str(args.gpu)])
    if args.overwrite:
        command.append("-w")
    command.extend([str(args.temp_ply), str(args.output)])
    subprocess.run(command, check=True)


def export_sog(args):
    count = checkpoint_to_standard_ply(args)
    convert_ply_to_sog(args)
    if not args.keep_temp_ply:
        Path(args.temp_ply).unlink(missing_ok=True)
    print(f"exported {count} Gaussians to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export checkpoint Gaussians to PlayCanvas SOG format.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--temp-ply", default=TEMP_PLY_PATH)
    parser.add_argument("--opacity-threshold", type=float, default=OPACITY_THRESHOLD)
    parser.add_argument("--splat-transform", default=SPLAT_TRANSFORM)
    parser.add_argument("--gpu", default=SPLAT_TRANSFORM_GPU, help="SplatTransform GPU adapter, e.g. 0, 1, or cpu.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output .sog if it already exists.")
    parser.add_argument("--scale-x", type=float, default=SCALE_X, help="Export-time multiplier for Gaussian x scale.")
    parser.add_argument("--scale-y", type=float, default=SCALE_Y, help="Export-time multiplier for Gaussian y scale.")
    parser.add_argument("--scale-z", type=float, default=SCALE_Z, help="Export-time multiplier for Gaussian z scale.")
    parser.add_argument("--keep-temp-ply", action=argparse.BooleanOptionalAction, default=KEEP_TEMP_PLY)
    return parser.parse_args()


if __name__ == "__main__":
    export_sog(parse_args())
