import argparse
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_PATH = "outputs/vanilla_gaussians.pt"
OUTPUT_PATH = "outputs/gaussians.ply"
OPACITY_THRESHOLD = 0.0
SCALE_X = 1.0
SCALE_Y = 1.0
SCALE_Z = 1.0
OFFSET_X = 0.0
OFFSET_Y = 0.0
OFFSET_Z = 0.0


def tensor_to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def color_to_rgb(colors):
    colors = np.clip(colors, 0.0, 1.0)
    if colors.shape[1] == 1:
        colors = np.repeat(colors, 3, axis=1)
    return np.clip(colors[:, :3] * 255.0, 0.0, 255.0).astype(np.uint8)


def write_ascii_ply(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        "comment exported from ultrasound Gaussian splatting checkpoint",
        f"element vertex {len(data['x'])}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float opacity",
        "property float scale_x",
        "property float scale_y",
        "property float scale_z",
        "property float normal_x",
        "property float normal_y",
        "property float normal_z",
        "end_header",
    ]

    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(header) + "\n")
        for i in range(len(data["x"])):
            file.write(
                f"{data['x'][i]:.8f} {data['y'][i]:.8f} {data['z'][i]:.8f} "
                f"{int(data['red'][i])} {int(data['green'][i])} {int(data['blue'][i])} "
                f"{data['opacity'][i]:.8f} "
                f"{data['scale_x'][i]:.8f} {data['scale_y'][i]:.8f} {data['scale_z'][i]:.8f} "
                f"{data['normal_x'][i]:.8f} {data['normal_y'][i]:.8f} {data['normal_z'][i]:.8f}\n"
            )


def export_ply(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state_dict"]

    means = tensor_to_numpy(state["means"])
    scales = tensor_to_numpy(torch.exp(state["log_scales"]))
    colors = color_to_rgb(tensor_to_numpy(state["colors"]))
    opacities = tensor_to_numpy(torch.sigmoid(state["logit_opacities"])).reshape(-1)

    if "disk_normals" in state:
        normals_tensor = torch.nn.functional.normalize(state["disk_normals"].float(), dim=-1, eps=1e-8)
        normals = tensor_to_numpy(normals_tensor)
    else:
        normals = np.zeros_like(means)
        normals[:, 2] = 1.0

    keep = opacities >= args.opacity_threshold
    means = means[keep]
    scales = scales[keep]
    colors = colors[keep]
    opacities = opacities[keep]
    normals = normals[keep]

    coordinate_scale = np.asarray([args.scale_x, args.scale_y, args.scale_z], dtype=np.float32)
    coordinate_offset = np.asarray([args.offset_x, args.offset_y, args.offset_z], dtype=np.float32)
    means = means * coordinate_scale[None, :] + coordinate_offset[None, :]
    scales = scales * np.abs(coordinate_scale[None, :])

    data = {
        "x": means[:, 0],
        "y": means[:, 1],
        "z": means[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "opacity": opacities,
        "scale_x": scales[:, 0],
        "scale_y": scales[:, 1],
        "scale_z": scales[:, 2],
        "normal_x": normals[:, 0],
        "normal_y": normals[:, 1],
        "normal_z": normals[:, 2],
    }
    write_ascii_ply(args.output, data)
    print(f"saved {len(means)} Gaussian vertices to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Gaussian checkpoint as a PLY point cloud.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--opacity-threshold", type=float, default=OPACITY_THRESHOLD)
    parser.add_argument("--scale-x", type=float, default=SCALE_X)
    parser.add_argument("--scale-y", type=float, default=SCALE_Y)
    parser.add_argument("--scale-z", type=float, default=SCALE_Z)
    parser.add_argument("--offset-x", type=float, default=OFFSET_X)
    parser.add_argument("--offset-y", type=float, default=OFFSET_Y)
    parser.add_argument("--offset-z", type=float, default=OFFSET_Z)
    return parser.parse_args()


if __name__ == "__main__":
    export_ply(parse_args())
