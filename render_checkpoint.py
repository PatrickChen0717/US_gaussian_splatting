import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from load_data import TrackedUltrasoundDataset
from ultrasound_losses import ultrasound_edge_map
from vanilla_gaussian_splatting import VanillaGaussianSplatting


CHECKPOINT_PATH = "outputs/vanilla_gaussians.pt"
IMAGE_DIR = "trial2.igs.mha"
POSES_PATH = None
OUTPUT_DIR = "outputs/rendered_checkpoint"
SLICE_INDEX = 0
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


def normalize_for_debug(image):
    image = image.detach().float()
    image = image - image.amin(dim=(-2, -1), keepdim=True)
    image = image / image.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return image


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


def render(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(checkpoint, device)

    height = args.height or checkpoint_value(checkpoint, "height", 128)
    width = args.width or checkpoint_value(checkpoint, "width", 128)
    grayscale = bool(checkpoint_value(checkpoint, "grayscale", True))

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
    target = sample["image"].to(device)
    pose = sample["pose"].to(device)

    calibration_kwargs = dataset.calibration.to_renderer_kwargs()
    calibration = checkpoint.get("calibration", {})
    image_t_probe = calibration.get("image_t_probe", calibration_kwargs["image_t_probe"])
    image_origin = calibration.get("image_plane_origin_px", calibration_kwargs["image_plane_origin_px"])
    pixel_to_mm = calibration.get("pixel_to_mm", calibration_kwargs["pixel_to_mm"])
    image_scale = calibration.get("image_scale", calibration_kwargs["image_scale"])

    with torch.no_grad():
        pred = model(
            pose,
            height,
            width,
            pixel_spacing=(
                checkpoint_value(checkpoint, "pixel_spacing_x", 1.0),
                checkpoint_value(checkpoint, "pixel_spacing_y", 1.0),
            ),
            slice_thickness=checkpoint_value(checkpoint, "slice_thickness", 1.0),
            shadowing=args.shadowing,
            shadow_strength=checkpoint_value(checkpoint, "shadow_strength", 1.0),
            image_t_probe=image_t_probe,
            image_plane_origin_px=tuple(image_origin),
            pixel_to_mm=pixel_to_mm,
            image_scale=image_scale,
            covariance_mode=checkpoint_value(checkpoint, "covariance_mode", "ultrasound_psf"),
            min_scale_mm=checkpoint_value(checkpoint, "min_scale_mm", 0.05),
            max_scale_mm=checkpoint_value(checkpoint, "max_scale_mm", 10.0),
            lateral_depth_slope=checkpoint_value(checkpoint, "lateral_depth_slope", 0.0),
            elevational_depth_slope=checkpoint_value(checkpoint, "elevational_depth_slope", 0.0),
            render_chunk_size=args.render_chunk_size,
            primitive_mode=checkpoint_value(checkpoint, "primitive_mode", "volume"),
            acoustic_rendering=args.acoustic_rendering,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"slice_{args.slice_index:06d}"

    pred_path = output_dir / f"{stem}_rendered.png"
    target_path = output_dir / f"{stem}_target.png"
    diff_path = output_dir / f"{stem}_absdiff.png"
    edge_path = output_dir / f"{stem}_rendered_edges.png"

    image_tensor_to_pil(pred).save(pred_path)
    image_tensor_to_pil(target).save(target_path)
    image_tensor_to_pil((pred - target).abs()).save(diff_path)

    edges = ultrasound_edge_map(
        pred,
        sigmas=(checkpoint_value(checkpoint, "filter_sigma", 1.0),),
        sobel_weight=checkpoint_value(checkpoint, "sobel_loss_weight", 1.5),
        kernel_size=checkpoint_value(checkpoint, "filter_kernel_size", 9),
        sobel_blur_kernel_size=checkpoint_value(checkpoint, "filter_kernel_size", 9),
        sobel_blur_sigma=checkpoint_value(checkpoint, "filter_sigma", 1.0),
    )
    image_tensor_to_pil(normalize_for_debug(edges.squeeze(0))).save(edge_path)

    print(f"saved rendered slice: {pred_path}")
    print(f"saved target slice:   {target_path}")
    print(f"saved abs diff:       {diff_path}")
    print(f"saved edges:          {edge_path}")
    print(f"source frame:         {sample['path']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render a trained ultrasound Gaussian checkpoint.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--poses", default=POSES_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--slice-index", type=int, default=SLICE_INDEX)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--render-chunk-size", type=int, default=8)
    parser.add_argument("--shadowing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--acoustic-rendering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
