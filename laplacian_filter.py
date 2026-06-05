import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


INPUT_PATH = r"C:\Users\Patrick\Documents\MAsc\cam_capture\captures\sync_session_20260428_154459\us_images_by_aruco_pose"
OUTPUT_FOLDER = "laplacian_outputs"
USE_RGB = False
KERNEL_SIZE = 9
SIGMA = 1
FILTER_MODE = "ultrasound_edges"
MULTISCALE_SIGMAS = (0.8, 1.5, 3.0)
SOBEL_WEIGHT = 1.5
ROBUST_LOW_PERCENTILE = 1.0
ROBUST_HIGH_PERCENTILE = 99.0
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def load_image(path, grayscale=True):
    mode = "L" if grayscale else "RGB"
    image = Image.open(path).convert(mode)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)

    if grayscale:
        return tensor.unsqueeze(0).unsqueeze(0)

    return tensor.permute(2, 0, 1).unsqueeze(0)


def save_image(tensor, path):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = normalize_for_display(tensor)

    if tensor.shape[0] == 1:
        array = tensor.squeeze(0).numpy()
        image = Image.fromarray((array * 255.0).astype(np.uint8), mode="L")
    else:
        array = tensor.permute(1, 2, 0).numpy()
        image = Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def normalize_for_display(tensor):
    tensor = tensor - tensor.amin(dim=(-2, -1), keepdim=True)
    denominator = tensor.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return tensor / denominator


def robust_normalize(image, low_percentile=1.0, high_percentile=99.0):
    """Percentile normalization is useful for ultrasound gain/shadow variation."""
    low = torch.quantile(image.flatten(), low_percentile / 100.0)
    high = torch.quantile(image.flatten(), high_percentile / 100.0)
    return ((image - low) / (high - low).clamp_min(1e-8)).clamp(0.0, 1.0)


def gaussian_kernel(kernel_size, sigma, channels, device):
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32)
    coords = coords - (kernel_size - 1) * 0.5
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def gaussian_blur(image, kernel_size=9, sigma=1.5):
    channels = image.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels, image.device)
    padding = kernel_size // 2
    return F.conv2d(image, kernel, padding=padding, groups=channels)


def laplacian_filter(image):
    channels = image.shape[1]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    )
    kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return F.conv2d(image, kernel, padding=1, groups=channels)


def laplacian_pyramid_high_pass(image, kernel_size=9, sigma=1.5):
    """
    One-level Laplacian pyramid response.

    This is the local detail term: original image minus its Gaussian-smoothed
    version. It suppresses broad brightness/shadow bias and keeps edges/texture.
    """
    blurred = gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
    return image - blurred


def multiscale_laplacian_pyramid(image, sigmas=(0.8, 1.5, 3.0)):
    """Combine several high-pass scales so weak and broad US boundaries survive."""
    response = torch.zeros_like(image)
    for sigma in sigmas:
        kernel_size = max(3, int(2 * round(3 * sigma) + 1))
        detail = laplacian_pyramid_high_pass(image, kernel_size=kernel_size, sigma=sigma)
        response = response + detail.abs()
    return response / max(len(sigmas), 1)


def sobel_edges(image):
    channels = image.shape[1]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    )
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    )
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)


def ultrasound_edge_filter(image):
    """
    Ultrasound-oriented edge/detail map.

    This keeps the image grayscale, reduces global gain/shadow bias with robust
    normalization, combines multi-scale Laplacian-pyramid detail, and adds a
    Sobel gradient term for sharper boundaries.
    """
    image = robust_normalize(
        image,
        low_percentile=ROBUST_LOW_PERCENTILE,
        high_percentile=ROBUST_HIGH_PERCENTILE,
    )
    detail = multiscale_laplacian_pyramid(image, sigmas=MULTISCALE_SIGMAS)
    gradient = sobel_edges(gaussian_blur(image, kernel_size=KERNEL_SIZE, sigma=SIGMA))
    return detail + SOBEL_WEIGHT * gradient


def apply_filter(image):
    if FILTER_MODE == "laplacian_pyramid":
        return laplacian_pyramid_high_pass(image, kernel_size=KERNEL_SIZE, sigma=SIGMA).abs()
    if FILTER_MODE == "multiscale_laplacian":
        return multiscale_laplacian_pyramid(image, sigmas=MULTISCALE_SIGMAS)
    if FILTER_MODE == "sobel":
        return sobel_edges(robust_normalize(image))
    if FILTER_MODE == "ultrasound_edges":
        return ultrasound_edge_filter(image)
    raise ValueError(f"Unknown FILTER_MODE: {FILTER_MODE}")


def list_image_files(path):
    path = Path(path)
    if path.is_file():
        return [path]

    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def filter_one_image(input_path, output_folder):
    output_path = output_folder / f"{input_path.stem}_{FILTER_MODE}{input_path.suffix}"
    image = load_image(input_path, grayscale=not USE_RGB)
    filtered = apply_filter(image)
    save_image(filtered, output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Apply ultrasound Laplacian/Sobel filtering.")
    parser.add_argument("--input-path", default=INPUT_PATH)
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER)
    parser.add_argument("--filter-mode", default=FILTER_MODE)
    parser.add_argument("--kernel-size", type=int, default=KERNEL_SIZE)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--sobel-weight", type=float, default=SOBEL_WEIGHT)
    parser.add_argument("--rgb", action=argparse.BooleanOptionalAction, default=USE_RGB)
    return parser.parse_args()


def main():
    global INPUT_PATH, OUTPUT_FOLDER, FILTER_MODE, KERNEL_SIZE, SIGMA, SOBEL_WEIGHT, USE_RGB

    args = parse_args()
    INPUT_PATH = args.input_path
    OUTPUT_FOLDER = args.output_folder
    FILTER_MODE = args.filter_mode
    KERNEL_SIZE = args.kernel_size
    SIGMA = args.sigma
    SOBEL_WEIGHT = args.sobel_weight
    USE_RGB = args.rgb

    if KERNEL_SIZE % 2 == 0:
        raise ValueError("KERNEL_SIZE must be odd")

    script_dir = Path(__file__).resolve().parent
    output_folder = script_dir / OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(INPUT_PATH)
    if not image_paths:
        raise ValueError(f"No supported images found in {INPUT_PATH}")

    for image_path in image_paths:
        output_path = filter_one_image(image_path, output_folder)
        print(f"saved {output_path}")

    print(f"processed {len(image_paths)} image(s)")


if __name__ == "__main__":
    main()
