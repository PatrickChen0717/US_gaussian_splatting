import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def resize_ultragray_images(input_path, output_path, height, width, chunk_size):
    images = np.load(input_path, mmap_mode="r")
    if images.ndim not in (3, 4):
        raise ValueError(
            f"Expected [N,H,W] or [N,H,W,C], got {images.shape}"
        )

    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(images), int(height), int(width)),
    )
    for start in range(0, len(images), int(chunk_size)):
        end = min(start + int(chunk_size), len(images))
        batch = np.asarray(images[start:end])
        if batch.ndim == 4:
            batch = batch.mean(axis=-1)
        batch = torch.from_numpy(
            batch.astype(np.float32, copy=False)
        ).unsqueeze(1)
        batch = F.interpolate(
            batch,
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        output[start:end] = (
            batch.round().clamp(0.0, 255.0).to(torch.uint8).numpy()
        )
        print(f"resized {end}/{len(images)} frames")
    output.flush()
    print(
        f"saved {len(images)} UltraG-Ray grayscale frames "
        f"with shape {height}x{width} to {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a pre-resized uint8 grayscale NPY so UltraG-Ray and the "
            "native trainer can use identical no-resize loader behavior."
        )
    )
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resize_ultragray_images(
        args.images,
        output_path,
        args.height,
        args.width,
        args.chunk_size,
    )
