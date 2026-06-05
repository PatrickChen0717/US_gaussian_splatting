from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def list_image_files(image_dir):
    """Return image paths in deterministic order."""
    image_dir = Path(image_dir)
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ImageFolderDataset(Dataset):
    """
    Load grayscale or RGB images from a directory.

    If poses are omitted, each sample receives an identity 4x4 camera matrix.
    Poses may be passed as a tensor/array with shape [N, 4, 4] or as a .npy file.
    """

    def __init__(self, image_dir, poses=None, image_size=None, grayscale=False):
        self.image_paths = list_image_files(image_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.image_size = image_size
        self.grayscale = grayscale
        self.poses = self._load_poses(poses)

        if self.poses is not None and len(self.poses) != len(self.image_paths):
            raise ValueError(
                f"Pose count ({len(self.poses)}) does not match image count "
                f"({len(self.image_paths)})"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("L" if self.grayscale else "RGB")
        image = np.asarray(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image)
        if self.grayscale:
            image_tensor = image_tensor.unsqueeze(0)
        else:
            image_tensor = image_tensor.permute(2, 0, 1)

        if self.image_size is not None:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.poses is None:
            pose = torch.eye(4, dtype=torch.float32)
        else:
            pose = self.poses[idx].float()

        return {
            "image": image_tensor,
            "pose": pose,
            "path": str(self.image_paths[idx]),
        }

    @staticmethod
    def _load_poses(poses):
        if poses is None:
            return None

        if isinstance(poses, (str, Path)):
            poses = np.load(poses)

        poses = torch.as_tensor(poses, dtype=torch.float32)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("poses must have shape [N, 4, 4]")

        return poses


class UltrasoundDataset(ImageFolderDataset):
    """Backward-compatible name for ultrasound image folders."""

    def __init__(self, image_dir, camera_poses=None, image_size=None):
        super().__init__(
            image_dir=image_dir,
            poses=camera_poses,
            image_size=image_size,
            grayscale=True,
        )
