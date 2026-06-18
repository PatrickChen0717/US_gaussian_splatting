from dataclasses import dataclass, field
from pathlib import Path
import csv
import math
import re
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# Edit these paths for your data.
IMAGE_DIR = r"C:\Users\Patrick\Documents\MAsc\EchoRaccoon\dataset\trial2.igs.mha"
POSES_PATH = None  # .npy [N, 4, 4] or EchoRaccoon ProbeToTracker .csv. MHA can provide poses directly.
# POSES_PATH = r"C:\Users\Patrick\Documents\MAsc\EchoRaccoon\dataset\trial2_out\trial2_frames.npy"

# Edit these calibration values manually.
DEFAULT_IMAGE_T_PROBE = np.asarray(
    [
        [0.0, 1.0, 0.0, -0.0],
        [1.0, -0.0, 0.0, -75.5],
        [0.0, -0.0, -1.0, -14.34],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DEFAULT_IMAGE_PLANE_ORIGIN_PX = np.asarray([299.0, 5.0], dtype=np.float64)
IMAGE_DEPTH_MM = 160.0
VIDEO_HEIGHT_PX = 726.0
IMAGE_SCALE = 1.0

IMAGE_SIZE = None  # Example: (512, 512), or None to keep original size.
GRAYSCALE = True
ONLY_OK_POSES = True
LOWPASS_GRAYSCALE_THRESHOLD = None  # Example: 25 removes pixels below grayscale value 25 on a 0-255 scale.
LOW_INTENSITY_THRESHOLD = LOWPASS_GRAYSCALE_THRESHOLD
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MATRIX_FIELDS = [f"m{row}{col}" for row in range(4) for col in range(4)]
FRAME_KEY_RE = re.compile(r"^Seq_Frame(\d+)_(.+)$")
MHA_EXTENSIONS = (".mha", ".igs.mha")
METAIMAGE_DTYPES = {
    "MET_CHAR": np.int8,
    "MET_UCHAR": np.uint8,
    "MET_SHORT": np.int16,
    "MET_USHORT": np.uint16,
    "MET_INT": np.int32,
    "MET_UINT": np.uint32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


@dataclass(frozen=True)
class UltrasoundCalibration:
    image_t_probe: np.ndarray = field(default_factory=lambda: DEFAULT_IMAGE_T_PROBE.copy())
    image_plane_origin_px: np.ndarray = field(
        default_factory=lambda: DEFAULT_IMAGE_PLANE_ORIGIN_PX.copy()
    )
    image_depth_mm: float = IMAGE_DEPTH_MM
    video_height_px: float = VIDEO_HEIGHT_PX
    image_scale: float = IMAGE_SCALE

    @property
    def pixel_to_mm(self):
        return self.image_depth_mm / self.video_height_px

    @property
    def probe_t_image(self):
        return np.linalg.inv(self.image_t_probe)

    def image_points_in_probe(self, rows, cols, keep_grid_shape=False):
        image_x = (
            (cols.astype(np.float64) - self.image_plane_origin_px[0])
            * self.pixel_to_mm
            * self.image_scale
        )
        image_y = (
            (rows.astype(np.float64) - self.image_plane_origin_px[1])
            * self.pixel_to_mm
            * self.image_scale
        )
        grid_x, grid_y = np.meshgrid(image_x, image_y)
        image_points = np.stack(
            [
                grid_x.reshape(-1),
                grid_y.reshape(-1),
                np.zeros(grid_x.size, dtype=np.float64),
                np.ones(grid_x.size, dtype=np.float64),
            ]
        )
        probe_points = self.probe_t_image @ image_points
        if keep_grid_shape:
            return probe_points.reshape(4, len(rows), len(cols))
        return probe_points

    def image_corners_in_probe(self, image_shape):
        height, width = image_shape[:2]
        image_x = (
            np.asarray([0.0, width - 1.0, width - 1.0, 0.0])
            - self.image_plane_origin_px[0]
        ) * (self.pixel_to_mm * self.image_scale)
        image_y = (
            np.asarray([0.0, 0.0, height - 1.0, height - 1.0])
            - self.image_plane_origin_px[1]
        ) * (self.pixel_to_mm * self.image_scale)
        image_corners = np.asarray(
            [
                [image_x[0], image_y[0], 0.0, 1.0],
                [image_x[1], image_y[1], 0.0, 1.0],
                [image_x[2], image_y[2], 0.0, 1.0],
                [image_x[3], image_y[3], 0.0, 1.0],
            ],
            dtype=np.float64,
        ).T
        return self.probe_t_image @ image_corners

    def image_normal_in_probe(self):
        normal = self.probe_t_image[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            raise ValueError("Image-to-probe transform produced a zero image normal.")
        return normal / norm

    def to_renderer_kwargs(self):
        return {
            "image_t_probe": self.image_t_probe,
            "image_plane_origin_px": tuple(self.image_plane_origin_px.tolist()),
            "pixel_to_mm": float(self.pixel_to_mm),
            "image_scale": float(self.image_scale),
        }


def list_image_files(image_dir):
    image_dir = Path(image_dir)
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def is_mha_path(path):
    name = str(path).lower()
    return any(name.endswith(extension) for extension in MHA_EXTENSIONS)


def read_local_mha(path):
    header = {}
    with Path(path).open("rb") as file:
        while True:
            line = file.readline()
            if not line:
                raise ValueError("Reached end of file before ElementDataFile = LOCAL")

            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded and "=" in decoded:
                key, value = decoded.split("=", 1)
                header[key.strip()] = value.strip()

            if decoded.lower() == "elementdatafile = local":
                break

        payload = file.read()

    if header.get("ElementDataFile", "").upper() != "LOCAL":
        raise ValueError("Only single-file MHA files with ElementDataFile = LOCAL are supported")
    return header, payload


def parse_bool(value):
    return value.strip().lower() in {"true", "1", "yes"}


def parse_ints(value):
    return [int(part) for part in value.split()]


def parse_float(value):
    cleaned = value.strip().lower()
    if cleaned in {"nan", "+nan", "-nan", "nan(ind)", "+nan(ind)", "-nan(ind)"}:
        return math.nan
    return float(value)


def parse_matrix(value):
    values = [parse_float(part) for part in value.split()]
    if len(values) != 16:
        raise ValueError(f"Expected 16 transform values, got {len(values)}")
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def frame_metadata(header):
    frames = {}
    for key, value in header.items():
        match = FRAME_KEY_RE.match(key)
        if not match:
            continue
        frame_index = int(match.group(1))
        field = match.group(2)
        frames.setdefault(frame_index, {})[field] = value
    return frames


def decode_mha_frames(header, payload):
    dim_size = parse_ints(header["DimSize"])
    if len(dim_size) < 3:
        raise ValueError(f"Expected a 3D image sequence in DimSize, got {header['DimSize']}")

    element_type = header["ElementType"].strip()
    if element_type not in METAIMAGE_DTYPES:
        raise ValueError(f"Unsupported ElementType {element_type!r}")

    dtype = np.dtype(METAIMAGE_DTYPES[element_type])
    dtype = dtype.newbyteorder(">" if parse_bool(header.get("BinaryDataByteOrderMSB", "False")) else "<")

    if parse_bool(header.get("CompressedData", "False")):
        payload = zlib.decompress(payload)

    channels = int(header.get("ElementNumberOfChannels", "1"))
    expected_values = int(np.prod(dim_size)) * channels
    frames = np.frombuffer(payload, dtype=dtype, count=expected_values)
    if frames.size != expected_values:
        raise ValueError(f"Expected {expected_values} voxels, decoded {frames.size}")

    width, height, frame_count = dim_size[:3]
    if channels == 1:
        frames = frames.reshape((frame_count, height, width))
    else:
        frames = frames.reshape((frame_count, height, width, channels))
    return np.ascontiguousarray(frames)


def load_mha_sequence(path, only_ok=True):
    header, payload = read_local_mha(path)
    frames = decode_mha_frames(header, payload)
    metadata_by_frame = frame_metadata(header)
    transforms = np.full((len(frames), 4, 4), np.nan, dtype=np.float64)
    statuses = []
    timestamps = np.full(len(frames), np.nan, dtype=np.float64)

    for frame_index in range(len(frames)):
        metadata = metadata_by_frame.get(frame_index, {})
        transform_value = metadata.get("ProbeToTrackerTransform")
        if transform_value:
            transforms[frame_index] = parse_matrix(transform_value)
        status = metadata.get("ProbeToTrackerTransformStatus")
        statuses.append(status)
        timestamp = metadata.get("Timestamp")
        if timestamp:
            timestamps[frame_index] = parse_float(timestamp)

    valid = np.all(np.isfinite(transforms), axis=(1, 2))
    if only_ok:
        valid &= np.asarray([status == "OK" for status in statuses], dtype=bool)
    return frames[valid], transforms[valid].astype(np.float32), timestamps[valid]


def load_pose_file(path, only_ok=True):
    if path is None:
        return None

    path = Path(path)
    if path.suffix.lower() == ".npy":
        poses = np.load(path)
    elif path.suffix.lower() == ".csv":
        poses = load_probe_to_tracker_csv(path, only_ok=only_ok)
    else:
        raise ValueError(f"Unsupported pose file type: {path.suffix}")

    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [N, 4, 4]")
    return poses


def load_probe_to_tracker_csv(csv_path, only_ok=True):
    transforms = []
    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [field for field in MATRIX_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing transform fields: {', '.join(missing)}")

        for row in reader:
            if only_ok and row.get("status") != "OK":
                continue
            matrix = np.asarray([float(row[field]) for field in MATRIX_FIELDS], dtype=np.float64)
            matrix = matrix.reshape(4, 4)
            if np.all(np.isfinite(matrix)):
                transforms.append(matrix)

    if not transforms:
        raise ValueError(f"No valid poses found in {csv_path}")
    return np.stack(transforms)


def remove_pixels_under_grayscale_value(tensor, grayscale_value_255=None):
    if grayscale_value_255 is None:
        return tensor
    grayscale_value = float(grayscale_value_255) / 255.0
    if tensor.shape[0] == 1:
        gray = tensor
    else:
        gray = (
            0.299 * tensor[0:1]
            + 0.587 * tensor[1:2]
            + 0.114 * tensor[2:3]
        )
    keep_mask = gray >= grayscale_value
    return tensor * keep_mask.to(tensor.dtype)


def load_image_tensor(path, grayscale=True, image_size=None, low_intensity_threshold=LOW_INTENSITY_THRESHOLD):
    mode = "L" if grayscale else "RGB"
    image = Image.open(path).convert(mode)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)

    if grayscale:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.permute(2, 0, 1)

    if image_size is not None:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    tensor = remove_pixels_under_grayscale_value(tensor, low_intensity_threshold)
    return tensor


def frame_to_tensor(frame, grayscale=True, image_size=None, low_intensity_threshold=LOW_INTENSITY_THRESHOLD):
    frame = np.asarray(frame)
    frame_float = frame.astype(np.float32, copy=False)
    if np.issubdtype(frame.dtype, np.integer):
        frame_float = frame_float / float(np.iinfo(frame.dtype).max)
    elif frame_float.size and float(np.nanmax(frame_float)) > 1.0:
        frame_float = frame_float / 255.0
    frame_float = np.nan_to_num(frame_float, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

    if frame_float.ndim == 2:
        tensor = torch.from_numpy(frame_float).unsqueeze(0)
    elif grayscale:
        if frame_float.shape[-1] >= 3:
            gray = (
                0.299 * frame_float[..., 0]
                + 0.587 * frame_float[..., 1]
                + 0.114 * frame_float[..., 2]
            )
        else:
            gray = frame_float.mean(axis=-1)
        tensor = torch.from_numpy(gray.astype(np.float32, copy=False)).unsqueeze(0)
    else:
        tensor = torch.from_numpy(frame_float).permute(2, 0, 1)

    if image_size is not None:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    tensor = remove_pixels_under_grayscale_value(tensor, low_intensity_threshold)
    return tensor.float()


class TrackedUltrasoundDataset(Dataset):
    def __init__(
        self,
        image_dir=IMAGE_DIR,
        poses_path=POSES_PATH,
        calibration=None,
        image_size=IMAGE_SIZE,
        grayscale=GRAYSCALE,
        only_ok_poses=ONLY_OK_POSES,
        low_intensity_threshold=LOW_INTENSITY_THRESHOLD,
    ):
        image_source = Path(image_dir)
        self.frames = None
        self.image_paths = None
        self.source_name = str(image_source)

        if image_source.is_file() and image_source.suffix.lower() == ".npy":
            self.frames = np.load(image_source)
            if self.frames.ndim not in (3, 4):
                raise ValueError(
                    f"Expected .npy images with shape [N, H, W] or [N, H, W, C], "
                    f"got {self.frames.shape}"
                )
            self.timestamps = None
            self.poses = load_pose_file(poses_path, only_ok=only_ok_poses)
        elif image_source.is_file() and is_mha_path(image_source):
            self.frames, mha_poses, self.timestamps = load_mha_sequence(
                image_source,
                only_ok=only_ok_poses,
            )
            self.poses = load_pose_file(poses_path, only_ok=only_ok_poses) if poses_path else mha_poses
        else:
            self.image_paths = list_image_files(image_source)
            if not self.image_paths:
                raise ValueError(f"No images found in {image_dir}")
            self.timestamps = None
            self.poses = load_pose_file(poses_path, only_ok=only_ok_poses)

        sample_count = len(self.frames) if self.frames is not None else len(self.image_paths)
        if self.poses is not None and len(self.poses) != sample_count:
            raise ValueError(
                f"Pose count ({len(self.poses)}) does not match image count "
                f"({sample_count})"
            )

        self.calibration = calibration or UltrasoundCalibration()
        self.image_size = image_size
        self.grayscale = grayscale
        self.low_intensity_threshold = low_intensity_threshold

    def __len__(self):
        if self.frames is not None:
            return len(self.frames)
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.frames is not None:
            image = frame_to_tensor(
                self.frames[index],
                grayscale=self.grayscale,
                image_size=self.image_size,
                low_intensity_threshold=self.low_intensity_threshold,
            )
            path = f"{self.source_name}::frame_{index:06d}"
        else:
            image = load_image_tensor(
                self.image_paths[index],
                grayscale=self.grayscale,
                image_size=self.image_size,
                low_intensity_threshold=self.low_intensity_threshold,
            )
            path = str(self.image_paths[index])

        if self.poses is None:
            pose = torch.eye(4, dtype=torch.float32)
        else:
            pose = torch.from_numpy(self.poses[index]).float()

        return {
            "image": image,
            "pose": pose,
            "path": path,
        }


class MultiTrackedUltrasoundDataset(Dataset):
    def __init__(
        self,
        image_dirs,
        poses_paths=None,
        calibration=None,
        image_size=IMAGE_SIZE,
        grayscale=GRAYSCALE,
        only_ok_poses=ONLY_OK_POSES,
        low_intensity_threshold=LOW_INTENSITY_THRESHOLD,
    ):
        if isinstance(image_dirs, (str, Path)):
            image_dirs = [image_dirs]
        image_dirs = list(image_dirs)
        if not image_dirs:
            raise ValueError("At least one image directory or MHA path is required")

        if poses_paths is None:
            poses_paths = [None] * len(image_dirs)
        elif isinstance(poses_paths, (str, Path)):
            poses_paths = [poses_paths]
        else:
            poses_paths = list(poses_paths)

        if len(poses_paths) == 0:
            poses_paths = [None] * len(image_dirs)
        if len(poses_paths) != len(image_dirs):
            raise ValueError(
                f"Expected one poses path per image source, got "
                f"{len(image_dirs)} image sources and {len(poses_paths)} poses paths"
            )

        self.calibration = calibration or UltrasoundCalibration()
        self.datasets = [
            TrackedUltrasoundDataset(
                image_dir=image_dir,
                poses_path=poses_path,
                calibration=self.calibration,
                image_size=image_size,
                grayscale=grayscale,
                only_ok_poses=only_ok_poses,
                low_intensity_threshold=low_intensity_threshold,
            )
            for image_dir, poses_path in zip(image_dirs, poses_paths)
        ]
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = np.cumsum(self.lengths).tolist()
        self.source_name = ";".join(dataset.source_name for dataset in self.datasets)

    def __len__(self):
        return int(self.cumulative_lengths[-1])

    def __getitem__(self, index):
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)

        dataset_index = int(np.searchsorted(self.cumulative_lengths, index, side="right"))
        previous = 0 if dataset_index == 0 else self.cumulative_lengths[dataset_index - 1]
        sample = self.datasets[dataset_index][index - previous]
        sample["sequence_index"] = dataset_index
        return sample


def create_dataset():
    return TrackedUltrasoundDataset()


if __name__ == "__main__":
    dataset = create_dataset()
    sample = dataset[0]
    print(f"loaded {len(dataset)} images")
    print(f"first image: {sample['path']}")
    print(f"image shape: {tuple(sample['image'].shape)}")
    print(f"pose shape: {tuple(sample['pose'].shape)}")
    print(f"pixel_to_mm: {dataset.calibration.pixel_to_mm:.6f}")
