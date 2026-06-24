import json
import math
import random
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from load_data import MultiTrackedUltrasoundDataset, TrackedUltrasoundDataset
from ultrasound_losses import ultrasound_edge_loss
from ultragray_cuda_backend import load_ultragray_rasterizer


C0 = 0.28209479177387814


def _exact_ultragray_args(args):
    if not bool(args.native_ultragray_exact):
        return args

    values = vars(args).copy()
    values.update(
        {
            "init": "random",
            "num_gaussians": 500_000,
            "native_global_scale": 1.0,
            "native_init_extent": 0.8,
            "native_init_scale": 0.05,
            "initial_transmittance": 0.99,
            "batch_size": 8,
            "steps": 30_000,
            "means_lr": 1e-4,
            "scales_lr": 5e-3,
            "quats_lr": 5e-3,
            "transmittances_lr": 5e-4,
            "intensity_lr": 5e-3,
            "sh_rest_lr": 1e-5,
            "lr_final_factor": 0.1,
            "sh_degree": 1,
            "sh_degree_interval": 1000,
            "l1_weight": 0.5,
            "ssim_weight": 0.5,
            "ultrasound_edges_weight": 0.0,
            "scale_prior_weight": 0.01,
            "densify_grad_threshold": 5e-6,
            "native_grow_scale3d": 0.01,
            "native_prune_scale3d": 0.1,
            "native_prune_scale3d_min": 1e-5,
            "densify_start": 1000,
            "densify_stop": 20_000,
            "densify_every": 2500,
            "max_gaussians": 500_000,
            "pose_sideways_noise": 0.0,
            "pose_frontback_noise": 0.2,
            "pose_updown_noise": 0.0,
            "cuda_tile_size_x": 4,
            "cuda_tile_size_y": 128,
            "random_seed": 42,
        }
    )
    exact = Namespace(**values)
    print(
        "native UltraG-Ray exact mode: overriding core training, loss, "
        "optimizer, initialization, and density-control settings; preserving "
        "dataset-specific ultrasound geometry from the config"
    )
    return exact


def _load_native_components(repo_path):
    rasterizer = load_ultragray_rasterizer(repo_path)
    try:
        from fused_ssim import fused_ssim
    except ImportError as error:
        raise ImportError(
            "Native UltraG-Ray training requires the fused_ssim package used "
            "by UltraG-Ray trainer.py. Install it in the active environment."
        ) from error
    from gsplat.exporter import export_splats, splat2ply_bytes
    from gsplat.strategy import Ultrasound3DStrategy

    return (
        rasterizer,
        export_splats,
        splat2ply_bytes,
        Ultrasound3DStrategy,
        fused_ssim,
    )


def _build_dataset_from_paths(image_dirs, pose_paths, args):
    # The native CUDA path consumes the same image representation as
    # UltraG-Ray regardless of whether its training hyperparameters are exact.
    use_ultragray_loader = True
    requested_image_size = (
        None
        if args.native_ultragray_loader_exact
        else (args.height, args.width)
    )
    if len(image_dirs) == 1:
        dataset = TrackedUltrasoundDataset(
            image_dir=image_dirs[0],
            poses_path=pose_paths[0] if pose_paths else None,
            image_size=requested_image_size,
            grayscale=True,
            low_intensity_threshold=None,
            image_value_scale=args.image_value_scale,
            ultragray_exact=use_ultragray_loader,
        )
    else:
        dataset = MultiTrackedUltrasoundDataset(
            image_dirs=image_dirs,
            poses_paths=pose_paths,
            image_size=requested_image_size,
            grayscale=True,
            low_intensity_threshold=None,
            image_value_scale=args.image_value_scale,
            ultragray_exact=use_ultragray_loader,
        )
    for source in _source_datasets(dataset):
        expected_size = (
            tuple(source.original_image_size)
            if args.native_ultragray_loader_exact
            else (int(args.height), int(args.width))
        )
        sample_shape = tuple(source[0]["image"].shape)
        if sample_shape != (1, *expected_size):
            raise RuntimeError(
                f"Loader failed to resize {source.source_name}: expected "
                f"(1, {expected_size[0]}, {expected_size[1]}), got "
                f"{sample_shape}"
            )
    return dataset


def _stored_dataset_size(dataset):
    sizes = {
        tuple(source.original_image_size)
        for source in _source_datasets(dataset)
    }
    if len(sizes) != 1:
        raise ValueError(
            "UltraG-Ray requires every image source in a split to have the "
            f"same stored dimensions, got {sorted(sizes)}"
        )
    return next(iter(sizes))


def _build_dataset(args):
    print(
        "native UltraG-Ray loader: RGB frames use channel mean, then "
        "astype(float32) / 255.0 with no clipping, sanitization, or "
        "intensity threshold; "
        + (
            "runtime resizing is disabled"
            if args.native_ultragray_loader_exact
            else "runtime bilinear resizing is enabled"
        )
    )
    return _build_dataset_from_paths(args.image_dir, args.poses, args)


def _build_explicit_validation_dataset(args):
    if not args.image_dir_val:
        return None
    return _build_dataset_from_paths(
        args.image_dir_val,
        args.pose_val,
        args,
    )


def _source_datasets(dataset):
    return dataset.datasets if hasattr(dataset, "datasets") else [dataset]


def _print_dataset_diagnostics(dataset, args):
    print(
        f"input convention: {args.native_pose_convention}; "
        f"translations are multiplied by {args.native_pose_translation_scale}"
    )
    source_pose_centers = []
    source_pose_spans = []
    for source_index, source in enumerate(_source_datasets(dataset)):
        sample_indices = np.linspace(
            0,
            max(len(source) - 1, 0),
            min(len(source), 16),
            dtype=np.int64,
        )
        if source.frames is not None:
            raw = np.asarray(source.frames[sample_indices], dtype=np.float32)
            finite = raw[np.isfinite(raw)]
            raw_min = float(finite.min()) if finite.size else float("nan")
            raw_max = float(finite.max()) if finite.size else float("nan")
            raw_mean = float(finite.mean()) if finite.size else float("nan")
            raw_dtype = source.frames.dtype
            invalid_fraction = float(
                1.0 - np.isfinite(raw).mean()
            )
        else:
            raw_min = raw_max = raw_mean = float("nan")
            raw_dtype = "image-files"
            invalid_fraction = 0.0

        processed = torch.stack(
            [source[int(index)]["image"] for index in sample_indices],
            dim=0,
        )
        stored_size = tuple(source.original_image_size)
        training_size = tuple(processed.shape[-2:])
        print(
            f"input source {source_index}: {source.source_name}, frames={len(source)}, "
            f"stored_size={stored_size[0]}x{stored_size[1]}, "
            f"training_size={training_size[0]}x{training_size[1]}, "
            f"raw_dtype={raw_dtype}, raw[min/max/mean]="
            f"{raw_min:.6g}/{raw_max:.6g}/{raw_mean:.6g}, "
            f"invalid={100.0 * invalid_fraction:.3f}%, "
            f"value_scale={args.image_value_scale}, processed[min/max/mean]="
            f"{processed.min().item():.6g}/{processed.max().item():.6g}/"
            f"{processed.mean().item():.6g}"
        )
        if source.frames is not None and (
            invalid_fraction > 0.0 or raw_min < 0.0
        ):
            print(
                "WARNING: this NPY is not an UltraG-Ray-style grayscale array: "
                "it contains negative or invalid values. The current loader "
                "must clip them, which can erase weak echoes. Prefer the "
                "original numbered image directory when available."
            )

        if source.poses is None:
            print(f"pose source {source_index}: no poses; identity transforms are used")
            continue
        poses = np.asarray(source.poses, dtype=np.float64)
        rotations = poses[:, :3, :3]
        identity = np.eye(3, dtype=np.float64)
        orthogonality = np.linalg.norm(
            np.transpose(rotations, (0, 2, 1)) @ rotations - identity,
            axis=(1, 2),
        )
        determinants = np.linalg.det(rotations)
        bottom_error = np.abs(
            poses[:, 3, :] - np.asarray([0.0, 0.0, 0.0, 1.0])
        ).max()
        translations = (
            poses[:, :3, 3] * float(args.native_pose_translation_scale)
        )
        translation_min = translations.min(axis=0)
        translation_max = translations.max(axis=0)
        translation_span = translation_max - translation_min
        translation_center = 0.5 * (translation_min + translation_max)
        source_pose_centers.append(translation_center)
        source_pose_spans.append(translation_span)
        print(
            f"pose source {source_index}: det[min/max]="
            f"{determinants.min():.6g}/{determinants.max():.6g}, "
            f"max_orthogonality_error={orthogonality.max():.6g}, "
            f"bottom_row_error={bottom_error:.6g}, "
            f"translation_min_cm={translation_min.tolist()}, "
            f"translation_max_cm={translation_max.tolist()}, "
            f"span_cm={translation_span.tolist()}, "
            f"center_cm={translation_center.tolist()}"
        )
        reflected = determinants < 0.0
        if reflected.any():
            print(
                f"pose source {source_index}: WARNING: "
                f"{int(reflected.sum())}/{len(determinants)} transforms have "
                "determinant -1 (an axis reflection). UltraG-Ray accepts and "
                "uses these matrices directly."
            )
        if (
            orthogonality.max() > 1e-2
            or np.abs(np.abs(determinants) - 1.0).max() > 1e-2
            or bottom_error > 1e-3
        ):
            raise ValueError(
                f"Pose source {source.source_name} does not contain valid "
                "orthogonal homogeneous camera transforms"
            )

    if len(source_pose_centers) > 1:
        centers = np.stack(source_pose_centers)
        spans = np.stack(source_pose_spans)
        center_distances = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :],
            axis=-1,
        )
        max_center_distance = float(center_distances.max())
        median_source_span = float(
            np.median(np.linalg.norm(spans, axis=-1))
        )
        print(
            "multi-source pose report: "
            f"sources={len(centers)}, "
            f"max_center_distance_cm={max_center_distance:.6g}, "
            f"median_source_span_cm={median_source_span:.6g}"
        )
        if max_center_distance > max(2.0 * median_source_span, 5.0):
            print(
                "WARNING: source pose clouds are far apart. The loader only "
                "concatenates sweeps and does not register their coordinate "
                "systems. Train one source at a time or provide globally "
                "registered camera-to-world poses."
            )


def _print_ultragray_parity(dataset, args):
    differences = []
    if args.init != "random":
        differences.append(
            f"init={args.init!r}; UltraG-Ray uses random cube initialization"
        )
    if int(args.num_gaussians) != 500_000:
        differences.append(
            f"num_gaussians={args.num_gaussians}; UltraG-Ray default is 500000"
        )
    if int(args.max_gaussians) != 500_000:
        differences.append(
            f"max_gaussians={args.max_gaussians}; UltraG-Ray default is 500000"
        )
    if len(_source_datasets(dataset)) > 1:
        differences.append(
            f"{len(_source_datasets(dataset))} sweeps are concatenated; "
            "UltraG-Ray expects all poses to already share one world frame"
        )
    if int(args.steps) != 30_000:
        differences.append(
            f"steps={args.steps}; UltraG-Ray default is 30000"
        )
    if int(args.densify_start) != 1000:
        differences.append(
            f"densify_start={args.densify_start}; UltraG-Ray default is 1000"
        )
    if int(args.densify_every) != 2500:
        differences.append(
            f"densify_every={args.densify_every}; UltraG-Ray default is 2500"
        )
    if int(args.densify_stop) != 20_000:
        differences.append(
            f"densify_stop={args.densify_stop}; UltraG-Ray default is 20000"
        )
    expected_l1 = 1.0 - float(args.ssim_weight)
    if (
        float(args.ultrasound_edges_weight) != 0.0
        or abs(float(args.l1_weight) - expected_l1) > 1e-8
    ):
        differences.append(
            "loss weights differ; UltraG-Ray uses "
            "(1-ssim_lambda)*L1 + ssim_lambda*SSIM with no edge loss"
        )
    if differences:
        print("native UltraG-Ray parity differences:")
        for difference in differences:
            print(f"  - {difference}")
    else:
        print("native UltraG-Ray parity: core training defaults match")


def _split_dataset(dataset, args, explicit_validation_dataset=None):
    if explicit_validation_dataset is not None:
        print(
            f"using {len(dataset)} explicit training slices and "
            f"{len(explicit_validation_dataset)} explicit validation slices"
        )
        return dataset, explicit_validation_dataset

    from vanilla_gaussian_splatting import (
        choose_source_validation_indices,
        choose_validation_indices,
    )

    source_validation = choose_source_validation_indices(
        dataset,
        args.validation_sources,
    )
    if source_validation is None:
        validation_indices, training_indices = choose_validation_indices(
            len(dataset),
            args.validation_slices,
            args.validation_fraction,
            args.validation_seed,
        )
    else:
        validation_indices, training_indices, _ = source_validation
    train_dataset = Subset(dataset, training_indices) if validation_indices else dataset
    validation_dataset = (
        Subset(dataset, validation_indices)
        if validation_indices
        else None
    )
    return train_dataset, validation_dataset


def _pose_from_dataset(dataset, index):
    if isinstance(dataset, Subset):
        return _pose_from_dataset(dataset.dataset, dataset.indices[index])
    if hasattr(dataset, "datasets"):
        source_index = int(
            np.searchsorted(dataset.cumulative_lengths, index, side="right")
        )
        previous = (
            0
            if source_index == 0
            else dataset.cumulative_lengths[source_index - 1]
        )
        return _pose_from_dataset(
            dataset.datasets[source_index],
            index - previous,
        )
    if dataset.poses is None:
        return torch.eye(4, dtype=torch.float32)
    return torch.from_numpy(dataset.poses[index]).float()


def _camera_to_world_batch(
    poses,
    translation_scale,
    pose_to_camera,
    scene_center=None,
):
    cameras = poses.clone()
    cameras[:, :3, 3] *= float(translation_scale)
    cameras = cameras @ pose_to_camera
    if scene_center is not None:
        cameras[:, :3, 3] -= scene_center
    return cameras


def _native_pose_to_camera(dataset, args, opening_width, far_plane, device):
    dtype = torch.float32
    if args.native_pose_convention == "camera_to_world":
        return torch.eye(4, device=device, dtype=dtype)

    calibration = dataset.calibration
    image_t_probe = torch.as_tensor(
        calibration.image_t_probe,
        device=device,
        dtype=dtype,
    ).clone()
    image_t_probe[:3, 3] *= float(args.native_pose_translation_scale)
    probe_t_image = torch.linalg.inv(image_t_probe)

    source_sizes = {
        tuple(source.original_image_size)
        for source in _source_datasets(dataset)
    }
    if len(source_sizes) != 1:
        raise ValueError(
            "All sources must have the same original image size for shared "
            "probe-to-image calibration"
        )
    original_height, original_width = next(iter(source_sizes))
    resize_x = float(args.width) / float(original_width)
    resize_y = float(args.height) / float(original_height)
    origin = torch.as_tensor(
        calibration.image_plane_origin_px,
        device=device,
        dtype=dtype,
    )
    resized_origin_x = (
        float(origin[0]) * resize_x
        if args.image_origin_x is None
        else float(args.image_origin_x)
    )
    resized_origin_y = (
        float(origin[1]) * resize_y
        if args.image_origin_y is None
        else float(args.image_origin_y)
    )
    spacing_x = float(opening_width) / float(args.width)
    spacing_y = float(far_plane) / float(args.height)

    image_t_camera = torch.eye(4, device=device, dtype=dtype)
    # Native CUDA camera axes: X=lateral, Y=elevational, Z=axial depth.
    image_t_camera[:3, :3] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    image_t_camera[0, 3] = (
        float(args.width) * 0.5 - resized_origin_x
    ) * spacing_x
    image_t_camera[1, 3] = -resized_origin_y * spacing_y
    pose_to_camera = probe_t_image @ image_t_camera
    print(
        "native probe calibration: "
        f"original_size={original_height}x{original_width}, "
        f"resized_origin=({resized_origin_x:.6g}, {resized_origin_y:.6g}), "
        f"spacing_cm=({spacing_x:.6g}, {spacing_y:.6g}), "
        f"probe_to_camera_translation_cm="
        f"{pose_to_camera[:3, 3].detach().cpu().tolist()}"
    )
    return pose_to_camera


def _collect_camera_to_worlds(
    dataset,
    translation_scale,
    pose_to_camera,
    device,
):
    cameras = []
    for index in range(len(dataset)):
        pose = _pose_from_dataset(dataset, index).to(device)
        pose = pose.clone()
        pose[:3, 3] *= float(translation_scale)
        cameras.append(pose @ pose_to_camera)
    return torch.stack(cameras, dim=0)


def _center_cameras(camera_to_worlds, opening_width, far_plane):
    device = camera_to_worlds.device
    dtype = camera_to_worlds.dtype
    local_corners = torch.tensor(
        [
            [-0.5 * opening_width, 0.0, 0.0],
            [0.5 * opening_width, 0.0, 0.0],
            [0.5 * opening_width, 0.0, far_plane],
            [-0.5 * opening_width, 0.0, far_plane],
        ],
        device=device,
        dtype=dtype,
    )
    rotations = camera_to_worlds[:, :3, :3]
    translations = camera_to_worlds[:, :3, 3]
    world_corners = (
        torch.einsum("nij,kj->nki", rotations, local_corners)
        + translations[:, None, :]
    )
    scene_center = world_corners.reshape(-1, 3).mean(dim=0)
    centered = camera_to_worlds.clone()
    centered[:, :3, 3] -= scene_center
    scene_scale = torch.linalg.norm(
        (world_corners - scene_center).reshape(-1, 3),
        dim=-1,
    ).max()
    return centered, scene_center, float(scene_scale.item())


def _sample_cosine_offset(magnitude, device, dtype):
    magnitude = float(magnitude)
    if magnitude <= 0.0:
        return torch.zeros((), device=device, dtype=dtype)
    for _ in range(100):
        value = torch.rand((), device=device, dtype=dtype) - 0.5
        if (
            torch.rand((), device=device, dtype=dtype)
            <= torch.cos(math.pi * value).square()
        ):
            return value * magnitude
    return torch.zeros((), device=device, dtype=dtype)


def _augment_camera_poses(camera_to_worlds, args):
    cameras = camera_to_worlds.clone()
    device = cameras.device
    dtype = cameras.dtype
    offsets = (
        (torch.rand((), device=device, dtype=dtype) - 0.5)
        * float(args.pose_sideways_noise),
        _sample_cosine_offset(args.pose_frontback_noise, device, dtype),
        (torch.rand((), device=device, dtype=dtype) - 0.5)
        * float(args.pose_updown_noise),
    )
    for axis, offset in enumerate(offsets):
        cameras[:, :3, 3] += offset * cameras[:, :3, axis]
    return cameras


def _sample_means_from_bright_pixels(
    dataset,
    camera_to_worlds,
    count,
    opening_width,
    far_plane,
    elevational_jitter,
    intensity_threshold,
    intensity_power,
    pixel_stride,
):
    device = camera_to_worlds.device
    means = torch.empty((count, 3), device=device)
    initial_echo = torch.empty((count,), device=device)
    frame_ids = torch.randint(0, len(dataset), (count,), device="cpu")
    frame_counts = torch.bincount(frame_ids, minlength=len(dataset))
    stride = max(int(pixel_stride), 1)
    threshold = float(intensity_threshold)
    power = float(intensity_power)
    write_offset = 0
    fallback_frames = 0

    for frame_index, frame_count in enumerate(frame_counts.tolist()):
        if frame_count == 0:
            continue
        image = dataset[frame_index]["image"].detach().float().cpu()
        if image.ndim == 3:
            image = image.mean(dim=0)
        sampled = image[::stride, ::stride]
        sampled_height, sampled_width = sampled.shape
        weights = (sampled - threshold).clamp_min(0.0).pow(power).reshape(-1)
        if float(weights.sum()) <= 0.0:
            weights = sampled.clamp_min(0.0).pow(power).reshape(-1)
            fallback_frames += 1
        if float(weights.sum()) <= 0.0:
            weights = torch.ones_like(weights)
            fallback_frames += 1

        selected = torch.multinomial(
            weights,
            frame_count,
            replacement=True,
        )
        rows = torch.div(selected, sampled_width, rounding_mode="floor")
        columns = selected % sampled_width
        source_rows = rows.float() * stride
        source_columns = columns.float() * stride
        height, width = image.shape

        lateral = (
            (source_columns + 0.5 + torch.rand(frame_count) - 0.5)
            / float(width)
            - 0.5
        ) * float(opening_width)
        axial = (
            (source_rows + torch.rand(frame_count) * stride)
            / max(float(height - 1), 1.0)
        ).clamp(0.0, 1.0) * float(far_plane)
        elevational = (
            torch.rand(frame_count) * 2.0 - 1.0
        ) * float(elevational_jitter)
        local = torch.stack(
            [lateral, elevational, axial],
            dim=1,
        ).to(device)
        camera = camera_to_worlds[frame_index]
        end = write_offset + frame_count
        means[write_offset:end] = (
            local @ camera[:3, :3].T + camera[:3, 3]
        )
        initial_echo[write_offset:end] = sampled.reshape(-1)[selected].to(device)
        write_offset = end

    print(
        "native SVRTK intensity sampling: "
        f"threshold={threshold}, power={power}, stride={stride}, "
        f"fallback_frames={fallback_frames}"
    )
    return means, initial_echo.clamp(0.0, 1.0)


def _create_splats(
    args,
    dataset,
    camera_to_worlds,
    opening_width,
    far_plane,
    scene_scale,
    device,
):
    count = int(args.num_gaussians)
    initial_echo = None
    if args.init == "random":
        init_device = (
            torch.device("cpu")
            if args.native_ultragray_exact
            else device
        )
        means = (
            float(args.native_init_extent)
            * scene_scale
            * (torch.rand((count, 3), device=init_device) * 2.0 - 1.0)
        ).to(device)
        camera_scale = torch.full(
            (3,),
            float(args.native_init_scale),
            device=init_device,
            dtype=torch.float32,
        ).to(device)
    else:
        means, initial_echo = _sample_means_from_bright_pixels(
            dataset,
            camera_to_worlds,
            count,
            opening_width,
            far_plane,
            max(0.1 * float(args.initial_scale_z), 1e-4),
            args.intensity_threshold,
            args.svrtk_intensity_power,
            args.pixel_stride,
        )
        # Existing config scale values are millimetres. Native renderer uses cm.
        camera_scale = torch.tensor(
            [
                args.initial_scale_x,
                args.initial_scale_z,
                args.initial_scale_y,
            ],
            device=device,
            dtype=torch.float32,
        ) * 0.1
    scales = camera_scale.clamp_min(1e-6).log().repeat(count, 1)
    quat_device = (
        torch.device("cpu")
        if args.native_ultragray_exact
        else device
    )
    quats = torch.rand((count, 4), device=quat_device).to(device)
    transmittances = torch.full(
        (count,),
        float(args.initial_transmittance),
        device=quat_device,
    ).logit(eps=1e-10).to(device)

    sh_degree = int(args.sh_degree)
    coefficients = (sh_degree + 1) ** 2
    colors = torch.zeros(
        (count, coefficients, 1),
        device=quat_device,
    )
    if initial_echo is None:
        colors[:, 0, :] = (0.1 - 0.5) / C0
    else:
        colors[:, 0, 0] = (initial_echo.to(quat_device) - 0.5) / C0
    colors = colors.to(device)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "transmittances": torch.nn.Parameter(transmittances),
            "sh0": torch.nn.Parameter(colors[:, :1, :]),
            "shN": torch.nn.Parameter(colors[:, 1:, :]),
        }
    )


def _create_optimizers(splats, args, scene_scale):
    batch_scale = math.sqrt(max(int(args.batch_size), 1))
    definitions = {
        "means": float(args.means_lr) * float(scene_scale),
        "scales": float(args.scales_lr),
        "quats": float(args.quats_lr),
        "transmittances": float(args.transmittances_lr),
        "sh0": float(args.intensity_lr),
        "shN": float(args.sh_rest_lr),
    }
    return {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * batch_scale, "name": name}],
            eps=1e-15 / batch_scale,
            betas=(
                1 - int(args.batch_size) * (1 - 0.9),
                1 - int(args.batch_size) * (1 - 0.999),
            ),
        )
        for name, lr in definitions.items()
    }


def _native_loss(render, target, splats, args, fused_ssim):
    render_bchw = render.permute(0, 3, 1, 2)
    l1 = F.l1_loss(render_bchw, target)
    ssim = 1.0 - fused_ssim(
        render_bchw,
        target,
        padding="valid",
    )
    edge = render_bchw.sum() * 0.0
    if float(args.ultrasound_edges_weight) > 0.0:
        edge = ultrasound_edge_loss(
            render_bchw,
            target,
            laplacian_weight=args.laplacian_loss_weight,
            edge_weight=args.edge_loss_weight,
            intensity_weight=args.intensity_loss_weight,
            sobel_weight=args.sobel_loss_weight,
            sigmas=(args.filter_sigma,),
            kernel_size=args.filter_kernel_size,
            sobel_blur_kernel_size=args.filter_kernel_size,
            sobel_blur_sigma=args.filter_sigma,
            use_confidence=args.use_confidence,
            background_threshold=args.confidence_background_threshold,
            background_weight=args.confidence_background_weight,
            dark_threshold=args.confidence_dark_threshold,
            shadow_weight=args.confidence_shadow_weight,
            bright_threshold=args.confidence_bright_threshold,
            shadow_start_offset=args.confidence_shadow_start_offset,
            enable_shadow_confidence=args.shadow_confidence,
            content_normalize=args.content_normalize,
            content_intensity_threshold=args.content_intensity_threshold,
            content_feature_threshold=args.content_feature_threshold,
            content_background_weight=args.content_background_weight,
        )
    scale = torch.exp(splats["scales"]).mean()
    total = (
        float(args.l1_weight) * l1
        + float(args.ssim_weight) * ssim
        + float(args.ultrasound_edges_weight) * edge
        + float(args.scale_prior_weight) * scale
    )
    return total, l1, ssim, edge, scale


@torch.no_grad()
def _evaluate_validation(
    validation_dataset,
    splats,
    args,
    device,
    pose_to_camera,
    scene_center,
    opening_width,
    far_plane,
    rasterizer,
    fused_ssim,
):
    if validation_dataset is None:
        return None
    loader = DataLoader(
        validation_dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_samples = 0
    sh_degree = int(args.sh_degree)
    intensities = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    for batch in loader:
        target = batch["image"].to(device, non_blocking=True)
        poses = batch["pose"].to(device, non_blocking=True)
        camera_to_worlds = _camera_to_world_batch(
            poses,
            args.native_pose_translation_scale,
            pose_to_camera,
            scene_center,
        )
        render, _, _, _, _ = rasterizer(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            transmittances=torch.sigmoid(splats["transmittances"]),
            intensities=intensities,
            viewmats=torch.linalg.inv(camera_to_worlds),
            width=int(args.width),
            height=int(args.height),
            near_plane=0.0,
            far_plane=far_plane,
            opening_angle=None,
            opening_width=opening_width,
            tile_size_x=int(args.cuda_tile_size_x),
            tile_size_y=int(args.cuda_tile_size_y),
            sh_degree=sh_degree,
        )
        loss, l1, ssim, _, _ = _native_loss(
            render,
            target,
            splats,
            args,
            fused_ssim,
        )
        batch_size = int(target.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_l1 += float(l1.item()) * batch_size
        total_ssim += float(ssim.item()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        return None
    return {
        "loss": total_loss / total_samples,
        "l1": total_l1 / total_samples,
        "ssim": total_ssim / total_samples,
        "samples": total_samples,
    }


def _raw_all_ply_bytes(splats, splat2ply_bytes):
    count = len(splats["means"])
    sh0 = splats["sh0"]
    shN = splats["shN"]
    if sh0.shape[-1] == 1:
        sh0 = sh0.expand(-1, -1, 3)
    if shN.shape[-1] == 1:
        shN = shN.expand(-1, -1, 3)
    sh0 = sh0.squeeze(1)
    shN = shN.permute(0, 2, 1).reshape(count, -1)
    return splat2ply_bytes(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=torch.ones(
            count,
            device=splats["means"].device,
        ),
        sh0=sh0,
        shN=shN,
    )


def _invalid_gaussian_count(splats):
    invalid = torch.zeros(
        len(splats["means"]),
        dtype=torch.bool,
        device=splats["means"].device,
    )
    for values in splats.values():
        invalid |= ~torch.isfinite(values).reshape(len(values), -1).all(dim=1)
    return int(invalid.sum().item())


def _export(output, splats, export_splats, splat2ply_bytes):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "splats": splats.state_dict(),
    }
    torch.save(checkpoint, output)
    all_checkpoint_path = output.with_name(
        f"{output.stem}_all{output.suffix}"
    )
    torch.save(checkpoint, all_checkpoint_path)
    ply_path = output.with_suffix(".ply")
    # Match UltraG-Ray trainer.py exactly: export every learned Gaussian and
    # use a constant opacity because ultrasound visibility is represented by
    # echo SH coefficients and acoustic transmittance, not 3DGS opacity.
    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=torch.ones(
            len(splats["means"]),
            device=splats["means"].device,
        ),
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=str(ply_path),
    )
    all_ply_path = output.with_name(f"{output.stem}_all.ply")
    all_ply_path.write_bytes(_raw_all_ply_bytes(splats, splat2ply_bytes))
    return (
        output,
        ply_path,
        all_checkpoint_path,
        all_ply_path,
        _invalid_gaussian_count(splats),
    )


def _export_initial_ply(output, splats, export_splats):
    initial_ply_path = Path(output).with_name(
        f"{Path(output).stem}_initial.ply"
    )
    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=torch.ones(
            len(splats["means"]),
            device=splats["means"].device,
        ),
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=str(initial_ply_path),
    )
    return initial_ply_path


def _export_sog(ply_path, args):
    if not bool(args.export_sog):
        return None

    executable = shutil.which(str(args.splat_transform))
    if executable is None:
        print(
            "WARNING: SOG export skipped because "
            f"{args.splat_transform!r} was not found. Install it with "
            "'npm install -g @playcanvas/splat-transform'."
        )
        return None

    sog_path = Path(ply_path).with_suffix(".sog")
    command = [executable]
    if args.sog_gpu is not None:
        command.extend(["-g", str(args.sog_gpu)])
    command.extend(["-w", str(ply_path), str(sog_path)])
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print(
            "WARNING: SOG export failed after training with exit code "
            f"{error.returncode}. Check the splat-transform output above."
        )
        return None
    return sog_path


def train_native_ultragray(args):
    from vanilla_gaussian_splatting import save_debug_visuals

    args = _exact_ultragray_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Native UltraG-Ray training requires CUDA")
    if not args.output:
        raise ValueError("Native UltraG-Ray training requires an output path")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed = int(args.random_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    (
        rasterizer,
        export_splats,
        splat2ply_bytes,
        strategy_type,
        fused_ssim,
    ) = _load_native_components(
        args.ultragray_repo_path
    )

    dataset = _build_dataset(args)
    explicit_validation_dataset = _build_explicit_validation_dataset(args)
    if args.native_ultragray_loader_exact:
        stored_height, stored_width = _stored_dataset_size(dataset)
        configured_size = (int(args.height), int(args.width))
        args.height = int(stored_height)
        args.width = int(stored_width)
        print(
            "native UltraG-Ray dimensions: "
            f"using stored NPY size {args.height}x{args.width}; "
            f"configured size {configured_size[0]}x{configured_size[1]} "
            "is ignored"
        )
        if explicit_validation_dataset is not None:
            validation_size = _stored_dataset_size(
                explicit_validation_dataset
            )
            if validation_size != (args.height, args.width):
                raise ValueError(
                    "UltraG-Ray requires train and validation images to have "
                    "the same stored dimensions; "
                    f"train={args.height}x{args.width}, "
                    f"validation={validation_size[0]}x{validation_size[1]}"
                )
    _print_dataset_diagnostics(dataset, args)
    _print_ultragray_parity(dataset, args)
    if explicit_validation_dataset is not None:
        print("explicit validation dataset diagnostics:")
        _print_dataset_diagnostics(explicit_validation_dataset, args)
    train_dataset, validation_dataset = _split_dataset(
        dataset,
        args,
        explicit_validation_dataset,
    )
    far_plane = float(args.ultrasound_far_plane)
    opening_width = (
        far_plane * float(args.width) / float(args.height)
        if args.ultrasound_opening_width is None
        else float(args.ultrasound_opening_width)
    )
    spacing_x = opening_width / float(args.width)
    spacing_y = far_plane / float(args.height)
    pose_to_camera = _native_pose_to_camera(
        dataset,
        args,
        opening_width,
        far_plane,
        device,
    )
    print(
        "native UltraG-Ray geometry: "
        f"batch={args.batch_size}, opening_width={opening_width:.6f}, "
        f"far_plane={far_plane:.6f} cm, "
        f"target_ratio={args.height}:{args.width}, "
        f"pixel_aspect={float(args.width) / float(args.height):.6f}, "
        f"physical_aspect={opening_width / far_plane:.6f}, "
        f"spacing=({spacing_x:.6f}, {spacing_y:.6f}) cm/px"
    )
    print(
        "native effective dimensions: "
        f"image={int(args.height)}x{int(args.width)} pixels, "
        f"physical={far_plane:.6f}x{opening_width:.6f} cm "
        "(depth x lateral)"
    )
    print(
        "native UltraG-Ray representation: "
        "world means + learned scales/quaternions + SH intensity + transmittance"
    )
    if int(args.accumulation_steps) != 1:
        print(
            "native UltraG-Ray uses true batch_size; "
            "accumulation_steps is ignored"
        )

    all_cameras = _collect_camera_to_worlds(
        train_dataset,
        args.native_pose_translation_scale,
        pose_to_camera,
        device,
    )
    all_cameras, scene_center, raw_scene_scale = _center_cameras(
        all_cameras,
        opening_width,
        far_plane,
    )
    scene_scale = (
        raw_scene_scale
        * 1.1
        * float(args.native_global_scale)
    )
    print(
        "native scene centering: "
        f"center={scene_center.detach().cpu().tolist()}, "
        f"raw_scene_scale={raw_scene_scale:.6f} cm, "
        f"training_scene_scale={scene_scale:.6f} cm"
    )
    print(
        "native pose augmentation: "
        f"sideways={args.pose_sideways_noise} cm, "
        f"frontback={args.pose_frontback_noise} cm, "
        f"updown={args.pose_updown_noise} cm"
    )
    splats = _create_splats(
        args,
        train_dataset,
        all_cameras,
        opening_width,
        far_plane,
        scene_scale,
        device,
    )
    print(
        "native initialization: "
        f"mode={args.init}, gaussians={len(splats['means'])}, "
        f"scale_cm[min/max]="
        f"{torch.exp(splats['scales']).min().item():.6g}/"
        f"{torch.exp(splats['scales']).max().item():.6g}, "
        f"means_lr={float(args.means_lr) * scene_scale:.6g}"
    )
    initial_ply_path = _export_initial_ply(
        args.output,
        splats,
        export_splats,
    )
    print(f"saved initial native UltraG-Ray PLY {initial_ply_path}")
    optimizers = _create_optimizers(splats, args, scene_scale)
    final_factor = float(args.lr_final_factor)
    gamma = final_factor ** (1.0 / max(int(args.steps), 1))
    schedulers = {
        name: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        for name, optimizer in optimizers.items()
    }

    strategy = strategy_type(
        grow_grad3d=float(args.densify_grad_threshold),
        grow_scale3d=float(args.native_grow_scale3d),
        prune_scale3d=float(args.native_prune_scale3d),
        prune_scale3d_min=float(args.native_prune_scale3d_min),
        refine_start_iter=int(args.densify_start),
        refine_stop_iter=min(int(args.densify_stop), int(args.steps)),
        refine_every=max(int(args.densify_every), 1),
        max_gaussians=int(args.max_gaussians),
        verbose=bool(args.log_densify),
    )
    print(
        f"native density control: start={args.densify_start}, "
        f"stop={min(int(args.densify_stop), int(args.steps))}, "
        f"every={args.densify_every}, max={args.max_gaussians}, "
        f"grow_scale3d={args.native_grow_scale3d}, "
        f"prune_scale3d={args.native_prune_scale3d}, "
        f"prune_scale3d_min={args.native_prune_scale3d_min}"
    )
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    loader = DataLoader(
        train_dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=True,
        num_workers=8 if args.native_ultragray_exact else 0,
        persistent_workers=bool(args.native_ultragray_exact),
        pin_memory=True,
        drop_last=False,
    )
    loader_iterator = iter(loader)
    final_export = None
    validation_history = []
    for step in range(int(args.steps)):
        try:
            batch = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            batch = next(loader_iterator)

        target = batch["image"].to(device, non_blocking=True)
        expected_target_shape = (
            int(target.shape[0]),
            1,
            int(args.height),
            int(args.width),
        )
        if tuple(target.shape) != expected_target_shape:
            raise RuntimeError(
                f"Training batch has wrong image shape: expected "
                f"{expected_target_shape}, got {tuple(target.shape)}"
            )
        poses = batch["pose"].to(device, non_blocking=True)
        camera_to_worlds = _camera_to_world_batch(
            poses,
            args.native_pose_translation_scale,
            pose_to_camera,
            scene_center,
        )
        camera_to_worlds = _augment_camera_poses(camera_to_worlds, args)
        sh_degree = min(
            step // max(int(args.sh_degree_interval), 1),
            int(args.sh_degree),
        )
        intensities = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        render, _, _, _, info = rasterizer(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            transmittances=torch.sigmoid(splats["transmittances"]),
            intensities=intensities,
            viewmats=torch.linalg.inv(camera_to_worlds),
            width=int(args.width),
            height=int(args.height),
            near_plane=0.0,
            far_plane=far_plane,
            opening_angle=None,
            opening_width=opening_width,
            tile_size_x=int(args.cuda_tile_size_x),
            tile_size_y=int(args.cuda_tile_size_y),
            sh_degree=sh_degree,
        )
        expected_render_shape = (
            int(target.shape[0]),
            int(args.height),
            int(args.width),
            1,
        )
        if tuple(render.shape) != expected_render_shape:
            raise RuntimeError(
                f"CUDA renderer returned wrong image shape: expected "
                f"{expected_render_shape}, got {tuple(render.shape)}"
            )
        loss, l1, ssim, edge, scale = _native_loss(
            render,
            target,
            splats,
            args,
            fused_ssim,
        )
        strategy.step_pre_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )
        loss.backward()

        strategy.step_post_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )
        update_sample_indices = torch.linspace(
            0,
            len(splats["means"]) - 1,
            min(len(splats["means"]), 4096),
            device=device,
        ).long()
        means_before_update = splats["means"].detach()[
            update_sample_indices
        ].clone()
        if bool(args.native_ultragray_exact) and step == int(args.steps) - 1:
            final_export = _export(
                args.output,
                splats,
                export_splats,
                splat2ply_bytes,
            )

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers.values():
            scheduler.step()

        validation_metrics = None
        exact_validation_steps = {
            999,
            9_999,
            19_999,
            24_999,
            29_999,
        }
        if (
            validation_dataset is not None
            and (
                (
                    args.native_ultragray_exact
                    and step in exact_validation_steps
                )
                or (
                    not args.native_ultragray_exact
                    and int(args.validation_every) > 0
                    and (
                        step == 0
                        or (step + 1) % int(args.validation_every) == 0
                        or step == int(args.steps) - 1
                    )
                )
            )
        ):
            validation_metrics = _evaluate_validation(
                validation_dataset,
                splats,
                args,
                device,
                pose_to_camera,
                scene_center,
                opening_width,
                far_plane,
                rasterizer,
                fused_ssim,
            )
            if validation_metrics is not None:
                validation_history.append(
                    {
                        "step": int(step),
                        **validation_metrics,
                    }
                )
                print(
                    f"validation step={step:05d} "
                    f"loss={validation_metrics['loss']:.6f} "
                    f"l1={validation_metrics['l1']:.6f} "
                    f"ssim={validation_metrics['ssim']:.6f} "
                    f"slices={validation_metrics['samples']}"
                )

        should_log = step == 0 or step % int(args.log_every) == 0
        if should_log:
            step_movement = torch.linalg.norm(
                splats["means"].detach()[update_sample_indices]
                - means_before_update,
                dim=-1,
            )
            visible = (info["radii"] > 0).all(dim=-1).any(dim=0).float().mean()
            print(
                f"step={step:05d} loss={loss.item():.6f} "
                f"l1={l1.item():.6f} ssim={ssim.item():.6f} "
                f"edge={edge.item():.6f} scale={scale.item():.6f} "
                f"visible={100.0 * visible.item():.2f}% "
                f"step_move={step_movement.mean().item():.6g} "
                f"gaussians={len(splats['means'])}"
            )

        if int(args.debug_every) > 0 and (
            step == 0 or step % int(args.debug_every) == 0
        ):
            save_debug_visuals(
                render[0].permute(2, 0, 1),
                target[0],
                step,
                args.debug_dir,
                sobel_weight=args.sobel_loss_weight,
                filter_kernel_size=args.filter_kernel_size,
                filter_sigma=args.filter_sigma,
            )

        if (
            args.output
            and int(args.checkpoint_every) > 0
            and step % int(args.checkpoint_every) == 0
        ):
            step_path = Path(args.output).with_name(
                f"{Path(args.output).stem}_{step:06d}{Path(args.output).suffix}"
            )
            torch.save(
                {"step": step, "splats": splats.state_dict()},
                step_path,
            )

    if final_export is None:
        final_export = _export(
            args.output,
            splats,
            export_splats,
            splat2ply_bytes,
        )
    (
        output_path,
        ply_path,
        all_checkpoint_path,
        all_ply_path,
        invalid_gaussians,
    ) = final_export
    sog_path = _export_sog(ply_path, args)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "trainer": "native_ultragray",
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "gaussians": len(splats["means"]),
                "opening_width": opening_width,
                "far_plane": far_plane,
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "pose_translation_scale": float(
                    args.native_pose_translation_scale
                ),
                "pose_convention": args.native_pose_convention,
                "pose_to_camera": [
                    [float(value) for value in row]
                    for row in pose_to_camera.detach().cpu()
                ],
                "scene_center": [
                    float(value)
                    for value in scene_center.detach().cpu()
                ],
                "scene_scale": scene_scale,
                "raw_scene_scale": raw_scene_scale,
                "init": args.init,
                "native_init_extent": float(args.native_init_extent),
                "native_init_scale": float(args.native_init_scale),
                "pose_sideways_noise": float(args.pose_sideways_noise),
                "pose_frontback_noise": float(args.pose_frontback_noise),
                "pose_updown_noise": float(args.pose_updown_noise),
                "ply_export": "ultragray_native",
                "initial_gaussians_ply": str(initial_ply_path),
                "sog_export": None if sog_path is None else str(sog_path),
                "all_gaussians_checkpoint": str(all_checkpoint_path),
                "all_gaussians_ply": str(all_ply_path),
                "all_gaussians_count": len(splats["means"]),
                "invalid_gaussians_in_all_export": invalid_gaussians,
                "validation_slices": (
                    0 if validation_dataset is None else len(validation_dataset)
                ),
                "validation_history": validation_history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved native UltraG-Ray checkpoint {output_path}")
    print(f"saved native UltraG-Ray PLY {ply_path}")
    if sog_path is not None:
        print(f"saved native UltraG-Ray SOG {sog_path}")
    print(
        f"saved all {len(splats['means'])} remaining Gaussians "
        f"without export filtering to {all_checkpoint_path}"
    )
    print(
        f"saved all {len(splats['means'])} remaining Gaussian vertices "
        f"without export filtering to {all_ply_path} "
        f"(invalid={invalid_gaussians})"
    )
    print(f"saved metadata {metadata_path}")
