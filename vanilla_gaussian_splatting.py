import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from load_data import MultiTrackedUltrasoundDataset, TrackedUltrasoundDataset
from ultrasound_losses import content_weight_map, ultrasound_confidence_map, ultrasound_edge_loss, ultrasound_edge_map
from ultrasound_projection import (
    DEFAULT_IMAGE_PLANE_ORIGIN_PX,
    DEFAULT_IMAGE_T_PROBE,
    DEFAULT_PIXEL_TO_MM,
    render_ultrasound_gaussians,
)


class VanillaGaussianSplatting(nn.Module):
    """
    Minimal differentiable ultrasound Gaussian splatting in plain PyTorch.

    This renders a tracked ultrasound slice by intersecting the 3D Gaussian
    field with the physical slice plane, not by using a pinhole camera.
    """

    def __init__(
        self,
        num_gaussians,
        channels=3,
        scene_scale=1.0,
        initial_scale=1.0,
        initial_opacity=0.5,
        initial_means=None,
        initial_colors=None,
    ):
        super().__init__()
        if initial_means is None:
            means = torch.empty(num_gaussians, 3)
            means[:, :2].uniform_(-scene_scale, scene_scale)
            means[:, 2].uniform_(scene_scale, 3.0 * scene_scale)
        else:
            means = initial_means.float()
            num_gaussians = len(means)

        self.means = nn.Parameter(means)
        initial_scale = torch.as_tensor(initial_scale, dtype=torch.float32)
        if initial_scale.ndim == 0:
            initial_scale = initial_scale.repeat(3)
        if initial_scale.shape != (3,):
            raise ValueError("initial_scale must be a scalar or a 3-value sequence")
        self.log_scales = nn.Parameter(initial_scale.clamp_min(1e-6).log().repeat(num_gaussians, 1))
        initial_opacity = float(initial_opacity)
        if initial_opacity <= 0.0 or initial_opacity >= 1.0:
            raise ValueError("initial_opacity must be between 0 and 1")
        opacity_logit = torch.logit(torch.as_tensor(initial_opacity, dtype=torch.float32))
        self.logit_opacities = nn.Parameter(opacity_logit.repeat(num_gaussians, 1))

        if initial_colors is None:
            colors = torch.rand(num_gaussians, channels)
        else:
            colors = initial_colors.float()
            if colors.shape[-1] != channels:
                if channels == 1:
                    colors = colors.mean(dim=-1, keepdim=True)
                elif colors.shape[-1] == 1:
                    colors = colors.repeat(1, channels)
                else:
                    raise ValueError("initial_colors must match the requested channel count")

        self.colors = nn.Parameter(colors.clamp(0.0, 1.0))
        normals = torch.randn(num_gaussians, 3) * 0.01
        normals[:, 2] = 1.0
        self.disk_normals = nn.Parameter(normals)
        self.acoustic_attenuation = nn.Parameter(torch.tensor(-4.0))
        self.acoustic_reflection = nn.Parameter(torch.tensor(-4.0))
        self.acoustic_scattering = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        slice_to_world,
        height,
        width,
        pixel_spacing=(1.0, 1.0),
        slice_thickness=1.0,
        shadowing=True,
        shadow_strength=1.0,
        image_t_probe=None,
        image_plane_origin_px=DEFAULT_IMAGE_PLANE_ORIGIN_PX,
        pixel_to_mm=DEFAULT_PIXEL_TO_MM,
        image_scale=1.0,
        covariance_mode="ultrasound_psf",
        min_scale_mm=0.05,
        max_scale_mm=10.0,
        lateral_depth_slope=0.0,
        elevational_depth_slope=0.0,
        render_chunk_size=256,
        primitive_mode="volume",
        acoustic_rendering=False,
    ):
        disk_normals = None
        if primitive_mode == "disk":
            disk_normals = F.normalize(self.disk_normals, dim=-1, eps=1e-8)

        image, _ = render_ultrasound_gaussians(
            self.means,
            torch.exp(self.log_scales),
            self.colors,
            self.logit_opacities,
            slice_to_world,
            height,
            width,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            shadowing=shadowing,
            shadow_strength=shadow_strength,
            image_t_probe=image_t_probe,
            image_plane_origin_px=image_plane_origin_px,
            pixel_to_mm=pixel_to_mm,
            image_scale=image_scale,
            covariance_mode=covariance_mode,
            min_scale_mm=min_scale_mm,
            max_scale_mm=max_scale_mm,
            lateral_depth_slope=lateral_depth_slope,
            elevational_depth_slope=elevational_depth_slope,
            render_chunk_size=render_chunk_size,
            primitive_mode=primitive_mode,
            disk_normals=disk_normals,
            acoustic_rendering=acoustic_rendering,
            attenuation_weight=F.softplus(self.acoustic_attenuation),
            reflection_weight=F.softplus(self.acoustic_reflection),
            scattering_weight=F.softplus(self.acoustic_scattering),
        )
        return image

    def scale_prior_loss(self, prior_scales, min_scale_mm=0.05, max_scale_mm=10.0):
        prior = torch.as_tensor(prior_scales, device=self.log_scales.device, dtype=self.log_scales.dtype)
        if prior.shape != (3,):
            raise ValueError("prior_scales must contain [lateral, axial, elevational] values")
        scales = torch.exp(self.log_scales).clamp(min_scale_mm, max_scale_mm)
        log_prior = prior.clamp_min(min_scale_mm).log()
        return F.mse_loss(scales.log(), log_prior.expand_as(scales.log()))

    @torch.no_grad()
    def stabilize_parameters(self, min_scale_mm=0.05, max_scale_mm=10.0):
        self.means.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
        self.means.clamp_(-1e4, 1e4)
        self.log_scales.nan_to_num_(nan=0.0, posinf=max_scale_mm, neginf=min_scale_mm)
        self.log_scales.clamp_(
            torch.log(torch.as_tensor(min_scale_mm, device=self.log_scales.device)),
            torch.log(torch.as_tensor(max_scale_mm, device=self.log_scales.device)),
        )
        self.logit_opacities.nan_to_num_(nan=0.0, posinf=10.0, neginf=-10.0)
        self.logit_opacities.clamp_(-10.0, 10.0)
        self.colors.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0)
        self.colors.clamp_(0.0, 1.0)
        self.disk_normals.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
        self.disk_normals.clamp_(-1.0, 1.0)
        self.acoustic_attenuation.clamp_(-10.0, 5.0)
        self.acoustic_reflection.clamp_(-10.0, 5.0)
        self.acoustic_scattering.clamp_(-10.0, 5.0)

    @torch.no_grad()
    def densify_and_prune(
        self,
        grad_threshold=1e-4,
        opacity_threshold=0.02,
        large_scale_threshold=2.0,
        split_factor=0.7,
        max_gaussians=10000,
        min_gaussians=100,
        max_new_gaussians=512,
    ):
        """
        Adapt the number of Gaussians during training.

        Prunes transparent Gaussians and splits large/high-gradient Gaussians.
        Because Parameter shapes change, recreate the optimizer after this call.
        """
        old_count = self.num_gaussians
        device = self.means.device

        opacities = torch.sigmoid(self.logit_opacities).squeeze(-1)
        keep = opacities >= opacity_threshold
        if keep.sum() < min_gaussians:
            best = torch.argsort(opacities, descending=True)[: min(min_gaussians, old_count)]
            keep = torch.zeros_like(opacities, dtype=torch.bool)
            keep[best] = True

        means = self.means.data[keep]
        log_scales = self.log_scales.data[keep]
        logit_opacities = self.logit_opacities.data[keep]
        colors = self.colors.data[keep]
        disk_normals = self.disk_normals.data[keep]

        grad = self.means.grad
        if grad is None:
            split_mask = torch.zeros(len(means), device=device, dtype=torch.bool)
        else:
            grad_norm = grad.detach().norm(dim=-1)[keep]
            scales = torch.exp(log_scales)
            large = scales.max(dim=-1).values >= large_scale_threshold
            split_mask = (grad_norm >= grad_threshold) & large

        remaining_capacity = max(max_gaussians - len(means), 0)
        split_indices = torch.nonzero(split_mask, as_tuple=False).squeeze(-1)
        if len(split_indices) > 0 and remaining_capacity > 0:
            split_indices = split_indices[: min(len(split_indices), remaining_capacity, max_new_gaussians)]
            parent_means = means[split_indices]
            parent_log_scales = log_scales[split_indices]
            parent_scales = torch.exp(parent_log_scales)
            offsets = torch.randn_like(parent_means) * parent_scales * 0.25

            child_means = parent_means + offsets
            child_log_scales = parent_log_scales + torch.log(
                torch.as_tensor(split_factor, device=device, dtype=parent_log_scales.dtype)
            )
            child_logit_opacities = logit_opacities[split_indices] - torch.log(
                torch.as_tensor(2.0, device=device, dtype=logit_opacities.dtype)
            )
            child_colors = colors[split_indices]
            child_disk_normals = disk_normals[split_indices]

            means[split_indices] = parent_means - offsets
            log_scales[split_indices] = child_log_scales
            logit_opacities[split_indices] = child_logit_opacities

            means = torch.cat([means, child_means], dim=0)
            log_scales = torch.cat([log_scales, child_log_scales], dim=0)
            logit_opacities = torch.cat([logit_opacities, child_logit_opacities], dim=0)
            colors = torch.cat([colors, child_colors], dim=0)
            disk_normals = torch.cat([disk_normals, child_disk_normals], dim=0)

        self.means = nn.Parameter(means.contiguous())
        self.log_scales = nn.Parameter(log_scales.contiguous())
        self.logit_opacities = nn.Parameter(logit_opacities.contiguous())
        self.colors = nn.Parameter(colors.contiguous())
        self.disk_normals = nn.Parameter(disk_normals.contiguous())

        return {
            "old_count": int(old_count),
            "new_count": int(len(means)),
            "pruned": int(old_count - keep.sum().item()),
            "split": int(len(split_indices)) if "split_indices" in locals() else 0,
        }

    @property
    def num_gaussians(self):
        return int(self.means.shape[0])


def normalize_image_for_save(image):
    image = image.detach().float().cpu()
    if image.ndim == 4:
        image = image.squeeze(0)
    if image.ndim == 2:
        image = image.unsqueeze(0)

    image = image - image.amin(dim=(-2, -1), keepdim=True)
    image = image / image.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    if image.shape[0] == 1:
        array = image.squeeze(0).numpy()
        return Image.fromarray((array * 255.0).astype(np.uint8), mode="L").convert("RGB")

    array = image[:3].permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")


def add_label(image, label):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 220), 20), fill=(0, 0, 0))
    draw.text((5, 4), label, fill=(255, 255, 255))
    return image


def save_debug_visuals(
    pred,
    target,
    step,
    output_dir,
    sobel_weight=1.5,
    filter_kernel_size=9,
    filter_sigma=1.0,
    confidence=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        raw_diff = (pred - target).abs()
        pred_edges = ultrasound_edge_map(
            pred,
            sigmas=(filter_sigma,),
            sobel_weight=sobel_weight,
            kernel_size=filter_kernel_size,
            sobel_blur_kernel_size=filter_kernel_size,
            sobel_blur_sigma=filter_sigma,
        ).squeeze(0)
        target_edges = ultrasound_edge_map(
            target,
            sigmas=(filter_sigma,),
            sobel_weight=sobel_weight,
            kernel_size=filter_kernel_size,
            sobel_blur_kernel_size=filter_kernel_size,
            sobel_blur_sigma=filter_sigma,
        ).squeeze(0)
        edge_diff = (pred_edges - target_edges).abs()

        panels = [
            ("target", target),
            ("rendered", pred),
            ("raw diff", raw_diff),
            ("target edges", target_edges),
            ("rendered edges", pred_edges),
            ("edge diff", edge_diff),
        ]
        if confidence is not None:
            panels.append(("confidence", confidence))

        images = [add_label(normalize_image_for_save(tensor), label) for label, tensor in panels]
        width = sum(image.width for image in images)
        height = max(image.height for image in images)
        montage = Image.new("RGB", (width, height), color=(0, 0, 0))

        x_offset = 0
        for image in images:
            montage.paste(image, (x_offset, 0))
            x_offset += image.width

        montage.save(output_dir / f"step_{step:06d}_debug.png")


def args_to_metadata(args):
    metadata = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            metadata[key] = str(value)
        else:
            metadata[key] = value
    return metadata


def save_checkpoint(
    model,
    output_path,
    args,
    step,
    final_loss=None,
    calibration_metadata=None,
    validation_history=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "num_gaussians": model.num_gaussians,
        "channels": int(model.colors.shape[-1]),
        "final_loss": None if final_loss is None else float(final_loss),
        "validation_history": validation_history or [],
        "calibration": calibration_metadata or {
            "image_t_probe": DEFAULT_IMAGE_T_PROBE,
            "image_plane_origin_px": [args.image_origin_x, args.image_origin_y],
            "pixel_to_mm": args.pixel_to_mm,
            "image_scale": args.image_scale,
        },
        "covariance": {
            "primitive_mode": args.primitive_mode,
            "mode": args.covariance_mode,
            "min_scale_mm": args.min_scale_mm,
            "max_scale_mm": args.max_scale_mm,
            "initial_opacity": args.initial_opacity,
            "scale_prior_lateral": args.scale_prior_lateral,
            "scale_prior_axial": args.scale_prior_axial,
            "scale_prior_elevational": args.scale_prior_elevational,
            "scale_prior_weight": args.scale_prior_weight,
            "lateral_depth_slope": args.lateral_depth_slope,
            "elevational_depth_slope": args.elevational_depth_slope,
            "acoustic_rendering": args.acoustic_rendering,
        },
        "loss": {
            "type": args.loss,
            "accumulation_steps": int(max(args.accumulation_steps, 1)),
            "validation_slices": int(max(args.validation_slices, 0)),
            "validation_fraction": float(args.validation_fraction),
            "validation_sources": list(args.validation_sources or []),
            "validation_every": int(max(args.validation_every, 0)),
            "validation_seed": int(args.validation_seed),
            "amp": bool(args.amp),
            "laplacian_weight": args.laplacian_loss_weight,
            "edge_weight": args.edge_loss_weight,
            "intensity_weight": args.intensity_loss_weight,
            "sobel_weight": args.sobel_loss_weight,
            "filter_kernel_size": args.filter_kernel_size,
            "filter_sigma": args.filter_sigma,
            "low_intensity_threshold_255": args.low_intensity_threshold,
            "content_normalize": args.content_normalize,
            "content_intensity_threshold": args.content_intensity_threshold,
            "content_feature_threshold": args.content_feature_threshold,
            "content_background_weight": args.content_background_weight,
            "use_confidence": args.use_confidence,
            "background_threshold": args.confidence_background_threshold,
            "background_weight": args.confidence_background_weight,
            "dark_threshold": args.confidence_dark_threshold,
            "shadow_weight": args.confidence_shadow_weight,
            "bright_threshold": args.confidence_bright_threshold,
            "shadow_start_offset": args.confidence_shadow_start_offset,
            "shadow_confidence": args.shadow_confidence,
        },
        "densification": {
            "init_jitter_voxels": args.init_jitter_voxels,
            "densify_every": args.densify_every,
            "densify_start": args.densify_start,
            "densify_grad_threshold": args.densify_grad_threshold,
            "prune_opacity_threshold": args.prune_opacity_threshold,
            "split_scale_threshold": args.split_scale_threshold,
            "split_scale_factor": args.split_scale_factor,
            "max_gaussians": args.max_gaussians,
            "min_gaussians": args.min_gaussians,
            "max_new_gaussians": args.max_new_gaussians,
        },
        "args": args_to_metadata(args),
    }

    torch.save(checkpoint, output_path)

    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path


def checkpoint_path_for_step(output_path, step):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_step_{int(step):06d}{output_path.suffix}")


def init_SVRTK(
    dataset,
    num_gaussians,
    grid_size=(64, 64, 64),
    pixel_spacing=(1.0, 1.0),
    pixel_stride=2,
    intensity_threshold=0.05,
    image_t_probe=DEFAULT_IMAGE_T_PROBE,
    image_plane_origin_px=DEFAULT_IMAGE_PLANE_ORIGIN_PX,
    pixel_to_mm=DEFAULT_PIXEL_TO_MM,
    image_scale=1.0,
    jitter_voxels=0.5,
    device="cpu",
):
    """
    Build an SVRTK-style initialization from tracked 2D slices.

    The function maps slice pixels into 3D with x = T_k y, averages the
    scattered intensities into a voxel grid, then seeds Gaussian centers from
    high-intensity voxels in that initial volume X0.
    """
    coords_per_slice = []
    values_per_slice = []
    bounds_min = None
    bounds_max = None

    for sample in dataset:
        image = sample["image"].to(device)
        pose = sample["pose"].to(device)
        coords, values = slice_pixels_to_world(
            image,
            pose,
            pixel_spacing=pixel_spacing,
            pixel_stride=pixel_stride,
            image_t_probe=image_t_probe,
            image_plane_origin_px=image_plane_origin_px,
            pixel_to_mm=pixel_to_mm,
            image_scale=image_scale,
        )

        coords_per_slice.append(coords)
        values_per_slice.append(values)
        current_min = coords.min(dim=0).values
        current_max = coords.max(dim=0).values
        bounds_min = current_min if bounds_min is None else torch.minimum(bounds_min, current_min)
        bounds_max = current_max if bounds_max is None else torch.maximum(bounds_max, current_max)

    padding = (bounds_max - bounds_min).clamp_min(1e-6) * 0.02
    bounds_min = bounds_min - padding
    bounds_max = bounds_max + padding

    volume, counts = scatter_slices_to_volume(
        coords_per_slice,
        values_per_slice,
        bounds_min,
        bounds_max,
        grid_size,
    )
    initial_volume = volume / counts.clamp_min(1.0)

    means, colors = sample_gaussians_from_volume(
        initial_volume,
        counts,
        bounds_min,
        bounds_max,
        num_gaussians,
        intensity_threshold=intensity_threshold,
        jitter_voxels=jitter_voxels,
    )

    return {
        "means": means.cpu(),
        "colors": colors.cpu(),
        "volume": initial_volume.cpu(),
        "counts": counts.cpu(),
        "bounds_min": bounds_min.cpu(),
        "bounds_max": bounds_max.cpu(),
    }


def slice_pixels_to_world(
    image,
    pose,
    pixel_spacing=(1.0, 1.0),
    pixel_stride=2,
    image_t_probe=DEFAULT_IMAGE_T_PROBE,
    image_plane_origin_px=DEFAULT_IMAGE_PLANE_ORIGIN_PX,
    pixel_to_mm=DEFAULT_PIXEL_TO_MM,
    image_scale=1.0,
):
    """Map sampled pixels from one image slice into 3D world coordinates."""
    if image.ndim == 2:
        image = image.unsqueeze(0)

    dtype = image.dtype
    channels, height, width = image.shape
    step = max(int(pixel_stride), 1)
    y_idx = torch.arange(0, height, step, device=image.device, dtype=dtype)
    x_idx = torch.arange(0, width, step, device=image.device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_idx, x_idx, indexing="ij")

    origin_px = torch.as_tensor(image_plane_origin_px, device=image.device, dtype=dtype)
    calibrated_spacing = pixel_to_mm * image_scale
    if calibrated_spacing > 0.0:
        spacing_x = calibrated_spacing
        spacing_y = calibrated_spacing
    else:
        spacing_x, spacing_y = pixel_spacing

    image_x = (grid_x - origin_px[0]) * spacing_x
    image_y = (grid_y - origin_px[1]) * spacing_y
    image_z = torch.zeros_like(image_x)
    ones = torch.ones_like(image_x)
    image_points = torch.stack([image_x, image_y, image_z, ones], dim=-1).reshape(-1, 4)

    image_t_probe = torch.as_tensor(image_t_probe, device=image.device, dtype=dtype)
    probe_t_image = torch.linalg.inv(image_t_probe)
    probe_points = (probe_t_image @ image_points.T)
    world_points = (pose @ probe_points).T[:, :3]
    sampled_image = image[:, ::step, ::step].reshape(channels, -1).T
    return world_points, sampled_image


def scatter_slices_to_volume(coords_per_slice, values_per_slice, bounds_min, bounds_max, grid_size):
    """Scattered-data interpolation by averaging all registered slice samples."""
    channels = values_per_slice[0].shape[-1]
    depth, height, width = grid_size
    volume = torch.zeros((channels, depth, height, width), device=bounds_min.device)
    counts = torch.zeros((1, depth, height, width), device=bounds_min.device)
    grid_shape = torch.tensor([width, height, depth], device=bounds_min.device, dtype=torch.float32)
    extent = (bounds_max - bounds_min).clamp_min(1e-6)

    for coords, values in zip(coords_per_slice, values_per_slice):
        normalized = (coords - bounds_min) / extent
        ijk = torch.floor(normalized * (grid_shape - 1)).long()
        valid = ((ijk >= 0) & (ijk < grid_shape.long())).all(dim=-1)
        if not valid.any():
            continue

        ijk = ijk[valid]
        values = values[valid]
        linear = ijk[:, 2] * height * width + ijk[:, 1] * width + ijk[:, 0]

        volume_flat = volume.view(channels, -1)
        counts_flat = counts.view(1, -1)
        volume_flat.index_add_(1, linear, values.T)
        counts_flat.index_add_(1, linear, torch.ones(1, len(linear), device=counts.device))

    return volume, counts


def sample_gaussians_from_volume(
    volume,
    counts,
    bounds_min,
    bounds_max,
    num_gaussians,
    intensity_threshold=0.05,
    jitter_voxels=0.5,
):
    """Choose Gaussian centers from occupied/high-intensity voxels of X0."""
    channels, depth, height, width = volume.shape
    intensity = volume.mean(dim=0)
    occupied = counts.squeeze(0) > 0
    candidate = occupied & (intensity >= intensity_threshold)
    if not candidate.any():
        candidate = occupied
    if not candidate.any():
        raise ValueError("SVRTK initialization found no slice samples inside the volume")

    scores = intensity[candidate].clamp_min(1e-6)
    candidate_indices = candidate.nonzero(as_tuple=False)
    if len(candidate_indices) >= num_gaussians:
        chosen = torch.multinomial(scores, num_gaussians, replacement=False)
    else:
        chosen = torch.multinomial(scores, num_gaussians, replacement=True)

    grid_denominator = torch.tensor(
        [max(depth - 1, 1), max(height - 1, 1), max(width - 1, 1)],
        device=volume.device,
        dtype=torch.float32,
    )
    jitter_voxels = max(float(jitter_voxels), 0.0)
    zyx = candidate_indices[chosen].float()
    if jitter_voxels > 0.0:
        jitter = (torch.rand_like(zyx) - 0.5) * jitter_voxels
        zyx = (zyx + jitter).clamp(
            min=torch.zeros(3, device=volume.device, dtype=torch.float32),
            max=grid_denominator,
        )
    xyz_normalized = torch.stack(
        [zyx[:, 2], zyx[:, 1], zyx[:, 0]],
        dim=-1,
    ) / grid_denominator[[2, 1, 0]]

    means = bounds_min + xyz_normalized * (bounds_max - bounds_min)
    colors = volume[:, candidate].T[chosen]
    return means, colors.clamp(0.0, 1.0)


def choose_source_validation_indices(dataset, validation_sources):
    if not validation_sources:
        return None
    if not hasattr(dataset, "datasets") or not hasattr(dataset, "cumulative_lengths"):
        raise ValueError("--validation-sources requires multiple --image-dir sources")

    requested = [str(source).lower() for source in validation_sources]
    matched_sources = []
    for source_index, source_dataset in enumerate(dataset.datasets):
        source_name = str(source_dataset.source_name).lower()
        for request in requested:
            index_match = request == str(source_index) or request == str(source_index + 1)
            name_match = request in source_name
            if index_match or name_match:
                matched_sources.append(source_index)
                break

    if not matched_sources:
        available = [
            f"{index}:{source_dataset.source_name}"
            for index, source_dataset in enumerate(dataset.datasets)
        ]
        raise ValueError(
            "No validation sources matched. Available sources: "
            + "; ".join(available)
        )

    validation_indices = []
    training_indices = []
    previous = 0
    matched_set = set(matched_sources)
    for source_index, length in enumerate(dataset.lengths):
        indices = list(range(previous, previous + length))
        if source_index in matched_set:
            validation_indices.extend(indices)
        else:
            training_indices.extend(indices)
        previous += length

    if not training_indices:
        raise ValueError("--validation-sources matched every source; no training slices remain")
    if not validation_indices:
        raise ValueError("--validation-sources did not select any validation slices")
    return validation_indices, training_indices, matched_sources


def choose_validation_indices(dataset_size, validation_slices, validation_fraction, seed):
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError("--validation-fraction must be in [0, 1).")

    validation_slices = max(int(validation_slices), 0)
    if validation_slices == 0 and validation_fraction > 0.0:
        validation_slices = max(1, round(dataset_size * validation_fraction))

    if validation_slices == 0:
        return [], list(range(dataset_size))
    if dataset_size < 2:
        raise ValueError("Validation requires at least 2 total slices")

    validation_count = min(validation_slices, dataset_size - 1)
    generator = torch.Generator().manual_seed(int(seed))
    shuffled = torch.randperm(dataset_size, generator=generator).tolist()
    validation_indices = sorted(shuffled[:validation_count])
    validation_set = set(validation_indices)
    training_indices = [index for index in range(dataset_size) if index not in validation_set]
    return validation_indices, training_indices


def render_prediction(
    model,
    pose,
    args,
    calibration_kwargs,
    image_origin_x,
    image_origin_y,
    pixel_to_mm,
    image_scale,
):
    return model(
        pose,
        args.height,
        args.width,
        pixel_spacing=(args.pixel_spacing_x, args.pixel_spacing_y),
        slice_thickness=args.slice_thickness,
        shadowing=args.shadowing,
        shadow_strength=args.shadow_strength,
        image_t_probe=calibration_kwargs["image_t_probe"],
        image_plane_origin_px=(image_origin_x, image_origin_y),
        pixel_to_mm=pixel_to_mm,
        image_scale=image_scale,
        covariance_mode=args.covariance_mode,
        min_scale_mm=args.min_scale_mm,
        max_scale_mm=args.max_scale_mm,
        lateral_depth_slope=args.lateral_depth_slope,
        elevational_depth_slope=args.elevational_depth_slope,
        render_chunk_size=args.render_chunk_size,
        primitive_mode=args.primitive_mode,
        acoustic_rendering=args.acoustic_rendering,
    )


def prediction_loss(pred, target, args, model=None):
    if args.loss == "ultrasound_edges":
        loss = ultrasound_edge_loss(
            pred,
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
    else:
        raw_weight = None
        if args.content_normalize:
            raw_weight = content_weight_map(
                target,
                intensity_threshold=args.content_intensity_threshold,
                background_weight=args.content_background_weight,
            )
        if raw_weight is None:
            loss = F.l1_loss(pred, target)
        else:
            raw_error = (pred - target).abs()
            if raw_weight.shape[1] == 1 and raw_error.shape[1] != 1:
                raw_weight = raw_weight.expand(-1, raw_error.shape[1], -1, -1)
            loss = (raw_error * raw_weight).sum() / raw_weight.sum().clamp_min(1e-8)

    if model is not None and args.scale_prior_weight > 0.0:
        loss = loss + args.scale_prior_weight * model.scale_prior_loss(
            prior_scales=(
                args.scale_prior_lateral,
                args.scale_prior_axial,
                args.scale_prior_elevational,
            ),
            min_scale_mm=args.min_scale_mm,
            max_scale_mm=args.max_scale_mm,
        )
    return loss


@torch.no_grad()
def evaluate_validation_loss(
    model,
    validation_loader,
    args,
    device,
    calibration_kwargs,
    image_origin_x,
    image_origin_y,
    pixel_to_mm,
    image_scale,
    use_amp,
):
    if validation_loader is None:
        return None

    was_training = model.training
    model.eval()
    losses = []
    for batch in validation_loader:
        target = batch["image"].squeeze(0).to(device)
        pose = batch["pose"].squeeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = render_prediction(
                model,
                pose,
                args,
                calibration_kwargs,
                image_origin_x,
                image_origin_y,
                pixel_to_mm,
                image_scale,
            )
            loss = prediction_loss(pred, target, args, model=None)
        losses.append(float(loss.detach().item()))

    if was_training:
        model.train()
    return sum(losses) / max(len(losses), 1)


def train(args):
    if args.filter_kernel_size % 2 == 0:
        raise ValueError("--filter-kernel-size must be odd")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    image_dirs = args.image_dir
    poses_paths = args.poses
    if poses_paths and len(poses_paths) not in {0, len(image_dirs)}:
        raise ValueError(
            f"--poses must be omitted or provide one path per --image-dir entry; "
            f"got {len(poses_paths)} poses for {len(image_dirs)} image sources"
        )
    if len(image_dirs) == 1:
        dataset = TrackedUltrasoundDataset(
            image_dir=image_dirs[0],
            poses_path=poses_paths[0] if poses_paths else None,
            image_size=(args.height, args.width),
            grayscale=args.grayscale,
            low_intensity_threshold=args.low_intensity_threshold,
        )
    else:
        dataset = MultiTrackedUltrasoundDataset(
            image_dirs=image_dirs,
            poses_paths=poses_paths,
            image_size=(args.height, args.width),
            grayscale=args.grayscale,
            low_intensity_threshold=args.low_intensity_threshold,
        )
    source_validation = choose_source_validation_indices(dataset, args.validation_sources)
    validation_source_names = []
    if source_validation is None:
        validation_indices, training_indices = choose_validation_indices(
            len(dataset),
            args.validation_slices,
            args.validation_fraction,
            args.validation_seed,
        )
    else:
        validation_indices, training_indices, validation_source_indices = source_validation
        validation_source_names = [
            dataset.datasets[index].source_name for index in validation_source_indices
        ]
    train_dataset = Subset(dataset, training_indices) if validation_indices else dataset
    validation_loader = (
        DataLoader(Subset(dataset, validation_indices), batch_size=1, shuffle=False)
        if validation_indices
        else None
    )
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    if validation_indices:
        print(
            f"using {len(training_indices)} training slices and "
            f"{len(validation_indices)} fixed validation slices"
        )
        if validation_source_names:
            print("validation sources: " + "; ".join(validation_source_names))

    calibration_kwargs = dataset.calibration.to_renderer_kwargs()
    image_origin = calibration_kwargs["image_plane_origin_px"]
    pixel_to_mm = calibration_kwargs["pixel_to_mm"] if args.pixel_to_mm is None else args.pixel_to_mm
    image_scale = calibration_kwargs["image_scale"] if args.image_scale is None else args.image_scale
    image_origin_x = image_origin[0] if args.image_origin_x is None else args.image_origin_x
    image_origin_y = image_origin[1] if args.image_origin_y is None else args.image_origin_y
    checkpoint_calibration_metadata = {
        "image_t_probe": calibration_kwargs["image_t_probe"].tolist(),
        "image_plane_origin_px": [image_origin_x, image_origin_y],
        "pixel_to_mm": pixel_to_mm,
        "image_scale": image_scale,
    }

    channels = 1 if args.grayscale else 3
    initial_means = None
    initial_colors = None
    if args.init == "svrtk":
        svrtk_init = init_SVRTK(
            train_dataset,
            args.num_gaussians,
            grid_size=(args.grid_depth, args.grid_height, args.grid_width),
            pixel_spacing=(args.pixel_spacing_x, args.pixel_spacing_y),
            pixel_stride=args.pixel_stride,
            intensity_threshold=args.intensity_threshold,
            image_t_probe=calibration_kwargs["image_t_probe"],
            image_plane_origin_px=(image_origin_x, image_origin_y),
            pixel_to_mm=pixel_to_mm,
            image_scale=image_scale,
            jitter_voxels=args.init_jitter_voxels,
            device=device,
        )
        initial_means = svrtk_init["means"]
        initial_colors = svrtk_init["colors"]
        print(f"initialized {len(initial_means)} Gaussians from SVRTK volume")

    model = VanillaGaussianSplatting(
        args.num_gaussians,
        channels=channels,
        initial_scale=(args.initial_scale_x, args.initial_scale_y, args.initial_scale_z),
        initial_opacity=args.initial_opacity,
        initial_means=initial_means,
        initial_colors=initial_colors,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    accumulation_steps = max(int(args.accumulation_steps), 1)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    validation_history = []

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_count = 0
        debug_pred = None
        debug_target = None

        for accumulation_index, batch in zip(range(accumulation_steps), loader):
            target = batch["image"].squeeze(0).to(device)
            pose = batch["pose"].squeeze(0).to(device)
            model.stabilize_parameters(
                min_scale_mm=args.min_scale_mm,
                max_scale_mm=args.max_scale_mm,
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = render_prediction(
                    model,
                    pose,
                    args,
                    calibration_kwargs,
                    image_origin_x,
                    image_origin_y,
                    pixel_to_mm,
                    image_scale,
                )
                loss = prediction_loss(pred, target, args, model=model)

            accumulated_loss += float(loss.detach().item())
            accumulated_count += 1
            if accumulation_index == 0:
                debug_pred = pred.detach()
                debug_target = target.detach()
            scaler.scale(loss).backward()

        if accumulated_count == 0:
            raise RuntimeError("No training slices were available from the data loader.")

        scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_count)

        loss = torch.as_tensor(accumulated_loss / accumulated_count, device=device)
        scaler.step(optimizer)
        scaler.update()
        model.stabilize_parameters(
            min_scale_mm=args.min_scale_mm,
            max_scale_mm=args.max_scale_mm,
        )

        if (
            args.densify_every > 0
            and step >= args.densify_start
            and step % args.densify_every == 0
        ):
            stats = model.densify_and_prune(
                grad_threshold=args.densify_grad_threshold,
                opacity_threshold=args.prune_opacity_threshold,
                large_scale_threshold=args.split_scale_threshold,
                split_factor=args.split_scale_factor,
                max_gaussians=args.max_gaussians,
                min_gaussians=args.min_gaussians,
                max_new_gaussians=args.max_new_gaussians,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            if args.log_densify:
                print(
                    "densify/prune "
                    f"step={step} old={stats['old_count']} new={stats['new_count']} "
                    f"pruned={stats['pruned']} split={stats['split']}"
                )

        if (
            debug_pred is not None
            and debug_target is not None
            and args.debug_every > 0
            and (step == 1 or step % args.debug_every == 0)
        ):
            confidence = None
            if args.use_confidence:
                confidence = ultrasound_confidence_map(
                    debug_target,
                    background_threshold=args.confidence_background_threshold,
                    background_weight=args.confidence_background_weight,
                    dark_threshold=args.confidence_dark_threshold,
                    shadow_weight=args.confidence_shadow_weight,
                    bright_threshold=args.confidence_bright_threshold,
                    shadow_start_offset=args.confidence_shadow_start_offset,
                    enable_shadow=args.shadow_confidence,
                ).squeeze(0)
            save_debug_visuals(
                debug_pred,
                debug_target,
                step,
                args.debug_dir,
                sobel_weight=args.sobel_loss_weight,
                filter_kernel_size=args.filter_kernel_size,
                filter_sigma=args.filter_sigma,
                confidence=confidence,
            )

        validation_loss = None
        if (
            validation_loader is not None
            and args.validation_every > 0
            and (step == 1 or step % args.validation_every == 0)
        ):
            validation_loss = evaluate_validation_loss(
                model,
                validation_loader,
                args,
                device,
                calibration_kwargs,
                image_origin_x,
                image_origin_y,
                pixel_to_mm,
                image_scale,
                use_amp,
            )
            validation_history.append(
                {"step": int(step), "loss": float(validation_loss)}
            )

        if validation_loss is not None:
            print(
                f"step={step:04d} loss={loss.item():.6f} "
                f"val_loss={validation_loss:.6f} gaussians={model.num_gaussians}"
            )
        elif step == 1 or step % args.log_every == 0:
            print(f"step={step:04d} loss={loss.item():.6f} gaussians={model.num_gaussians}")

        if args.output and args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            step_output = checkpoint_path_for_step(args.output, step)
            output_path, metadata_path = save_checkpoint(
                model,
                step_output,
                args,
                step=step,
                final_loss=loss.item(),
                validation_history=validation_history,
                calibration_metadata=checkpoint_calibration_metadata,
            )
            print(f"saved step checkpoint {output_path}")
            print(f"saved step metadata {metadata_path}")

    if args.output:
        output_path, metadata_path = save_checkpoint(
            model,
            args.output,
            args,
            step=args.steps,
            final_loss=loss.item() if "loss" in locals() else None,
            validation_history=validation_history,
            calibration_metadata=checkpoint_calibration_metadata,
        )
        print(f"saved checkpoint {output_path}")
        print(f"saved metadata {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a vanilla PyTorch Gaussian splatting model on images.")
    parser.add_argument("--image-dir", required=True, nargs="+", help="One or more image folders or tracked .mha/.igs.mha sequences.")
    parser.add_argument("--poses", nargs="*", default=None, help="Optional .npy/.csv pose file for each image source. Leave unset for .igs.mha embedded poses.")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-gaussians", type=int, default=512)
    parser.add_argument("--init", choices=("random", "svrtk"), default="random")
    parser.add_argument(
        "--init-jitter-voxels",
        type=float,
        default=0.5,
        help="Randomly jitter SVRTK-initialized Gaussian centers by this many grid voxels.",
    )
    parser.add_argument("--grid-depth", type=int, default=64)
    parser.add_argument("--grid-height", type=int, default=64)
    parser.add_argument("--grid-width", type=int, default=64)
    parser.add_argument("--pixel-spacing-x", type=float, default=1.0)
    parser.add_argument("--pixel-spacing-y", type=float, default=1.0)
    parser.add_argument("--pixel-to-mm", type=float, default=None)
    parser.add_argument("--image-scale", type=float, default=None)
    parser.add_argument("--image-origin-x", type=float, default=None)
    parser.add_argument("--image-origin-y", type=float, default=None)
    parser.add_argument("--slice-thickness", type=float, default=1.0)
    parser.add_argument("--initial-scale-x", "--initial-scale-lateral", dest="initial_scale_x", type=float, default=1.0)
    parser.add_argument("--initial-scale-y", "--initial-scale-axial", dest="initial_scale_y", type=float, default=0.5)
    parser.add_argument("--initial-scale-z", "--initial-scale-elevational", dest="initial_scale_z", type=float, default=2.0)
    parser.add_argument(
        "--initial-opacity",
        type=float,
        default=0.5,
        help="Initial per-Gaussian opacity probability. Lower values reduce early blob saturation.",
    )
    parser.add_argument("--primitive-mode", choices=("volume", "disk", "dot"), default="volume")
    parser.add_argument("--covariance-mode", choices=("ultrasound_psf", "world_axis_aligned"), default="ultrasound_psf")
    parser.add_argument("--min-scale-mm", type=float, default=0.05)
    parser.add_argument("--max-scale-mm", type=float, default=10.0)
    parser.add_argument("--scale-prior-lateral", type=float, default=1.0)
    parser.add_argument("--scale-prior-axial", type=float, default=0.5)
    parser.add_argument("--scale-prior-elevational", type=float, default=2.0)
    parser.add_argument("--scale-prior-weight", type=float, default=1e-3)
    parser.add_argument("--lateral-depth-slope", type=float, default=0.0)
    parser.add_argument("--elevational-depth-slope", type=float, default=0.0)
    parser.add_argument("--acoustic-rendering", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-chunk-size", type=int, default=128)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--intensity-threshold", type=float, default=0.05)
    parser.add_argument("--shadowing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shadow-strength", type=float, default=1.0)
    parser.add_argument("--loss", choices=("ultrasound_edges", "raw_l1"), default="ultrasound_edges")
    parser.add_argument("--laplacian-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--intensity-loss-weight", type=float, default=0.05)
    parser.add_argument("--sobel-loss-weight", type=float, default=1.5)
    parser.add_argument("--filter-kernel-size", type=int, default=9)
    parser.add_argument("--filter-sigma", type=float, default=1.0)
    parser.add_argument(
        "--low-intensity-threshold",
        type=float,
        default=None,
        help="Set pixels below this grayscale value to 0 during data loading, using a 0-255 scale. Example: 25.",
    )
    parser.add_argument("--content-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--content-intensity-threshold", type=float, default=0.03)
    parser.add_argument("--content-feature-threshold", type=float, default=0.05)
    parser.add_argument("--content-background-weight", type=float, default=0.05)
    parser.add_argument("--use-confidence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--confidence-background-threshold", type=float, default=0.02)
    parser.add_argument("--confidence-background-weight", type=float, default=0.0)
    parser.add_argument("--confidence-dark-threshold", type=float, default=0.08)
    parser.add_argument("--confidence-shadow-weight", type=float, default=0.2)
    parser.add_argument("--confidence-bright-threshold", type=float, default=0.65)
    parser.add_argument("--confidence-shadow-start-offset", type=int, default=8)
    parser.add_argument("--shadow-confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Number of shuffled slices to average before each optimizer update.",
    )
    parser.add_argument(
        "--validation-slices",
        type=int,
        default=0,
        help="Hold out this many fixed slices and report their loss during training.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help=(
            "Hold out this fraction of slices for validation. Ignored when "
            "--validation-slices is greater than 0."
        ),
    )
    parser.add_argument(
        "--validation-sources",
        nargs="*",
        default=None,
        help=(
            "Hold out entire image sources for validation. Values can be "
            "0-based/1-based source indices or substrings such as right3. "
            "Overrides --validation-slices and --validation-fraction."
        ),
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=100,
        help="Evaluate fixed validation slices every N optimizer steps.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=1234,
        help="Seed used to choose fixed validation slice indices.",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--densify-every", type=int, default=0)
    parser.add_argument("--densify-start", type=int, default=100)
    parser.add_argument("--densify-grad-threshold", type=float, default=1e-4)
    parser.add_argument("--prune-opacity-threshold", type=float, default=0.02)
    parser.add_argument("--split-scale-threshold", type=float, default=2.0)
    parser.add_argument("--split-scale-factor", type=float, default=0.7)
    parser.add_argument("--max-gaussians", type=int, default=10000)
    parser.add_argument("--min-gaussians", type=int, default=100)
    parser.add_argument("--max-new-gaussians", type=int, default=512)
    parser.add_argument("--log-densify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-every", type=int, default=0)
    parser.add_argument("--debug-dir", default="outputs/debug_visuals")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA automatic mixed precision to reduce GPU memory.",
    )
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--output", default="outputs/checkpoints/vanilla_gaussians.pt")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save an extra .pt checkpoint and .json metadata every N optimizer steps. 0 disables periodic saves.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
