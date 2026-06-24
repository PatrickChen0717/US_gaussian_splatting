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
from ultrasound_losses import ultrasound_confidence_map, ultrasound_edge_loss, ultrasound_edge_map
from ultrasound_projection import (
    DEFAULT_IMAGE_PLANE_ORIGIN_PX,
    DEFAULT_IMAGE_T_PROBE,
    DEFAULT_PIXEL_TO_MM,
    render_ultrasound_gaussians,
)
from ultragray_cuda_backend import render_ultrasound_cuda


TRAIN_CONFIG_DEFAULTS = {
    "image_dir": [],
    "poses": None,
    "image_dir_val": None,
    "pose_val": None,
    "height": 128,
    "width": 128,
    "num_gaussians": 512,
    "init": "random",
    "init_jitter_voxels": 0.5,
    "svrtk_fill_fraction": 0.2,
    "svrtk_fill_opacity": 0.05,
    "svrtk_fill_endpoint_margin": 0.1,
    "svrtk_intensity_power": 2.0,
    "grid_depth": 64,
    "grid_height": 64,
    "grid_width": 64,
    "pixel_spacing_x": 1.0,
    "pixel_spacing_y": 1.0,
    "pixel_to_mm": None,
    "image_scale": None,
    "image_origin_x": None,
    "image_origin_y": None,
    "pose_correction": "none",
    "pose_correction_max_translation_mm": 2.0,
    "pose_correction_max_rotation_deg": 2.0,
    "pose_correction_weight": 0.01,
    "pose_correction_lr": 5e-4,
    "slice_thickness": 1.0,
    "initial_scale_x": 1.0,
    "initial_scale_y": 0.5,
    "initial_scale_z": 2.0,
    "initial_opacity": 0.5,
    "initial_transmittance": 0.99,
    "primitive_mode": "volume",
    "covariance_mode": "ultrasound_psf",
    "min_scale_mm": 0.05,
    "max_scale_mm": 10.0,
    "scale_prior_lateral": 1.0,
    "scale_prior_axial": 0.5,
    "scale_prior_elevational": 2.0,
    "scale_prior_weight": 1e-3,
    "lateral_depth_slope": 0.0,
    "elevational_depth_slope": 0.0,
    "acoustic_rendering": False,
    "renderer_backend": "torch",
    "ultragray_repo_path": None,
    "cuda_tile_size_x": 4,
    "cuda_tile_size_y": 128,
    "native_pose_translation_scale": 0.1,
    "native_pose_convention": "probe_to_world",
    "native_ultragray_exact": False,
    "native_ultragray_loader_exact": True,
    "native_global_scale": 1.0,
    "native_init_extent": 0.8,
    "native_init_scale": 0.05,
    "native_grow_scale3d": 0.01,
    "native_prune_scale3d": 0.1,
    "native_prune_scale3d_min": 1e-5,
    "random_seed": 42,
    "ultrasound_far_plane": 5.0,
    "ultrasound_opening_width": None,
    "pose_sideways_noise": 0.0,
    "pose_frontback_noise": 0.2,
    "pose_updown_noise": 0.0,
    "render_chunk_size": 128,
    "pixel_stride": 2,
    "intensity_threshold": 0.05,
    "shadowing": True,
    "shadow_strength": 1.0,
    "max_visible_gaussians_per_slice": None,
    "laplacian_loss_weight": 1.0,
    "edge_loss_weight": 1.0,
    "intensity_loss_weight": 0.05,
    "sobel_loss_weight": 1.5,
    "ultrasound_edges_weight": 0.8,
    "l1_weight": 0.5,
    "ssim_weight": 0.2,
    "ssim_window_size": 11,
    "opacity_sparsity_weight": 0.0,
    "filter_kernel_size": 9,
    "filter_sigma": 1.0,
    "low_intensity_threshold": None,
    "image_value_scale": 255.0,
    "content_normalize": True,
    "content_intensity_threshold": 0.03,
    "content_feature_threshold": 0.05,
    "content_background_weight": 0.05,
    "use_confidence": False,
    "confidence_background_threshold": 0.02,
    "confidence_background_weight": 0.0,
    "confidence_dark_threshold": 0.08,
    "confidence_shadow_weight": 0.2,
    "confidence_bright_threshold": 0.65,
    "confidence_shadow_start_offset": 8,
    "shadow_confidence": True,
    "steps": 500,
    "batch_size": 8,
    "accumulation_steps": 1,
    "validation_slices": 0,
    "validation_fraction": 0.0,
    "validation_sources": None,
    "validation_every": 100,
    "validation_seed": 1234,
    "lr": 1e-2,
    "means_lr": 1e-4,
    "scales_lr": 5e-3,
    "quats_lr": 5e-3,
    "transmittances_lr": 5e-4,
    "intensity_lr": 5e-3,
    "sh_rest_lr": 1e-5,
    "lr_final_factor": 0.1,
    "sh_degree": 1,
    "sh_degree_interval": 1000,
    "log_every": 25,
    "densify_every": 0,
    "densify_start": 100,
    "densify_stop": 20000,
    "densify_grad_threshold": 1e-4,
    "prune_opacity_threshold": 0.02,
    "split_scale_threshold": 2.0,
    "split_scale_factor": 0.7,
    "max_gaussians": 10000,
    "min_gaussians": 100,
    "max_new_gaussians": 512,
    "log_densify": True,
    "debug_every": 0,
    "debug_dir": "outputs/debug_visuals",
    "device": "cuda",
    "amp": False,
    "grayscale": False,
    "output": "outputs/checkpoints/vanilla_gaussians.pt",
    "checkpoint_every": 0,
    "export_sog": True,
    "splat_transform": "splat-transform",
    "sog_gpu": None,
}


def load_train_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a JSON object")

    unknown_keys = sorted(set(loaded) - set(TRAIN_CONFIG_DEFAULTS))
    if unknown_keys:
        raise ValueError(
            f"{config_path} contains unknown config keys: {', '.join(unknown_keys)}"
        )

    config = dict(TRAIN_CONFIG_DEFAULTS)
    config.update(loaded)
    if not config["image_dir"]:
        raise ValueError(f"{config_path} must set image_dir to one or more inputs")
    if isinstance(config["image_dir"], str):
        config["image_dir"] = [config["image_dir"]]
    if config["poses"] is not None and isinstance(config["poses"], str):
        config["poses"] = [config["poses"]]
    if config["image_dir_val"] is not None and isinstance(config["image_dir_val"], str):
        config["image_dir_val"] = [config["image_dir_val"]]
    if config["pose_val"] is not None and isinstance(config["pose_val"], str):
        config["pose_val"] = [config["pose_val"]]
    if config["validation_sources"] is not None and isinstance(config["validation_sources"], str):
        config["validation_sources"] = [config["validation_sources"]]
    if config["pose_val"] and not config["image_dir_val"]:
        raise ValueError(f"{config_path} sets pose_val without image_dir_val")
    if config["image_dir_val"] and config["pose_val"] is not None:
        if len(config["pose_val"]) != len(config["image_dir_val"]):
            raise ValueError(
                f"{config_path} must provide one pose_val per image_dir_val; "
                f"got {len(config['pose_val'])} poses for "
                f"{len(config['image_dir_val'])} image sources"
            )

    def resolve_path(value):
        if value is None:
            return None
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        return str((config_path.parent / path).resolve())

    config["image_dir"] = [resolve_path(path) for path in config["image_dir"]]
    if config["poses"] is not None:
        config["poses"] = [resolve_path(path) for path in config["poses"]]
    if config["image_dir_val"] is not None:
        config["image_dir_val"] = [
            resolve_path(path) for path in config["image_dir_val"]
        ]
    if config["pose_val"] is not None:
        config["pose_val"] = [resolve_path(path) for path in config["pose_val"]]
    if config["output"]:
        config["output"] = resolve_path(config["output"])
    if config["debug_dir"]:
        config["debug_dir"] = resolve_path(config["debug_dir"])
    if config["ultragray_repo_path"]:
        config["ultragray_repo_path"] = resolve_path(config["ultragray_repo_path"])

    config["config_path"] = str(config_path)
    config["loss_config_path"] = str(config_path)
    if config["init"] not in {"random", "svrtk"}:
        raise ValueError("config init must be either 'random' or 'svrtk'")
    if config["covariance_mode"] not in {"ultrasound_psf", "world_axis_aligned", "full_cholesky"}:
        raise ValueError(
            "config covariance_mode must be 'ultrasound_psf', "
            "'world_axis_aligned', or 'full_cholesky'"
        )
    if config["renderer_backend"] not in {"torch", "ultragray_cuda"}:
        raise ValueError("config renderer_backend must be 'torch' or 'ultragray_cuda'")
    if int(config["cuda_tile_size_x"]) < 1 or int(config["cuda_tile_size_y"]) < 1:
        raise ValueError("CUDA tile sizes must be positive")
    if (
        config["renderer_backend"] == "ultragray_cuda"
        and config["covariance_mode"] != "ultrasound_psf"
    ):
        raise ValueError(
            "ultragray_cuda currently supports covariance_mode='ultrasound_psf'"
        )
    if config["renderer_backend"] == "ultragray_cuda" and config["primitive_mode"] != "volume":
        raise ValueError("ultragray_cuda currently supports primitive_mode='volume'")
    if config["renderer_backend"] == "ultragray_cuda" and config["acoustic_rendering"]:
        raise ValueError(
            "ultragray_cuda already applies its acoustic echo/transmittance model; "
            "set acoustic_rendering to false"
        )
    if not 0.0 <= float(config["svrtk_fill_fraction"]) <= 1.0:
        raise ValueError("config svrtk_fill_fraction must be between 0 and 1")
    if not 0.0 < float(config["svrtk_fill_opacity"]) < 1.0:
        raise ValueError("config svrtk_fill_opacity must be between 0 and 1")
    if not 0.0 <= float(config["svrtk_fill_endpoint_margin"]) < 0.5:
        raise ValueError("config svrtk_fill_endpoint_margin must be in [0, 0.5)")
    if float(config["svrtk_intensity_power"]) <= 0.0:
        raise ValueError("config svrtk_intensity_power must be positive")
    ssim_weight = min(max(float(config["ssim_weight"]), 0.0), 1.0)
    ultrasound_edges_weight = float(config["ultrasound_edges_weight"])
    l1_weight = float(config["l1_weight"])
    scale_prior_weight = float(config["scale_prior_weight"])
    if ultrasound_edges_weight < 0.0:
        raise ValueError("config ultrasound_edges_weight must be non-negative")
    if l1_weight < 0.0:
        raise ValueError("config l1_weight must be non-negative")
    if scale_prior_weight < 0.0:
        raise ValueError("config scale_prior_weight must be non-negative")
    if int(config["batch_size"]) < 1 or int(config["batch_size"]) > 10:
        raise ValueError("config batch_size must be between 1 and 10")
    if float(config["native_pose_translation_scale"]) <= 0.0:
        raise ValueError("config native_pose_translation_scale must be positive")
    if config["native_pose_convention"] not in {
        "probe_to_world",
        "camera_to_world",
    }:
        raise ValueError(
            "config native_pose_convention must be "
            "'probe_to_world' or 'camera_to_world'"
        )
    if float(config["native_global_scale"]) <= 0.0:
        raise ValueError("config native_global_scale must be positive")
    if float(config["native_init_extent"]) <= 0.0:
        raise ValueError("config native_init_extent must be positive")
    if float(config["native_init_scale"]) <= 0.0:
        raise ValueError("config native_init_scale must be positive")
    if float(config["native_grow_scale3d"]) <= 0.0:
        raise ValueError("config native_grow_scale3d must be positive")
    if float(config["native_prune_scale3d"]) <= 0.0:
        raise ValueError("config native_prune_scale3d must be positive")
    if float(config["native_prune_scale3d_min"]) < 0.0:
        raise ValueError("config native_prune_scale3d_min must be non-negative")
    if float(config["image_value_scale"]) <= 0.0:
        raise ValueError("config image_value_scale must be positive")
    if float(config["ultrasound_far_plane"]) <= 0.0:
        raise ValueError("config ultrasound_far_plane must be positive")
    if (
        config["ultrasound_opening_width"] is not None
        and float(config["ultrasound_opening_width"]) <= 0.0
    ):
        raise ValueError("config ultrasound_opening_width must be positive")
    for noise_key in (
        "pose_sideways_noise",
        "pose_frontback_noise",
        "pose_updown_noise",
    ):
        if float(config[noise_key]) < 0.0:
            raise ValueError(f"config {noise_key} must be non-negative")
    if (
        int(config["densify_every"]) > 0
        and int(config["max_gaussians"]) < int(config["num_gaussians"])
    ):
        config["max_gaussians"] = int(config["num_gaussians"])
    if int(config["densify_stop"]) < int(config["densify_start"]):
        raise ValueError("config densify_stop must be at least densify_start")
    ssim_window_size = int(config["ssim_window_size"])
    if ssim_window_size < 3:
        raise ValueError("config ssim_window_size must be at least 3")

    config["ssim_weight"] = ssim_weight
    config["ultrasound_edges_weight"] = ultrasound_edges_weight
    config["l1_weight"] = l1_weight
    config["scale_prior_weight"] = scale_prior_weight
    config["ssim_window_size"] = ssim_window_size
    return argparse.Namespace(**config)


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
        initial_transmittance=0.99,
        initial_means=None,
        initial_colors=None,
        initial_opacities=None,
        lean_storage=False,
    ):
        super().__init__()
        self.lean_storage = bool(lean_storage)
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
        if self.lean_storage:
            self.register_parameter("raw_cholesky", None)
        else:
            self.raw_cholesky = nn.Parameter(
                self._initial_raw_cholesky(
                    initial_scale.clamp_min(1e-6),
                    num_gaussians,
                )
            )
        if initial_opacities is None:
            initial_opacity = float(initial_opacity)
            if initial_opacity <= 0.0 or initial_opacity >= 1.0:
                raise ValueError("initial_opacity must be between 0 and 1")
            opacity_probabilities = torch.full(
                (num_gaussians, 1),
                initial_opacity,
                dtype=torch.float32,
            )
        else:
            opacity_probabilities = torch.as_tensor(
                initial_opacities,
                dtype=torch.float32,
            ).reshape(-1, 1)
            if len(opacity_probabilities) != num_gaussians:
                raise ValueError("initial_opacities must contain one value per Gaussian")
            if not ((opacity_probabilities > 0.0) & (opacity_probabilities < 1.0)).all():
                raise ValueError("initial_opacities must be between 0 and 1")
        self.logit_opacities = nn.Parameter(torch.logit(opacity_probabilities))
        initial_transmittance = float(initial_transmittance)
        if initial_transmittance <= 0.0 or initial_transmittance >= 1.0:
            raise ValueError("initial_transmittance must be between 0 and 1")
        transmittance_logit = torch.logit(
            torch.as_tensor(initial_transmittance, dtype=torch.float32)
        )
        self.logit_transmittances = nn.Parameter(
            transmittance_logit.repeat(num_gaussians, 1)
        )

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
        if self.lean_storage:
            self.register_parameter("disk_normals", None)
            self.register_parameter("acoustic_attenuation", None)
            self.register_parameter("acoustic_reflection", None)
            self.register_parameter("acoustic_scattering", None)
        else:
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
        max_visible_gaussians_per_slice=None,
        primitive_mode="volume",
        acoustic_rendering=False,
        renderer_backend="torch",
        ultragray_repo_path=None,
        cuda_tile_size_x=4,
        cuda_tile_size_y=128,
    ):
        if renderer_backend == "ultragray_cuda":
            return render_ultrasound_cuda(
                means=self.means,
                scales=torch.exp(self.log_scales),
                colors=self.colors,
                opacities=self.logit_opacities,
                transmittances=self.logit_transmittances,
                slice_to_world=slice_to_world,
                image_height=height,
                image_width=width,
                pixel_spacing=pixel_spacing,
                image_t_probe=(
                    DEFAULT_IMAGE_T_PROBE
                    if image_t_probe is None
                    else image_t_probe
                ),
                image_plane_origin_px=image_plane_origin_px,
                pixel_to_mm=pixel_to_mm,
                image_scale=image_scale,
                shadowing=shadowing,
                shadow_strength=shadow_strength,
                max_visible_gaussians_per_slice=max_visible_gaussians_per_slice,
                ultragray_repo_path=ultragray_repo_path,
                tile_size_x=cuda_tile_size_x,
                tile_size_y=cuda_tile_size_y,
            )[0]
        if renderer_backend != "torch":
            raise ValueError(f"Unknown renderer backend: {renderer_backend}")

        disk_normals = None
        if primitive_mode == "disk":
            if self.disk_normals is None:
                raise RuntimeError("disk primitives are unavailable with lean CUDA storage")
            disk_normals = F.normalize(self.disk_normals, dim=-1, eps=1e-8)
        covariances = None
        if covariance_mode == "full_cholesky":
            covariances = self.full_covariances(min_scale_mm=min_scale_mm, max_scale_mm=max_scale_mm)
        attenuation_weight = 0.0
        reflection_weight = 0.0
        scattering_weight = 0.0
        if acoustic_rendering:
            if self.acoustic_attenuation is None:
                raise RuntimeError("acoustic_rendering is unavailable with lean CUDA storage")
            attenuation_weight = F.softplus(self.acoustic_attenuation)
            reflection_weight = F.softplus(self.acoustic_reflection)
            scattering_weight = F.softplus(self.acoustic_scattering)

        image, _ = render_ultrasound_gaussians(
            self.means,
            torch.exp(self.log_scales),
            self.colors,
            self.logit_opacities,
            slice_to_world,
            height,
            width,
            transmittances=self.logit_transmittances,
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
            max_visible_gaussians_per_slice=max_visible_gaussians_per_slice,
            primitive_mode=primitive_mode,
            disk_normals=disk_normals,
            acoustic_rendering=acoustic_rendering,
            attenuation_weight=attenuation_weight,
            reflection_weight=reflection_weight,
            scattering_weight=scattering_weight,
            covariances=covariances,
        )
        return image

    @staticmethod
    def _inverse_softplus(value):
        value = torch.as_tensor(value, dtype=torch.float32).clamp_min(1e-6)
        return value + torch.log(-torch.expm1(-value))

    @classmethod
    def _initial_raw_cholesky(cls, initial_scale, num_gaussians):
        raw = torch.zeros((num_gaussians, 6), dtype=torch.float32)
        raw_diag = cls._inverse_softplus(initial_scale)
        raw[:, 0] = raw_diag[0]
        raw[:, 2] = raw_diag[1]
        raw[:, 5] = raw_diag[2]
        return raw

    def cholesky_factors(self, min_scale_mm=0.05, max_scale_mm=10.0):
        raw = self.raw_cholesky
        if raw is None:
            raise RuntimeError("full Cholesky covariance is unavailable with lean CUDA storage")
        diag = F.softplus(raw[:, [0, 2, 5]]).clamp(min_scale_mm, max_scale_mm)
        offdiag = raw[:, [1, 3, 4]].clamp(-max_scale_mm, max_scale_mm)
        factors = raw.new_zeros((len(raw), 3, 3))
        factors[:, 0, 0] = diag[:, 0]
        factors[:, 1, 0] = offdiag[:, 0]
        factors[:, 1, 1] = diag[:, 1]
        factors[:, 2, 0] = offdiag[:, 1]
        factors[:, 2, 1] = offdiag[:, 2]
        factors[:, 2, 2] = diag[:, 2]
        return factors

    def full_covariances(self, min_scale_mm=0.05, max_scale_mm=10.0, epsilon=1e-6):
        factors = self.cholesky_factors(min_scale_mm=min_scale_mm, max_scale_mm=max_scale_mm)
        eye = torch.eye(3, device=factors.device, dtype=factors.dtype).unsqueeze(0)
        return factors @ factors.transpose(-1, -2) + epsilon * eye

    def effective_scales(self, covariance_mode="ultrasound_psf", min_scale_mm=0.05, max_scale_mm=10.0):
        if covariance_mode == "full_cholesky":
            covariances = self.full_covariances(min_scale_mm=min_scale_mm, max_scale_mm=max_scale_mm)
            return covariances.diagonal(dim1=-2, dim2=-1).sqrt().clamp(min_scale_mm, max_scale_mm)
        return torch.exp(self.log_scales).clamp(min_scale_mm, max_scale_mm)

    def scale_prior_loss(self, prior_scales, min_scale_mm=0.05, max_scale_mm=10.0, covariance_mode="ultrasound_psf"):
        prior = torch.as_tensor(prior_scales, device=self.log_scales.device, dtype=self.log_scales.dtype)
        if prior.shape != (3,):
            raise ValueError("prior_scales must contain [lateral, axial, elevational] values")
        scales = self.effective_scales(
            covariance_mode=covariance_mode,
            min_scale_mm=min_scale_mm,
            max_scale_mm=max_scale_mm,
        )
        log_prior = prior.clamp_min(min_scale_mm).log()
        return F.mse_loss(scales.log(), log_prior.expand_as(scales.log()))

    def opacity_sparsity_loss(self):
        return torch.sigmoid(self.logit_opacities).mean()

    @torch.no_grad()
    def stabilize_parameters(self, min_scale_mm=0.05, max_scale_mm=10.0):
        self.means.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
        self.means.clamp_(-1e4, 1e4)
        self.log_scales.nan_to_num_(nan=0.0, posinf=max_scale_mm, neginf=min_scale_mm)
        self.log_scales.clamp_(
            torch.log(torch.as_tensor(min_scale_mm, device=self.log_scales.device)),
            torch.log(torch.as_tensor(max_scale_mm, device=self.log_scales.device)),
        )
        if self.raw_cholesky is not None:
            self.raw_cholesky.nan_to_num_(
                nan=0.0,
                posinf=max_scale_mm,
                neginf=-max_scale_mm,
            )
            self.raw_cholesky[:, [1, 3, 4]].clamp_(-max_scale_mm, max_scale_mm)
            raw_min = self._inverse_softplus(
                torch.as_tensor(min_scale_mm, device=self.raw_cholesky.device)
            )
            raw_max = self._inverse_softplus(
                torch.as_tensor(max_scale_mm, device=self.raw_cholesky.device)
            )
            self.raw_cholesky[:, [0, 2, 5]].clamp_(raw_min, raw_max)
        self.logit_opacities.nan_to_num_(nan=0.0, posinf=10.0, neginf=-10.0)
        self.logit_opacities.clamp_(-10.0, 10.0)
        if not hasattr(self, "logit_transmittances"):
            transmittance_logit = torch.logit(
                torch.tensor(0.99, device=self.logit_opacities.device)
            )
            self.logit_transmittances = nn.Parameter(
                torch.full_like(self.logit_opacities, transmittance_logit)
            )
        self.logit_transmittances.nan_to_num_(nan=0.0, posinf=10.0, neginf=-10.0)
        self.logit_transmittances.clamp_(-10.0, 10.0)
        self.colors.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0)
        self.colors.clamp_(0.0, 1.0)
        if self.disk_normals is not None:
            self.disk_normals.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
            self.disk_normals.clamp_(-1.0, 1.0)
        if self.acoustic_attenuation is not None:
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
        covariance_mode="ultrasound_psf",
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
        raw_cholesky = (
            None
            if self.raw_cholesky is None
            else self.raw_cholesky.data[keep]
        )
        logit_opacities = self.logit_opacities.data[keep]
        logit_transmittances = self.logit_transmittances.data[keep]
        colors = self.colors.data[keep]
        disk_normals = (
            None
            if self.disk_normals is None
            else self.disk_normals.data[keep]
        )

        grad = self.means.grad
        if grad is None:
            split_mask = torch.zeros(len(means), device=device, dtype=torch.bool)
        else:
            grad_norm = grad.detach().norm(dim=-1)[keep]
            if covariance_mode == "full_cholesky":
                kept_covariances = self.full_covariances()[keep]
                scales = kept_covariances.diagonal(dim1=-2, dim2=-1).sqrt()
            else:
                scales = torch.exp(log_scales)
            large = scales.max(dim=-1).values >= large_scale_threshold
            split_mask = (grad_norm >= grad_threshold) & large

        remaining_capacity = max(max_gaussians - len(means), 0)
        split_indices = torch.nonzero(split_mask, as_tuple=False).squeeze(-1)
        if len(split_indices) > 0 and remaining_capacity > 0:
            split_indices = split_indices[: min(len(split_indices), remaining_capacity, max_new_gaussians)]
            parent_means = means[split_indices]
            parent_log_scales = log_scales[split_indices]
            parent_raw_cholesky = (
                None
                if raw_cholesky is None
                else raw_cholesky[split_indices]
            )
            if covariance_mode == "full_cholesky":
                parent_scales = scales[split_indices]
            else:
                parent_scales = torch.exp(parent_log_scales)
            offsets = torch.randn_like(parent_means) * parent_scales * 0.25

            child_means = parent_means + offsets
            child_log_scales = parent_log_scales + torch.log(
                torch.as_tensor(split_factor, device=device, dtype=parent_log_scales.dtype)
            )
            child_raw_cholesky = None
            if parent_raw_cholesky is not None:
                child_raw_cholesky = parent_raw_cholesky.clone()
                child_raw_cholesky[:, [0, 2, 5]] = (
                    child_raw_cholesky[:, [0, 2, 5]]
                    + torch.log(
                        torch.as_tensor(
                            split_factor,
                            device=device,
                            dtype=parent_raw_cholesky.dtype,
                        )
                    )
                )
                child_raw_cholesky[:, [1, 3, 4]] = (
                    child_raw_cholesky[:, [1, 3, 4]] * split_factor
                )
            child_logit_opacities = logit_opacities[split_indices] - torch.log(
                torch.as_tensor(2.0, device=device, dtype=logit_opacities.dtype)
            )
            child_logit_transmittances = logit_transmittances[split_indices]
            child_colors = colors[split_indices]
            child_disk_normals = (
                None
                if disk_normals is None
                else disk_normals[split_indices]
            )

            means[split_indices] = parent_means - offsets
            log_scales[split_indices] = child_log_scales
            if raw_cholesky is not None:
                raw_cholesky[split_indices] = child_raw_cholesky
            logit_opacities[split_indices] = child_logit_opacities
            logit_transmittances[split_indices] = child_logit_transmittances

            means = torch.cat([means, child_means], dim=0)
            log_scales = torch.cat([log_scales, child_log_scales], dim=0)
            if raw_cholesky is not None:
                raw_cholesky = torch.cat(
                    [raw_cholesky, child_raw_cholesky],
                    dim=0,
                )
            logit_opacities = torch.cat([logit_opacities, child_logit_opacities], dim=0)
            logit_transmittances = torch.cat(
                [logit_transmittances, child_logit_transmittances], dim=0
            )
            colors = torch.cat([colors, child_colors], dim=0)
            if disk_normals is not None:
                disk_normals = torch.cat(
                    [disk_normals, child_disk_normals],
                    dim=0,
                )

        self.means = nn.Parameter(means.contiguous())
        self.log_scales = nn.Parameter(log_scales.contiguous())
        if raw_cholesky is not None:
            self.raw_cholesky = nn.Parameter(raw_cholesky.contiguous())
        self.logit_opacities = nn.Parameter(logit_opacities.contiguous())
        self.logit_transmittances = nn.Parameter(logit_transmittances.contiguous())
        self.colors = nn.Parameter(colors.contiguous())
        if disk_normals is not None:
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


class SourcePoseCorrection(nn.Module):
    def __init__(
        self,
        source_count,
        max_translation_mm=2.0,
        max_rotation_deg=2.0,
    ):
        super().__init__()
        self.max_translation_mm = float(max_translation_mm)
        self.max_rotation_rad = float(max_rotation_deg) * np.pi / 180.0
        self.raw_translation = nn.Parameter(torch.zeros(source_count, 3))
        self.raw_rotation = nn.Parameter(torch.zeros(source_count, 3))

    def correction_parameters(self, source_indices):
        source_indices = source_indices.long().reshape(-1)
        translation = self.max_translation_mm * torch.tanh(self.raw_translation[source_indices])
        rotation = self.max_rotation_rad * torch.tanh(self.raw_rotation[source_indices])
        return translation, rotation

    def forward(self, poses, source_indices):
        squeeze_output = poses.ndim == 2
        if squeeze_output:
            poses = poses.unsqueeze(0)
        source_indices = source_indices.to(device=poses.device)
        translation, rotation = self.correction_parameters(source_indices)
        delta = se3_delta_to_matrix(rotation, translation, dtype=poses.dtype)
        corrected = delta @ poses
        return corrected.squeeze(0) if squeeze_output else corrected

    def regularization_loss(self):
        translation_fraction = torch.tanh(self.raw_translation).square().mean()
        rotation_fraction = torch.tanh(self.raw_rotation).square().mean()
        return translation_fraction + rotation_fraction

    @torch.no_grad()
    def metadata(self, source_names=None):
        translation = self.max_translation_mm * torch.tanh(self.raw_translation.detach())
        rotation = self.max_rotation_rad * torch.tanh(self.raw_rotation.detach())
        rotation_deg = rotation * (180.0 / np.pi)
        records = []
        for index in range(len(translation)):
            records.append(
                {
                    "source_index": int(index),
                    "source_name": None if source_names is None else source_names[index],
                    "translation_mm_xyz": [float(value) for value in translation[index].cpu()],
                    "rotation_deg_xyz": [float(value) for value in rotation_deg[index].cpu()],
                }
            )
        return records


def skew_symmetric(vectors):
    zero = torch.zeros_like(vectors[..., 0])
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def se3_delta_to_matrix(rotation_vectors, translations, dtype):
    rotation_vectors = rotation_vectors.to(dtype=dtype)
    translations = translations.to(dtype=dtype)
    angles = torch.linalg.norm(rotation_vectors, dim=-1, keepdim=True)
    axes = rotation_vectors / angles.clamp_min(1e-8)
    skew = skew_symmetric(axes)
    eye = torch.eye(3, device=rotation_vectors.device, dtype=dtype).expand(
        rotation_vectors.shape[0], 3, 3
    )
    sin = torch.sin(angles)[..., None]
    cos = torch.cos(angles)[..., None]
    rotation = eye + sin * skew + (1.0 - cos) * (skew @ skew)
    small = angles.squeeze(-1) < 1e-8
    if small.any():
        rotation = torch.where(small[:, None, None], eye, rotation)

    delta = torch.eye(4, device=rotation_vectors.device, dtype=dtype).expand(
        rotation_vectors.shape[0], 4, 4
    ).clone()
    delta[:, :3, :3] = rotation
    delta[:, :3, 3] = translations
    return delta


def normalize_image_for_save(image, normalize=True):
    image = image.detach().float().cpu()
    if image.ndim == 4:
        image = image.squeeze(0)
    if image.ndim == 2:
        image = image.unsqueeze(0)

    if normalize:
        image = image - image.amin(dim=(-2, -1), keepdim=True)
        image = image / image.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    else:
        image = image.clamp(0.0, 1.0)

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
        pred_stats = (
            float(pred.min().item()),
            float(pred.max().item()),
            float(pred.mean().item()),
        )
        target_stats = (
            float(target.min().item()),
            float(target.max().item()),
            float(target.mean().item()),
        )
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
            (
                "target "
                f"{target_stats[0]:.3g}/{target_stats[1]:.3g}/{target_stats[2]:.3g}",
                target,
                False,
            ),
            (
                "rendered "
                f"{pred_stats[0]:.3g}/{pred_stats[1]:.3g}/{pred_stats[2]:.3g}",
                pred,
                False,
            ),
            ("raw diff", raw_diff, True),
            ("target edges", target_edges, True),
            ("rendered edges", pred_edges, True),
            ("edge diff", edge_diff, True),
        ]
        if confidence is not None:
            panels.append(("confidence", confidence, True))

        images = [
            add_label(
                normalize_image_for_save(tensor, normalize=normalize),
                label,
            )
            for label, tensor, normalize in panels
        ]
        width = sum(image.width for image in images)
        height = max(image.height for image in images)
        montage = Image.new("RGB", (width, height), color=(0, 0, 0))

        x_offset = 0
        for image in images:
            montage.paste(image, (x_offset, 0))
            x_offset += image.width

        print(
            f"debug step={step} "
            f"target[min/max/mean]={target_stats[0]:.6g}/"
            f"{target_stats[1]:.6g}/{target_stats[2]:.6g} "
            f"rendered[min/max/mean]={pred_stats[0]:.6g}/"
            f"{pred_stats[1]:.6g}/{pred_stats[2]:.6g}"
        )

        montage.save(output_dir / f"step_{step:06d}_debug.png")


def args_to_metadata(args):
    metadata = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            metadata[key] = str(value)
        else:
            metadata[key] = value
    return metadata


def colors_to_rgb_uint8(colors):
    colors = colors.detach().float().cpu().clamp(0.0, 1.0).numpy()
    if colors.shape[1] == 1:
        colors = np.repeat(colors, 3, axis=1)
    return np.clip(colors[:, :3] * 255.0, 0.0, 255.0).astype(np.uint8)


def write_gaussians_ply(model, output_path, opacity_threshold=0.0):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        means = model.means.detach().float().cpu().numpy()
        scales = model.effective_scales().detach().float().cpu().numpy()
        colors = colors_to_rgb_uint8(model.colors)
        opacities = torch.sigmoid(model.logit_opacities).detach().float().cpu().numpy().reshape(-1)
        if getattr(model, "disk_normals", None) is not None:
            normals = F.normalize(model.disk_normals.detach().float(), dim=-1, eps=1e-8).cpu().numpy()
        else:
            normals = np.zeros_like(means)
            normals[:, 2] = 1.0

    keep = opacities >= opacity_threshold
    means = means[keep]
    scales = scales[keep]
    colors = colors[keep]
    opacities = opacities[keep]
    normals = normals[keep]

    header = [
        "ply",
        "format ascii 1.0",
        "comment exported from ultrasound Gaussian splatting training",
        f"element vertex {len(means)}",
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

    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(header) + "\n")
        for i in range(len(means)):
            file.write(
                f"{means[i, 0]:.8f} {means[i, 1]:.8f} {means[i, 2]:.8f} "
                f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])} "
                f"{opacities[i]:.8f} "
                f"{scales[i, 0]:.8f} {scales[i, 1]:.8f} {scales[i, 2]:.8f} "
                f"{normals[i, 0]:.8f} {normals[i, 1]:.8f} {normals[i, 2]:.8f}\n"
            )


def save_checkpoint(
    model,
    output_path,
    args,
    step,
    final_loss=None,
    calibration_metadata=None,
    validation_history=None,
    pose_corrector=None,
    source_names=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "pose_correction_state_dict": (
            None if pose_corrector is None else pose_corrector.state_dict()
        ),
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
        "pose_correction": {
            "mode": args.pose_correction,
            "max_translation_mm": args.pose_correction_max_translation_mm,
            "max_rotation_deg": args.pose_correction_max_rotation_deg,
            "weight": args.pose_correction_weight,
            "lr": args.pose_correction_lr,
            "sources": [] if pose_corrector is None else pose_corrector.metadata(source_names),
        },
        "covariance": {
            "primitive_mode": args.primitive_mode,
            "mode": args.covariance_mode,
            "min_scale_mm": args.min_scale_mm,
            "max_scale_mm": args.max_scale_mm,
            "initial_opacity": args.initial_opacity,
            "initial_transmittance": args.initial_transmittance,
            "scale_prior_lateral": args.scale_prior_lateral,
            "scale_prior_axial": args.scale_prior_axial,
            "scale_prior_elevational": args.scale_prior_elevational,
            "scale_prior_weight": args.scale_prior_weight,
            "lateral_depth_slope": args.lateral_depth_slope,
            "elevational_depth_slope": args.elevational_depth_slope,
            "acoustic_rendering": args.acoustic_rendering,
        },
        "loss": {
            "type": "ultrasound_edges_ssim",
            "accumulation_steps": int(max(args.accumulation_steps, 1)),
            "validation_slices": int(max(args.validation_slices, 0)),
            "validation_fraction": float(args.validation_fraction),
            "validation_sources": list(args.validation_sources or []),
            "validation_every": int(max(args.validation_every, 0)),
            "validation_seed": int(args.validation_seed),
            "amp": bool(args.amp),
            "config_path": args.loss_config_path,
            "laplacian_weight": args.laplacian_loss_weight,
            "edge_weight": args.edge_loss_weight,
            "intensity_weight": args.intensity_loss_weight,
            "sobel_weight": args.sobel_loss_weight,
            "ultrasound_edges_weight": args.ultrasound_edges_weight,
            "l1_weight": args.l1_weight,
            "ssim_weight": args.ssim_weight,
            "scale_prior_weight": args.scale_prior_weight,
            "ssim_window_size": args.ssim_window_size,
            "opacity_sparsity_weight": args.opacity_sparsity_weight,
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
    ply_path = output_path.with_suffix(".ply")
    write_gaussians_ply(model, ply_path)

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state_dict", "pose_correction_state_dict"}
    }
    metadata["ply_path"] = str(ply_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path, ply_path


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
    fill_fraction=0.2,
    supported_opacity=0.5,
    fill_opacity=0.05,
    fill_endpoint_margin=0.1,
    device="cpu",
):
    """
    Build an SVRTK-style initialization from tracked 2D slices.

    The function maps slice pixels into 3D with x = T_k y, averages the
    scattered intensities into a voxel grid, then seeds Gaussian centers from
    high-intensity voxels in that initial volume X0. A configurable fraction
    is sampled directly between consecutive tracked slices.
    """
    coords_per_slice = []
    values_per_slice = []
    sequence_ids = []
    pose_centers = []
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
        sequence_index = sample.get("sequence_index", 0)
        if torch.is_tensor(sequence_index):
            sequence_index = int(sequence_index.reshape(-1)[0].item())
        sequence_ids.append(int(sequence_index))
        pose_centers.append(pose[:3, 3])
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

    fill_count = min(
        max(round(num_gaussians * float(fill_fraction)), 0),
        num_gaussians,
    )
    supported_count = num_gaussians - fill_count

    supported_means, supported_colors = sample_gaussians_from_volume(
        initial_volume,
        counts,
        bounds_min,
        bounds_max,
        supported_count,
        intensity_threshold=intensity_threshold,
        jitter_voxels=jitter_voxels,
    )
    fill_means, fill_colors = sample_between_tracked_slices(
        coords_per_slice,
        values_per_slice,
        sequence_ids,
        pose_centers,
        fill_count,
        endpoint_margin=fill_endpoint_margin,
    )
    means = torch.cat([supported_means, fill_means], dim=0)
    colors = torch.cat([supported_colors, fill_colors], dim=0)
    opacities = torch.cat(
        [
            torch.full(
                (supported_count, 1),
                float(supported_opacity),
                device=means.device,
            ),
            torch.full(
                (fill_count, 1),
                float(fill_opacity),
                device=means.device,
            ),
        ],
        dim=0,
    )

    return {
        "means": means.cpu(),
        "colors": colors.cpu(),
        "opacities": opacities.cpu(),
        "volume": initial_volume.cpu(),
        "counts": counts.cpu(),
        "bounds_min": bounds_min.cpu(),
        "bounds_max": bounds_max.cpu(),
        "supported_count": supported_count,
        "fill_count": fill_count,
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


def sample_between_tracked_slices(
    coords_per_slice,
    values_per_slice,
    sequence_ids,
    pose_centers,
    num_samples,
    endpoint_margin=0.1,
):
    """Sample corresponding image locations between neighboring tracked planes."""
    if num_samples <= 0:
        channels = values_per_slice[0].shape[-1]
        return (
            coords_per_slice[0].new_empty((0, 3)),
            values_per_slice[0].new_empty((0, channels)),
        )

    pair_indices = []
    pair_weights = []
    for index in range(len(coords_per_slice) - 1):
        if sequence_ids[index] != sequence_ids[index + 1]:
            continue
        if len(coords_per_slice[index]) != len(coords_per_slice[index + 1]):
            continue
        spacing = torch.linalg.norm(pose_centers[index + 1] - pose_centers[index])
        if not torch.isfinite(spacing) or spacing <= 1e-8:
            continue
        pair_indices.append(index)
        pair_weights.append(spacing)

    if not pair_indices:
        raise ValueError(
            "SVRTK between-slice fill requires consecutive poses from the same sequence"
        )

    weights = torch.stack(pair_weights).to(
        device=coords_per_slice[0].device,
        dtype=coords_per_slice[0].dtype,
    )
    chosen_pairs = torch.multinomial(
        weights.clamp_min(1e-8),
        num_samples,
        replacement=True,
    )

    fill_means = coords_per_slice[0].new_empty((num_samples, 3))
    channels = values_per_slice[0].shape[-1]
    fill_colors = values_per_slice[0].new_empty((num_samples, channels))
    endpoint_margin = min(max(float(endpoint_margin), 0.0), 0.499)

    for pair_choice in torch.unique(chosen_pairs):
        output_indices = torch.nonzero(
            chosen_pairs == pair_choice,
            as_tuple=False,
        ).squeeze(-1)
        pair_index = pair_indices[int(pair_choice.item())]
        first_coords = coords_per_slice[pair_index]
        second_coords = coords_per_slice[pair_index + 1]
        first_values = values_per_slice[pair_index]
        second_values = values_per_slice[pair_index + 1]

        sample_count = len(output_indices)
        pixel_indices = torch.randint(
            0,
            len(first_coords),
            (sample_count,),
            device=first_coords.device,
        )
        t = torch.rand(
            sample_count,
            1,
            device=first_coords.device,
            dtype=first_coords.dtype,
        )
        t = endpoint_margin + t * (1.0 - 2.0 * endpoint_margin)
        fill_means[output_indices] = torch.lerp(
            first_coords[pixel_indices],
            second_coords[pixel_indices],
            t,
        )
        fill_colors[output_indices] = torch.lerp(
            first_values[pixel_indices],
            second_values[pixel_indices],
            t,
        )

    return fill_means, fill_colors.clamp(0.0, 1.0)


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
    if num_gaussians == 0:
        chosen = torch.empty(0, device=volume.device, dtype=torch.long)
    elif len(candidate_indices) >= num_gaussians:
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


def resize_adjusted_calibration(dataset, height, width, calibration_kwargs, args):
    if hasattr(dataset, "datasets"):
        original_sizes = {
            tuple(source.original_image_size)
            for source in dataset.datasets
        }
    else:
        original_sizes = {tuple(dataset.original_image_size)}
    if len(original_sizes) != 1:
        raise ValueError(
            "All image sources must have the same original dimensions when "
            "using shared ultrasound calibration"
        )

    original_height, original_width = next(iter(original_sizes))
    resize_x = float(width) / float(original_width)
    resize_y = float(height) / float(original_height)

    image_origin = calibration_kwargs["image_plane_origin_px"]
    image_origin_x = (
        float(image_origin[0]) * resize_x
        if args.image_origin_x is None
        else float(args.image_origin_x)
    )
    image_origin_y = (
        float(image_origin[1]) * resize_y
        if args.image_origin_y is None
        else float(args.image_origin_y)
    )
    base_pixel_to_mm = (
        float(calibration_kwargs["pixel_to_mm"])
        if args.pixel_to_mm is None
        else float(args.pixel_to_mm)
    )
    base_image_scale = (
        float(calibration_kwargs["image_scale"])
        if args.image_scale is None
        else float(args.image_scale)
    )
    original_spacing = base_pixel_to_mm * base_image_scale
    spacing_x = original_spacing / resize_x
    spacing_y = original_spacing / resize_y

    # A non-positive pixel_to_mm makes both renderers use the independent
    # lateral and axial pixel spacings supplied below.
    return image_origin_x, image_origin_y, 0.0, 1.0, spacing_x, spacing_y


def random_means_from_tracked_bounds(
    dataset,
    num_gaussians,
    height,
    width,
    image_t_probe,
    image_plane_origin_px,
    pixel_to_mm,
    image_scale,
    pixel_spacing,
    padding_fraction=0.02,
    device="cpu",
):
    dtype = torch.float32
    calibrated_spacing = float(pixel_to_mm) * float(image_scale)
    if calibrated_spacing > 0.0:
        spacing_x = calibrated_spacing
        spacing_y = calibrated_spacing
    else:
        spacing_x = float(pixel_spacing[0])
        spacing_y = float(pixel_spacing[1])
    if spacing_x <= 0.0 or spacing_y <= 0.0:
        raise ValueError("Random tracked-volume initialization requires positive pixel spacing")

    origin = torch.as_tensor(
        image_plane_origin_px,
        device=device,
        dtype=dtype,
    )
    pixel_corners = torch.tensor(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        device=device,
        dtype=dtype,
    )
    image_corners = torch.zeros((4, 4), device=device, dtype=dtype)
    image_corners[:, 0] = (pixel_corners[:, 0] - origin[0]) * spacing_x
    image_corners[:, 1] = (pixel_corners[:, 1] - origin[1]) * spacing_y
    image_corners[:, 3] = 1.0
    probe_t_image = torch.linalg.inv(
        torch.as_tensor(image_t_probe, device=device, dtype=dtype)
    )
    probe_corners = (probe_t_image @ image_corners.T)

    bounds_min = None
    bounds_max = None
    for sample in dataset:
        pose = sample["pose"].to(device=device, dtype=dtype)
        world_corners = (pose @ probe_corners).T[:, :3]
        current_min = world_corners.min(dim=0).values
        current_max = world_corners.max(dim=0).values
        bounds_min = (
            current_min
            if bounds_min is None
            else torch.minimum(bounds_min, current_min)
        )
        bounds_max = (
            current_max
            if bounds_max is None
            else torch.maximum(bounds_max, current_max)
        )

    extent = (bounds_max - bounds_min).clamp_min(1e-3)
    padding = extent * max(float(padding_fraction), 0.0)
    bounds_min = bounds_min - padding
    bounds_max = bounds_max + padding
    means = bounds_min + torch.rand(
        (int(num_gaussians), 3),
        device=device,
        dtype=dtype,
    ) * (bounds_max - bounds_min)
    return means, bounds_min, bounds_max


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
        max_visible_gaussians_per_slice=args.max_visible_gaussians_per_slice,
        primitive_mode=args.primitive_mode,
        acoustic_rendering=args.acoustic_rendering,
        renderer_backend=args.renderer_backend,
        ultragray_repo_path=args.ultragray_repo_path,
        cuda_tile_size_x=args.cuda_tile_size_x,
        cuda_tile_size_y=args.cuda_tile_size_y,
    )


def batch_source_index(batch, device):
    if "sequence_index" not in batch:
        return torch.zeros(1, device=device, dtype=torch.long)
    source_index = batch["sequence_index"]
    if not torch.is_tensor(source_index):
        source_index = torch.as_tensor(source_index)
    return source_index.to(device=device, dtype=torch.long).reshape(-1)


def maybe_correct_pose(pose, source_index, pose_corrector):
    if pose_corrector is None:
        return pose
    return pose_corrector(pose, source_index)


def _as_bchw(image):
    if image.ndim == 2:
        return image.unsqueeze(0).unsqueeze(0)
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError(f"Expected image tensor with 2, 3, or 4 dims, got {image.ndim}")


def ssim_loss(pred, target, window_size=11, data_range=1.0):
    pred = _as_bchw(pred).clamp(0.0, data_range)
    target = _as_bchw(target).clamp(0.0, data_range)
    if pred.shape[1] != target.shape[1]:
        if pred.shape[1] == 1:
            pred = pred.expand(-1, target.shape[1], -1, -1)
        elif target.shape[1] == 1:
            target = target.expand(-1, pred.shape[1], -1, -1)
    if pred.shape != target.shape:
        raise ValueError(f"SSIM expects matching pred/target shapes, got {pred.shape} and {target.shape}")

    window_size = int(window_size)
    if window_size < 3:
        raise ValueError("config ssim_window_size must be at least 3")
    if window_size % 2 == 0:
        window_size += 1

    padding = window_size // 2
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_pred_target = mu_pred * mu_target

    sigma_pred = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_pred_sq
    sigma_target = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_pred_target

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_pred_target + c1) * (2.0 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2)
    ssim = numerator / denominator.clamp_min(1e-8)
    return 1.0 - ssim.clamp(-1.0, 1.0).mean()


def prediction_loss(pred, target, args, model=None):
    pred_pixels = _as_bchw(pred)
    target_pixels = _as_bchw(target)
    if pred_pixels.shape[1] != target_pixels.shape[1]:
        if pred_pixels.shape[1] == 1:
            pred_pixels = pred_pixels.expand(
                -1, target_pixels.shape[1], -1, -1
            )
        elif target_pixels.shape[1] == 1:
            target_pixels = target_pixels.expand(
                -1, pred_pixels.shape[1], -1, -1
            )
    if pred_pixels.shape != target_pixels.shape:
        raise ValueError(
            "L1 expects matching pred/target shapes, got "
            f"{pred_pixels.shape} and {target_pixels.shape}"
        )
    pixel_loss = F.l1_loss(pred_pixels, target_pixels)
    edge_loss = ultrasound_edge_loss(
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
    structure_loss = ssim_loss(
        pred,
        target,
        window_size=args.ssim_window_size,
        data_range=1.0,
    )
    loss = (
        args.ultrasound_edges_weight * edge_loss
        + args.ssim_weight * structure_loss
        + args.l1_weight * pixel_loss
    )

    if model is not None and args.scale_prior_weight > 0.0:
        loss = loss + args.scale_prior_weight * model.scale_prior_loss(
            prior_scales=(
                args.scale_prior_lateral,
                args.scale_prior_axial,
                args.scale_prior_elevational,
            ),
            min_scale_mm=args.min_scale_mm,
            max_scale_mm=args.max_scale_mm,
            covariance_mode=args.covariance_mode,
        )
    if model is not None and args.opacity_sparsity_weight > 0.0:
        loss = loss + args.opacity_sparsity_weight * model.opacity_sparsity_loss()
    return loss


def gaussian_learning_diagnostics(model, initial_means_sample, sample_indices):
    diagnostics = {}
    parameter_names = (
        "means",
        "log_scales",
        "colors",
        "logit_opacities",
        "logit_transmittances",
    )
    for name in parameter_names:
        parameter = getattr(model, name)
        grad = parameter.grad
        if grad is None:
            diagnostics[f"{name}_grad_mean"] = 0.0
            diagnostics[f"{name}_grad_max"] = 0.0
            diagnostics[f"{name}_active_fraction"] = 0.0
            continue
        detached = grad.detach()
        row_magnitude = detached.abs().reshape(len(detached), -1).amax(dim=-1)
        diagnostics[f"{name}_grad_mean"] = float(detached.abs().mean().item())
        diagnostics[f"{name}_grad_max"] = float(detached.abs().max().item())
        diagnostics[f"{name}_active_fraction"] = float(
            (row_magnitude > 1e-12).float().mean().item()
        )

    current_sample = model.means.detach()[sample_indices]
    displacement = torch.linalg.norm(
        current_sample - initial_means_sample,
        dim=-1,
    )
    diagnostics["mean_displacement"] = float(displacement.mean().item())
    diagnostics["max_displacement"] = float(displacement.max().item())
    return diagnostics


def format_learning_diagnostics(diagnostics):
    return (
        "learn "
        f"visible_grad={100.0 * diagnostics['means_active_fraction']:.2f}% "
        f"mean_grad={diagnostics['means_grad_mean']:.3g} "
        f"max_grad={diagnostics['means_grad_max']:.3g} "
        f"move={diagnostics['mean_displacement']:.4g}/"
        f"{diagnostics['max_displacement']:.4g} "
        f"scale_grad={diagnostics['log_scales_grad_mean']:.3g} "
        f"intensity_grad={diagnostics['colors_grad_mean']:.3g} "
        f"opacity_grad={diagnostics['logit_opacities_grad_mean']:.3g} "
        f"trans_grad={diagnostics['logit_transmittances_grad_mean']:.3g}"
    )


@torch.no_grad()
def evaluate_validation_loss(
    model,
    pose_corrector,
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
    correction_was_training = False if pose_corrector is None else pose_corrector.training
    model.eval()
    if pose_corrector is not None:
        pose_corrector.eval()
    losses = []
    for batch in validation_loader:
        target = batch["image"].squeeze(0).to(device)
        pose = batch["pose"].squeeze(0).to(device)
        source_index = batch_source_index(batch, device)
        pose = maybe_correct_pose(pose, source_index, pose_corrector)
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
    if pose_corrector is not None and correction_was_training:
        pose_corrector.train()
    return sum(losses) / max(len(losses), 1)


def train(args):
    if args.renderer_backend == "ultragray_cuda":
        from ultragray_native_training import train_native_ultragray

        return train_native_ultragray(args)

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
            image_value_scale=args.image_value_scale,
        )
        source_names = [dataset.source_name]
    else:
        dataset = MultiTrackedUltrasoundDataset(
            image_dirs=image_dirs,
            poses_paths=poses_paths,
            image_size=(args.height, args.width),
            grayscale=args.grayscale,
            low_intensity_threshold=args.low_intensity_threshold,
            image_value_scale=args.image_value_scale,
        )
        source_names = [source_dataset.source_name for source_dataset in dataset.datasets]
    explicit_validation_dataset = None
    if args.image_dir_val:
        if len(args.image_dir_val) == 1:
            explicit_validation_dataset = TrackedUltrasoundDataset(
                image_dir=args.image_dir_val[0],
                poses_path=args.pose_val[0] if args.pose_val else None,
                image_size=(args.height, args.width),
                grayscale=args.grayscale,
                low_intensity_threshold=args.low_intensity_threshold,
                image_value_scale=args.image_value_scale,
            )
        else:
            explicit_validation_dataset = MultiTrackedUltrasoundDataset(
                image_dirs=args.image_dir_val,
                poses_paths=args.pose_val,
                image_size=(args.height, args.width),
                grayscale=args.grayscale,
                low_intensity_threshold=args.low_intensity_threshold,
                image_value_scale=args.image_value_scale,
            )

    source_validation = (
        None
        if explicit_validation_dataset is not None
        else choose_source_validation_indices(dataset, args.validation_sources)
    )
    validation_source_names = []
    if explicit_validation_dataset is not None:
        validation_indices = []
        training_indices = list(range(len(dataset)))
    elif source_validation is None:
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
        DataLoader(explicit_validation_dataset, batch_size=1, shuffle=False)
        if explicit_validation_dataset is not None
        else (
            DataLoader(Subset(dataset, validation_indices), batch_size=1, shuffle=False)
            if validation_indices
            else None
        )
    )
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    if explicit_validation_dataset is not None:
        print(
            f"using {len(dataset)} explicit training slices and "
            f"{len(explicit_validation_dataset)} explicit validation slices"
        )
    elif validation_indices:
        print(
            f"using {len(training_indices)} training slices and "
            f"{len(validation_indices)} fixed validation slices"
        )
        if validation_source_names:
            print("validation sources: " + "; ".join(validation_source_names))

    calibration_kwargs = dataset.calibration.to_renderer_kwargs()
    (
        image_origin_x,
        image_origin_y,
        pixel_to_mm,
        image_scale,
        pixel_spacing_x,
        pixel_spacing_y,
    ) = resize_adjusted_calibration(
        dataset,
        args.height,
        args.width,
        calibration_kwargs,
        args,
    )
    args.pixel_spacing_x = pixel_spacing_x
    args.pixel_spacing_y = pixel_spacing_y
    print(
        "resize-adjusted calibration: "
        f"origin=({image_origin_x:.3f}, {image_origin_y:.3f}) px, "
        f"spacing=({pixel_spacing_x:.6f}, {pixel_spacing_y:.6f}) mm/px"
    )
    checkpoint_calibration_metadata = {
        "image_t_probe": calibration_kwargs["image_t_probe"].tolist(),
        "image_plane_origin_px": [image_origin_x, image_origin_y],
        "pixel_to_mm": pixel_to_mm,
        "image_scale": image_scale,
        "pixel_spacing_x": pixel_spacing_x,
        "pixel_spacing_y": pixel_spacing_y,
    }

    channels = 1 if args.grayscale else 3
    initial_means = None
    initial_colors = None
    initial_opacities = None
    if args.init == "random":
        initial_means, bounds_min, bounds_max = random_means_from_tracked_bounds(
            train_dataset,
            args.num_gaussians,
            args.height,
            args.width,
            image_t_probe=calibration_kwargs["image_t_probe"],
            image_plane_origin_px=(image_origin_x, image_origin_y),
            pixel_to_mm=pixel_to_mm,
            image_scale=image_scale,
            pixel_spacing=(args.pixel_spacing_x, args.pixel_spacing_y),
            device=device,
        )
        print(
            f"initialized {len(initial_means)} random Gaussians inside tracked bounds: "
            f"min={bounds_min.detach().cpu().tolist()}, "
            f"max={bounds_max.detach().cpu().tolist()}"
        )
    elif args.init == "svrtk":
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
            fill_fraction=args.svrtk_fill_fraction,
            supported_opacity=args.initial_opacity,
            fill_opacity=args.svrtk_fill_opacity,
            fill_endpoint_margin=args.svrtk_fill_endpoint_margin,
            device=device,
        )
        initial_means = svrtk_init["means"]
        initial_colors = svrtk_init["colors"]
        initial_opacities = svrtk_init["opacities"]
        print(
            f"initialized {len(initial_means)} Gaussians from SVRTK volume: "
            f"{svrtk_init['supported_count']} image-supported, "
            f"{svrtk_init['fill_count']} between-slice fill"
        )

    model = VanillaGaussianSplatting(
        args.num_gaussians,
        channels=channels,
        initial_scale=(args.initial_scale_x, args.initial_scale_y, args.initial_scale_z),
        initial_opacity=args.initial_opacity,
        initial_transmittance=args.initial_transmittance,
        initial_means=initial_means,
        initial_colors=initial_colors,
        initial_opacities=initial_opacities,
        lean_storage=args.renderer_backend == "ultragray_cuda",
    ).to(device)
    diagnostic_sample_count = min(model.num_gaussians, 4096)
    diagnostic_indices = torch.linspace(
        0,
        model.num_gaussians - 1,
        diagnostic_sample_count,
        device=device,
    ).long()
    initial_means_sample = model.means.detach()[diagnostic_indices].clone()
    pose_corrector = None
    if args.pose_correction == "source":
        pose_corrector = SourcePoseCorrection(
            len(source_names),
            max_translation_mm=args.pose_correction_max_translation_mm,
            max_rotation_deg=args.pose_correction_max_rotation_deg,
        ).to(device)
        print(
            "enabled source pose correction: "
            f"{len(source_names)} sources, "
            f"translation <= {args.pose_correction_max_translation_mm} mm, "
            f"rotation <= {args.pose_correction_max_rotation_deg} deg"
        )

    optimizer_parameters = [{"params": model.parameters(), "lr": args.lr}]
    if pose_corrector is not None:
        optimizer_parameters.append(
            {"params": pose_corrector.parameters(), "lr": args.pose_correction_lr}
        )
    optimizer = torch.optim.Adam(optimizer_parameters)
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
            source_index = batch_source_index(batch, device)
            pose = maybe_correct_pose(pose, source_index, pose_corrector)
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
                if pose_corrector is not None and args.pose_correction_weight > 0.0:
                    loss = loss + (
                        args.pose_correction_weight
                        * pose_corrector.regularization_loss()
                    )

            accumulated_loss += float(loss.detach().item())
            accumulated_count += 1
            if accumulation_index == 0:
                debug_pred = pred.detach()
                debug_target = target.detach()
            scaler.scale(loss).backward()

        if accumulated_count == 0:
            raise RuntimeError("No training slices were available from the data loader.")

        scaler.unscale_(optimizer)
        for parameter_group in optimizer.param_groups:
            for parameter in parameter_group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_count)

        should_log = step == 1 or step % args.log_every == 0
        learning_diagnostics = None
        if should_log:
            learning_diagnostics = gaussian_learning_diagnostics(
                model,
                initial_means_sample,
                diagnostic_indices,
            )

        loss = torch.as_tensor(accumulated_loss / accumulated_count, device=device)
        scaler.step(optimizer)
        scaler.update()
        model.stabilize_parameters(
            min_scale_mm=args.min_scale_mm,
            max_scale_mm=args.max_scale_mm,
        )
        if learning_diagnostics is not None:
            updated_sample = model.means.detach()[diagnostic_indices]
            displacement = torch.linalg.norm(
                updated_sample - initial_means_sample,
                dim=-1,
            )
            learning_diagnostics["mean_displacement"] = float(
                displacement.mean().item()
            )
            learning_diagnostics["max_displacement"] = float(
                displacement.max().item()
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
                covariance_mode=args.covariance_mode,
            )
            optimizer_parameters = [{"params": model.parameters(), "lr": args.lr}]
            if pose_corrector is not None:
                optimizer_parameters.append(
                    {"params": pose_corrector.parameters(), "lr": args.pose_correction_lr}
                )
            optimizer = torch.optim.Adam(optimizer_parameters)
            diagnostic_sample_count = min(model.num_gaussians, 4096)
            diagnostic_indices = torch.linspace(
                0,
                model.num_gaussians - 1,
                diagnostic_sample_count,
                device=device,
            ).long()
            initial_means_sample = model.means.detach()[diagnostic_indices].clone()
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
                pose_corrector,
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
        elif should_log:
            print(f"step={step:04d} loss={loss.item():.6f} gaussians={model.num_gaussians}")
        if learning_diagnostics is not None:
            print(format_learning_diagnostics(learning_diagnostics))

        if args.output and args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            step_output = checkpoint_path_for_step(args.output, step)
            output_path, metadata_path, ply_path = save_checkpoint(
                model,
                step_output,
                args,
                step=step,
                final_loss=loss.item(),
                validation_history=validation_history,
                calibration_metadata=checkpoint_calibration_metadata,
                pose_corrector=pose_corrector,
                source_names=source_names,
            )
            print(f"saved step checkpoint {output_path}")
            print(f"saved step metadata {metadata_path}")
            print(f"saved step ply {ply_path}")

    if args.output:
        output_path, metadata_path, ply_path = save_checkpoint(
            model,
            args.output,
            args,
            step=args.steps,
            final_loss=loss.item() if "loss" in locals() else None,
            validation_history=validation_history,
            calibration_metadata=checkpoint_calibration_metadata,
            pose_corrector=pose_corrector,
            source_names=source_names,
        )
        print(f"saved checkpoint {output_path}")
        print(f"saved metadata {metadata_path}")
        print(f"saved ply {ply_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a vanilla PyTorch Gaussian splatting model on images.")
    parser.add_argument("config", nargs="?", help="Path to a JSON training config.")
    parser.add_argument("--config", dest="config_flag", help="Path to a JSON training config.")
    parsed, unknown = parser.parse_known_args()
    if unknown:
        parser.error(
            "all training options now belong in the config JSON; only --config is accepted"
        )
    config_path = parsed.config_flag or parsed.config
    if config_path is None:
        parser.error("provide a config path, for example: python vanilla_gaussian_splatting.py --config train_config.json")
    return load_train_config(config_path)


if __name__ == "__main__":
    train(parse_args())

