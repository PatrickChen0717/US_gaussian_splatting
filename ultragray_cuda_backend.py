import importlib
import sys
from pathlib import Path

import torch


_ULTRAGRAY_RASTERIZER = None
_ULTRAGRAY_ROOT = None
_UTF8_HASH_PATCHED = False


def _candidate_repo_paths(configured_path=None):
    if configured_path:
        yield Path(configured_path).expanduser()

    repo_dir = Path(__file__).resolve().parent
    for name in ("UltraG-Ray", "UltraG_Ray", "ultragray"):
        yield repo_dir.parent / name


def _enable_utf8_extension_hashing():
    global _UTF8_HASH_PATCHED
    if _UTF8_HASH_PATCHED:
        return

    # PyTorch 2.4 reads extension sources using the Windows locale while
    # computing its JIT cache hash. UltraG-Ray contains UTF-8 CUDA comments.
    if sys.platform == "win32":
        import distutils
        import distutils._msvccompiler as msvccompiler
        from torch.utils import _cpp_extension_versioner as versioner

        if not hasattr(distutils, "_msvccompiler"):
            distutils._msvccompiler = msvccompiler

        def hash_source_files_utf8(hash_value, source_files):
            for filename in source_files:
                with open(filename, encoding="utf-8") as source:
                    hash_value = versioner.update_hash(hash_value, source.read())
            return hash_value

        versioner.hash_source_files = hash_source_files_utf8
    _UTF8_HASH_PATCHED = True


def load_ultragray_rasterizer(configured_path=None):
    global _ULTRAGRAY_RASTERIZER, _ULTRAGRAY_ROOT
    if _ULTRAGRAY_RASTERIZER is not None:
        return _ULTRAGRAY_RASTERIZER

    root = next(
        (
            candidate.resolve()
            for candidate in _candidate_repo_paths(configured_path)
            if (candidate / "gsplat" / "rendering.py").is_file()
        ),
        None,
    )
    if root is None:
        raise FileNotFoundError(
            "Could not find UltraG-Ray. Set ultragray_repo_path in train_config.json."
        )

    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    _enable_utf8_extension_hashing()

    loaded_gsplat = sys.modules.get("gsplat")
    if loaded_gsplat is not None:
        loaded_path = Path(getattr(loaded_gsplat, "__file__", "")).resolve()
        if root not in loaded_path.parents:
            raise ImportError(
                f"A different gsplat package is already loaded from {loaded_path}. "
                "Start a fresh Python process before using ultragray_cuda."
            )

    rendering = importlib.import_module("gsplat.rendering")
    _ULTRAGRAY_RASTERIZER = rendering.ultrasound_rasterization
    _ULTRAGRAY_ROOT = root
    return _ULTRAGRAY_RASTERIZER


def _matrix4(value, device, dtype):
    matrix = torch.as_tensor(value, device=device, dtype=dtype)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {tuple(matrix.shape)}")
    return matrix


def _camera_to_world(
    slice_to_world,
    image_t_probe,
    image_plane_origin_px,
    spacing_x,
    spacing_y,
    image_width,
):
    device = slice_to_world.device
    dtype = slice_to_world.dtype
    image_t_probe = _matrix4(image_t_probe, device, dtype)
    probe_t_image = torch.linalg.inv(image_t_probe)

    origin_px = torch.as_tensor(
        image_plane_origin_px,
        device=device,
        dtype=dtype,
    )
    image_t_camera = torch.eye(4, device=device, dtype=dtype)
    # CUDA camera axes: X=lateral, Y=elevational normal, Z=axial depth.
    image_t_camera[:3, :3] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    image_t_camera[0, 3] = (image_width * 0.5 - origin_px[0]) * spacing_x
    image_t_camera[1, 3] = -origin_px[1] * spacing_y
    return slice_to_world @ probe_t_image @ image_t_camera


def render_ultrasound_cuda(
    means,
    scales,
    colors,
    opacities,
    transmittances,
    slice_to_world,
    image_height,
    image_width,
    pixel_spacing,
    image_t_probe,
    image_plane_origin_px,
    pixel_to_mm,
    image_scale,
    shadowing,
    shadow_strength,
    max_visible_gaussians_per_slice=None,
    ultragray_repo_path=None,
    tile_size_x=4,
    tile_size_y=128,
):
    if means.device.type != "cuda":
        raise RuntimeError("ultragray_cuda renderer requires a CUDA device")

    calibrated_spacing = float(pixel_to_mm) * float(image_scale)
    if calibrated_spacing > 0.0:
        spacing_x = calibrated_spacing
        spacing_y = calibrated_spacing
    else:
        spacing_x = float(pixel_spacing[0])
        spacing_y = float(pixel_spacing[1])
    if spacing_x <= 0.0 or spacing_y <= 0.0:
        raise ValueError("CUDA renderer requires positive physical pixel spacing")

    camera_to_world = _camera_to_world(
        slice_to_world,
        image_t_probe,
        image_plane_origin_px,
        spacing_x,
        spacing_y,
        image_width,
    )
    world_to_camera = torch.linalg.inv(camera_to_world)
    means_h = torch.cat(
        [means, torch.ones_like(means[:, :1])],
        dim=-1,
    )
    camera_means = (world_to_camera @ means_h.T).T[:, :3]

    opacity_probability = torch.sigmoid(opacities).reshape(-1)
    if (
        max_visible_gaussians_per_slice is not None
        and int(max_visible_gaussians_per_slice) > 0
        and len(camera_means) > int(max_visible_gaussians_per_slice)
    ):
        keep = torch.topk(
            opacity_probability.detach(),
            int(max_visible_gaussians_per_slice),
        ).indices
        camera_means = camera_means[keep]
        scales = scales[keep]
        colors = colors[keep]
        opacity_probability = opacity_probability[keep]
        if transmittances is not None:
            transmittances = transmittances[keep]

    # Model scale order is lateral, axial, elevational. CUDA camera order is X, Y, Z.
    camera_scales = scales[:, [0, 2, 1]]
    quats = torch.zeros(
        (len(camera_means), 4),
        device=means.device,
        dtype=means.dtype,
    )
    quats[:, 0] = 1.0

    intensities = colors.clamp(0.0, 1.0) * opacity_probability[:, None]
    if transmittances is None or not shadowing:
        local_transmittance = torch.ones_like(opacity_probability)
    else:
        local_transmittance = torch.sigmoid(transmittances).reshape(-1)
        if shadow_strength != 1.0:
            attenuation = (1.0 - local_transmittance) * float(shadow_strength)
            local_transmittance = (1.0 - attenuation).clamp(0.0, 1.0)

    rasterizer = load_ultragray_rasterizer(ultragray_repo_path)
    identity_view = torch.eye(
        4,
        device=means.device,
        dtype=means.dtype,
    ).unsqueeze(0)
    opening_width = float(image_width) * spacing_x
    far_plane = float(image_height) * spacing_y

    rendered, _, _, _, info = rasterizer(
        means=camera_means,
        quats=quats,
        scales=camera_scales,
        transmittances=local_transmittance,
        intensities=intensities,
        viewmats=identity_view,
        width=int(image_width),
        height=int(image_height),
        near_plane=0.0,
        far_plane=far_plane,
        opening_angle=None,
        opening_width=opening_width,
        tile_size_x=int(tile_size_x),
        tile_size_y=int(tile_size_y),
        sh_degree=None,
    )
    return rendered.squeeze(0).permute(2, 0, 1).clamp(0.0, 1.0), info
