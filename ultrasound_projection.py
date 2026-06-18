import torch


DEFAULT_IMAGE_T_PROBE = (
    (0.0, 1.0, 0.0, -0.0),
    (1.0, -0.0, 0.0, -75.5),
    (0.0, -0.0, -1.0, -14.34),
    (0.0, 0.0, 0.0, 1.0),
)
DEFAULT_IMAGE_PLANE_ORIGIN_PX = (299.0, 5.0)
DEFAULT_PIXEL_TO_MM = 160.0 / 726.0


def project_ultrasound_gaussians(
    means,
    scales,
    slice_to_world,
    image_height,
    image_width,
    pixel_spacing=(1.0, 1.0),
    slice_thickness=1.0,
    min_sigma_pixels=0.5,
    image_t_probe=None,
    image_plane_origin_px=DEFAULT_IMAGE_PLANE_ORIGIN_PX,
    pixel_to_mm=DEFAULT_PIXEL_TO_MM,
    image_scale=1.0,
    covariance_mode="ultrasound_psf",
    min_scale_mm=0.05,
    max_scale_mm=10.0,
    lateral_depth_slope=0.0,
    elevational_depth_slope=0.0,
    primitive_mode="volume",
    disk_normals=None,
    covariances=None,
):
    """
    Project 3D Gaussians onto a tracked ultrasound slice.

    The geometry follows the EchoRaccoon calibration chain:

        tracker/world point -> probe coordinates -> calibrated image coordinates

    The image coordinates are physical millimeters in the ultrasound image plane.
    They are converted to pixels using image_plane_origin_px and pixel_to_mm.
    """
    device = means.device
    dtype = means.dtype
    scales = _as_anisotropic_scales(scales, device, dtype)
    tracker_t_probe = slice_to_world
    probe_t_tracker = torch.linalg.inv(tracker_t_probe)
    image_t_probe = _matrix4(
        DEFAULT_IMAGE_T_PROBE if image_t_probe is None else image_t_probe,
        device=device,
        dtype=dtype,
    )
    image_t_tracker = image_t_probe @ probe_t_tracker

    means_h = torch.cat(
        [means, torch.ones((len(means), 1), device=device, dtype=dtype)],
        dim=-1,
    )
    means_image = (image_t_tracker @ means_h.T).T[:, :3]

    origin_px = torch.as_tensor(image_plane_origin_px, device=device, dtype=dtype)
    calibrated_spacing = pixel_to_mm * image_scale
    if calibrated_spacing > 0.0:
        spacing_x = calibrated_spacing
        spacing_y = calibrated_spacing
    else:
        spacing_x, spacing_y = pixel_spacing

    u = means_image[:, 0] / spacing_x + origin_px[0]
    v = means_image[:, 1] / spacing_y + origin_px[1]
    plane_distance = means_image[:, 2]

    if covariance_mode == "full_cholesky":
        sigma_image = gaussian_scales_in_image_frame(
            scales=scales,
            means_image=means_image,
            image_t_tracker=image_t_tracker,
            covariance_mode=covariance_mode,
            min_scale_mm=min_scale_mm,
            max_scale_mm=max_scale_mm,
            lateral_depth_slope=lateral_depth_slope,
            elevational_depth_slope=elevational_depth_slope,
            covariances=covariances,
        )
    elif primitive_mode == "disk":
        if disk_normals is None:
            raise ValueError("disk_normals must be provided when primitive_mode='disk'")
        sigma_image = gaussian_disk_scales_in_image_frame(
            scales=scales,
            disk_normals=disk_normals,
            image_t_tracker=image_t_tracker,
            min_scale_mm=min_scale_mm,
            max_scale_mm=max_scale_mm,
        )
    elif primitive_mode == "dot":
        sigma_image = gaussian_dot_scales_in_image_frame(
            scales=scales,
            min_scale_mm=min_scale_mm,
            max_scale_mm=max_scale_mm,
        )
    else:
        sigma_image = gaussian_scales_in_image_frame(
            scales=scales,
            means_image=means_image,
            image_t_tracker=image_t_tracker,
            covariance_mode=covariance_mode,
            min_scale_mm=min_scale_mm,
            max_scale_mm=max_scale_mm,
            lateral_depth_slope=lateral_depth_slope,
            elevational_depth_slope=elevational_depth_slope,
            covariances=covariances,
        )
    sigma_x = (sigma_image[:, 0] / spacing_x).clamp_min(min_sigma_pixels)
    sigma_y = (sigma_image[:, 1] / spacing_y).clamp_min(min_sigma_pixels)
    sigma_z = sigma_image[:, 2].clamp_min(slice_thickness)

    plane_weight = torch.exp(-0.5 * (plane_distance / sigma_z).square())
    radius_x = 3.0 * sigma_x
    radius_y = 3.0 * sigma_y

    visible = (
        (plane_weight > 1e-4)
        & (u >= -radius_x)
        & (u < image_width + radius_x)
        & (v >= -radius_y)
        & (v < image_height + radius_y)
    )
    finite = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(plane_distance)
        & torch.isfinite(plane_weight)
        & torch.isfinite(sigma_x)
        & torch.isfinite(sigma_y)
        & (sigma_x > 0.0)
        & (sigma_y > 0.0)
    )
    visible = visible & finite

    return {
        "xys": torch.stack([u, v], dim=-1),
        "axial_depths": means_image[:, 1],
        "plane_distances": plane_distance,
        "plane_weights": plane_weight,
        "sigmas": torch.stack([sigma_x, sigma_y], dim=-1),
        "radii": torch.stack([radius_x, radius_y], dim=-1),
        "visible": visible,
        "means_image": means_image,
        "sigma_image": sigma_image,
    }


def render_ultrasound_gaussians(
    means,
    scales,
    colors,
    opacities,
    slice_to_world,
    image_height,
    image_width,
    transmittances=None,
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
    disk_normals=None,
    acoustic_rendering=False,
    attenuation_weight=0.0,
    reflection_weight=0.0,
    scattering_weight=0.0,
    covariances=None,
    max_visible_gaussians_per_slice=None,
):
    """
    Render an ultrasound slice from 3D Gaussians.

    The image is formed by drawing the intersection of each Gaussian with the
    tracked slice plane. If shadowing is enabled, a separate per-Gaussian
    transmittance can attenuate deeper contributions along the axial image
    direction.
    """
    projected = project_ultrasound_gaussians(
        means,
        scales,
        slice_to_world,
        image_height,
        image_width,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
        image_t_probe=image_t_probe,
        image_plane_origin_px=image_plane_origin_px,
        pixel_to_mm=pixel_to_mm,
        image_scale=image_scale,
        covariance_mode=covariance_mode,
        min_scale_mm=min_scale_mm,
        max_scale_mm=max_scale_mm,
        lateral_depth_slope=lateral_depth_slope,
        elevational_depth_slope=elevational_depth_slope,
        primitive_mode=primitive_mode,
        disk_normals=disk_normals,
        covariances=covariances,
    )

    visible = projected["visible"]
    channels = colors.shape[-1]
    device = means.device
    image = torch.zeros((channels, image_height, image_width), device=device)

    if not visible.any():
        return image + means.sum() * 0.0, projected

    xys = projected["xys"][visible]
    sigmas = projected["sigmas"][visible]
    depths = projected["axial_depths"][visible]
    plane_weights = projected["plane_weights"][visible]
    colors = colors[visible].clamp(0.0, 1.0)
    if acoustic_rendering:
        colors = acoustic_intensity(
            colors,
            projected["axial_depths"][visible],
            attenuation_weight=attenuation_weight,
            reflection_weight=reflection_weight,
            scattering_weight=scattering_weight,
        )
    opacities = opacities[visible].reshape(-1).sigmoid() * plane_weights
    if transmittances is None:
        local_transmittances = None
    else:
        local_transmittances = transmittances[visible].reshape(-1).sigmoid()

    if (
        max_visible_gaussians_per_slice is not None
        and int(max_visible_gaussians_per_slice) > 0
        and len(xys) > int(max_visible_gaussians_per_slice)
    ):
        keep_count = int(max_visible_gaussians_per_slice)
        scores = opacities.detach()
        keep = torch.topk(scores, keep_count, largest=True).indices
        xys = xys[keep]
        sigmas = sigmas[keep]
        depths = depths[keep]
        plane_weights = plane_weights[keep]
        colors = colors[keep]
        opacities = opacities[keep]
        if local_transmittances is not None:
            local_transmittances = local_transmittances[keep]

    yy, xx = torch.meshgrid(
        torch.arange(image_height, device=device, dtype=means.dtype),
        torch.arange(image_width, device=device, dtype=means.dtype),
        indexing="ij",
    )

    # assumption of even attenuation across the slice thickness
    if shadowing:
        order = torch.argsort(depths)
        xys = xys[order]
        sigmas = sigmas[order]
        opacities = opacities[order]
        plane_weights = plane_weights[order]
        if local_transmittances is not None:
            local_transmittances = local_transmittances[order]
        colors = colors[order]
        accumulated_transmittance = torch.ones(
            (1, image_height, image_width), device=device
        )
        for start in range(0, len(xys), max(int(render_chunk_size), 1)):
            end = min(start + max(int(render_chunk_size), 1), len(xys))
            alpha = gaussian_alpha_chunk(
                xx,
                yy,
                xys[start:end],
                sigmas[start:end],
                opacities[start:end],
            )
            if local_transmittances is None:
                shadow_alpha = alpha
            else:
                shadow_alpha = gaussian_alpha_chunk(
                    xx,
                    yy,
                    xys[start:end],
                    sigmas[start:end],
                    plane_weights[start:end],
                )
            for local_index, (alpha_i, shadow_alpha_i, color_i) in enumerate(
                zip(alpha, shadow_alpha, colors[start:end])
            ):
                alpha_i = alpha_i.unsqueeze(0)
                shadow_alpha_i = shadow_alpha_i.unsqueeze(0)
                if local_transmittances is None:
                    gaussian_transmittance = torch.exp(
                        -shadow_strength * shadow_alpha_i
                    )
                else:
                    local_t = local_transmittances[start + local_index]
                    gaussian_transmittance = 1.0 - shadow_alpha_i * (1.0 - local_t)
                    if shadow_strength != 1.0:
                        gaussian_transmittance = gaussian_transmittance.clamp_min(
                            1e-6
                        ).pow(shadow_strength)
                image = (
                    image
                    + accumulated_transmittance * alpha_i * color_i[:, None, None]
                )
                accumulated_transmittance = (
                    accumulated_transmittance
                    * gaussian_transmittance.clamp(0.0, 1.0)
                )
    else:
        for start in range(0, len(xys), max(int(render_chunk_size), 1)):
            end = min(start + max(int(render_chunk_size), 1), len(xys))
            alpha = gaussian_alpha_chunk(
                xx,
                yy,
                xys[start:end],
                sigmas[start:end],
                opacities[start:end],
            )
            image = image + (alpha[:, None] * colors[start:end, :, None, None]).sum(dim=0)

    return image.clamp(0.0, 1.0), projected


def gaussian_alpha_chunk(xx, yy, xys, sigmas, opacities):
    dx = (xx.unsqueeze(0) - xys[:, 0, None, None]) / sigmas[:, 0, None, None]
    dy = (yy.unsqueeze(0) - xys[:, 1, None, None]) / sigmas[:, 1, None, None]
    alpha = torch.exp(-0.5 * (dx.square() + dy.square()))
    return (alpha * opacities[:, None, None]).clamp(0.0, 0.99)


def gaussian_scales_in_image_frame(
    scales,
    means_image,
    image_t_tracker,
    covariance_mode="ultrasound_psf",
    min_scale_mm=0.05,
    max_scale_mm=10.0,
    lateral_depth_slope=0.0,
    elevational_depth_slope=0.0,
    covariances=None,
):
    """
    Return Gaussian standard deviations in calibrated image coordinates.

    ultrasound_psf mode interprets scales as [lateral, axial, elevational] mm.
    world_axis_aligned mode interprets scales as [world_x, world_y, world_z] mm
    and rotates their diagonal covariance into the image frame.
    """
    scales = scales.clamp(min_scale_mm, max_scale_mm)
    if covariance_mode == "ultrasound_psf":
        axial_depth = means_image[:, 1].abs()
        psf_scales = scales.clone()
        psf_scales[:, 0] = psf_scales[:, 0] + lateral_depth_slope * axial_depth
        psf_scales[:, 2] = psf_scales[:, 2] + elevational_depth_slope * axial_depth
        return psf_scales.clamp(min_scale_mm, max_scale_mm)

    if covariance_mode == "world_axis_aligned":
        return project_axis_aligned_covariance_to_image(scales, image_t_tracker)

    if covariance_mode == "full_cholesky":
        if covariances is None:
            raise ValueError("covariances must be provided when covariance_mode='full_cholesky'")
        return project_full_covariance_to_image(covariances, image_t_tracker, min_scale_mm, max_scale_mm)

    raise ValueError(f"Unknown covariance_mode: {covariance_mode}")


def gaussian_dot_scales_in_image_frame(
    scales,
    min_scale_mm=0.05,
    max_scale_mm=10.0,
):
    """
    Treat each primitive as an isotropic 3D dot.

    A dot uses one radius in all directions, so its projected standard deviation
    is stable across slice orientations and less prone to forming anisotropic
    sheet-like blobs.
    """
    scales = torch.nan_to_num(scales, nan=min_scale_mm, posinf=max_scale_mm, neginf=min_scale_mm)
    radius = scales.clamp(min_scale_mm, max_scale_mm).mean(dim=-1, keepdim=True)
    return radius.repeat(1, 3).clamp(min_scale_mm, max_scale_mm)


def gaussian_disk_scales_in_image_frame(
    scales,
    disk_normals,
    image_t_tracker,
    min_scale_mm=0.05,
    max_scale_mm=10.0,
):
    """
    Approximate an oriented 2D Gaussian disk as a flattened 3D covariance.

    scales are [tangent_u, tangent_v, normal_thickness] in mm. The learnable
    disk normal defines the thin direction. The diagonal standard deviations
    in image coordinates are used by the lightweight rasterizer.
    """
    scales = torch.nan_to_num(scales, nan=min_scale_mm, posinf=max_scale_mm, neginf=min_scale_mm)
    scales = scales.clamp(min_scale_mm, max_scale_mm)
    disk_normals = torch.nan_to_num(disk_normals, nan=0.0, posinf=1.0, neginf=-1.0)
    normals = torch.nn.functional.normalize(disk_normals, dim=-1, eps=1e-8)
    tangent_u, tangent_v = orthonormal_tangent_basis(normals)

    basis = torch.stack([tangent_u, tangent_v, normals], dim=-1)
    variances = scales.square()
    covariance_world = basis @ torch.diag_embed(variances) @ basis.transpose(-1, -2)

    rotation = image_t_tracker[:3, :3]
    covariance_image = rotation @ covariance_world @ rotation.T
    image_variances = covariance_image.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
    return image_variances.sqrt().clamp(min_scale_mm, max_scale_mm)


def orthonormal_tangent_basis(normals):
    reference_z = torch.zeros_like(normals)
    reference_z[:, 2] = 1.0
    reference_y = torch.zeros_like(normals)
    reference_y[:, 1] = 1.0
    use_y = normals[:, 2].abs() > 0.9
    reference = torch.where(use_y[:, None], reference_y, reference_z)
    tangent_u = torch.cross(reference, normals, dim=-1)
    tangent_u = torch.nn.functional.normalize(tangent_u, dim=-1, eps=1e-8)
    tangent_v = torch.cross(normals, tangent_u, dim=-1)
    tangent_v = torch.nn.functional.normalize(tangent_v, dim=-1, eps=1e-8)
    return tangent_u, tangent_v


def acoustic_intensity(
    colors,
    axial_depths,
    attenuation_weight=0.0,
    reflection_weight=0.0,
    scattering_weight=0.0,
):
    depth = axial_depths.abs()
    depth = depth / depth.detach().amax().clamp_min(1e-8)
    attenuation = -attenuation_weight * depth[:, None]
    reflection = reflection_weight * colors.square()
    scattering = scattering_weight * colors * colors.mean(dim=-1, keepdim=True)
    return (colors + attenuation + reflection + scattering).clamp(0.0, 1.0)


def project_axis_aligned_covariance_to_image(scales, image_t_tracker):
    """
    Approximate 3D covariance projection into calibrated image coordinates.

    scales are interpreted as standard deviations along tracker/world x, y, z.
    This rotates that diagonal covariance into the image/probe frame and returns
    only the diagonal standard deviations [image_x, image_y, image_z].
    """
    rotation = image_t_tracker[:3, :3]
    variances = scales.square()
    image_variances = variances @ rotation.square().T
    return image_variances.clamp_min(1e-12).sqrt()


def project_full_covariance_to_image(covariances, image_t_tracker, min_scale_mm=0.05, max_scale_mm=10.0):
    """
    Rotate full 3D covariance matrices into calibrated image coordinates.

    covariances are positive-definite matrices built upstream as L @ L.T + eps I.
    The lightweight rasterizer still uses diagonal standard deviations in image
    coordinates, but those diagonals now come from a full learnable covariance.
    """
    rotation = image_t_tracker[:3, :3]
    covariance_image = rotation @ covariances @ rotation.T
    image_variances = covariance_image.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
    return image_variances.sqrt().clamp(min_scale_mm, max_scale_mm)


def _matrix4(matrix, device, dtype):
    matrix = torch.as_tensor(matrix, device=device, dtype=dtype)
    if matrix.shape != (4, 4):
        raise ValueError("calibration transform must have shape [4, 4]")
    return matrix


def _as_anisotropic_scales(scales, device, dtype):
    scales = torch.as_tensor(scales, device=device, dtype=dtype)
    if scales.ndim == 1:
        return scales[:, None].repeat(1, 3)
    if scales.shape[-1] == 1:
        return scales.repeat(1, 3)
    if scales.shape[-1] != 3:
        raise ValueError("scales must have shape [N], [N, 1], or [N, 3]")
    return scales

