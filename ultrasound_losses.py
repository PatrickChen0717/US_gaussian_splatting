import torch
import torch.nn.functional as F


def ensure_bchw(image):
    if image.ndim == 2:
        return image.unsqueeze(0).unsqueeze(0)
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError("image must have shape [H, W], [C, H, W], or [B, C, H, W]")


def gaussian_kernel(kernel_size, sigma, channels, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - (kernel_size - 1) * 0.5
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def gaussian_blur(image, kernel_size=9, sigma=1.5):
    image = ensure_bchw(image)
    channels = image.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels, image.device, image.dtype)
    return F.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)


def robust_normalize(image, eps=1e-8):
    image = ensure_bchw(image)
    mean = image.mean(dim=(-2, -1), keepdim=True)
    std = image.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (image - mean) / std


def laplacian_pyramid_high_pass(image, kernel_size=9, sigma=1.5):
    image = ensure_bchw(image)
    return image - gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)


def multiscale_laplacian_pyramid(image, sigmas=(0.8, 1.5, 3.0), kernel_size=None):
    image = ensure_bchw(image)
    response = torch.zeros_like(image)
    for sigma in sigmas:
        current_kernel_size = kernel_size
        if current_kernel_size is None:
            current_kernel_size = max(3, int(2 * round(3 * sigma) + 1))
        response = response + laplacian_pyramid_high_pass(image, current_kernel_size, sigma).abs()
    return response / max(len(sigmas), 1)


def sobel_edges(image):
    image = ensure_bchw(image)
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


def ultrasound_edge_map(
    image,
    sigmas=(0.8, 1.5, 3.0),
    sobel_weight=1.0,
    kernel_size=None,
    sobel_blur_kernel_size=5,
    sobel_blur_sigma=0.8,
):
    image = robust_normalize(image)
    detail = multiscale_laplacian_pyramid(image, sigmas=sigmas, kernel_size=kernel_size)
    gradient = sobel_edges(gaussian_blur(image, kernel_size=sobel_blur_kernel_size, sigma=sobel_blur_sigma))
    return detail + sobel_weight * gradient


def grayscale_bchw(image):
    image = ensure_bchw(image)
    if image.shape[1] == 1:
        return image
    return image.mean(dim=1, keepdim=True)


def ultrasound_confidence_map(
    target,
    background_threshold=0.02,
    background_weight=0.0,
    dark_threshold=0.08,
    shadow_weight=0.2,
    bright_threshold=0.65,
    shadow_start_offset=8,
    enable_shadow=True,
):
    """
    Build a soft reliability map from the target ultrasound slice.

    Low/black background receives low weight. If a bright reflector appears in
    an A-line/column, dark pixels deeper than it are downweighted as likely
    acoustic shadow/dropout.
    """
    gray = grayscale_bchw(target).detach()
    gray = gray.clamp(0.0, 1.0)
    confidence = torch.ones_like(gray)

    if background_threshold is not None:
        confidence = torch.where(
            gray <= background_threshold,
            torch.full_like(confidence, background_weight),
            confidence,
        )

    if enable_shadow:
        bright = gray >= bright_threshold
        if shadow_start_offset > 0:
            bright = F.pad(bright.float(), (0, 0, shadow_start_offset, 0))[:, :, :-shadow_start_offset, :].bool()
        bright_above = torch.cumsum(bright.float(), dim=-2) > 0
        dark_below = gray <= dark_threshold
        shadow = bright_above & dark_below
        confidence = torch.where(
            shadow,
            torch.full_like(confidence, shadow_weight),
            confidence,
        )

    return confidence.clamp(0.0, 1.0)


def weighted_l1(pred, target, weight=None, eps=1e-8):
    error = (pred - target).abs()
    if weight is None:
        return error.mean()
    weight = ensure_bchw(weight).to(device=error.device, dtype=error.dtype)
    if weight.shape[1] == 1 and error.shape[1] != 1:
        weight = weight.expand(-1, error.shape[1], -1, -1)
    return (error * weight).sum() / weight.sum().clamp_min(eps)


def content_weight_map(
    target,
    feature=None,
    intensity_threshold=0.03,
    feature_threshold=0.05,
    background_weight=0.05,
    eps=1e-8,
):
    """
    Weight useful ultrasound content more than empty dark background.

    The returned map is still nonzero in dark regions, so empty slices can
    penalize hallucinated signal, but most gradient comes from visible tissue
    and structural edges.
    """
    gray = grayscale_bchw(target).detach().clamp(0.0, 1.0)
    content = torch.where(gray >= intensity_threshold, gray, torch.zeros_like(gray))

    if feature is not None:
        feature_gray = grayscale_bchw(feature).detach().abs()
        feature_scale = feature_gray.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
        feature_norm = feature_gray / feature_scale
        feature_content = torch.where(
            feature_norm >= feature_threshold,
            feature_norm,
            torch.zeros_like(feature_norm),
        )
        content = torch.maximum(content, feature_content)

    return content.clamp_min(background_weight).clamp(0.0, 1.0)


def combine_weights(*weights):
    combined = None
    for weight in weights:
        if weight is None:
            continue
        weight = ensure_bchw(weight)
        combined = weight if combined is None else combined * weight
    return combined


def ultrasound_edge_loss(
    pred,
    target,
    laplacian_weight=1.0,
    edge_weight=1.0,
    intensity_weight=0.0,
    sigmas=(0.8, 1.5, 3.0),
    kernel_size=None,
    sobel_weight=1.0,
    sobel_blur_kernel_size=5,
    sobel_blur_sigma=0.8,
    confidence=None,
    use_confidence=False,
    background_threshold=0.02,
    background_weight=0.0,
    dark_threshold=0.08,
    shadow_weight=0.2,
    bright_threshold=0.65,
    shadow_start_offset=8,
    enable_shadow_confidence=True,
    content_normalize=True,
    content_intensity_threshold=0.03,
    content_feature_threshold=0.05,
    content_background_weight=0.05,
):
    """
    Shadow-robust ultrasound loss.

    Compares local high-pass and edge structure instead of trusting raw
    intensity everywhere. Keep intensity_weight small or zero for shadowed data.
    """
    pred = ensure_bchw(pred)
    target = ensure_bchw(target)
    if use_confidence and confidence is None:
        confidence = ultrasound_confidence_map(
            target,
            background_threshold=background_threshold,
            background_weight=background_weight,
            dark_threshold=dark_threshold,
            shadow_weight=shadow_weight,
            bright_threshold=bright_threshold,
            shadow_start_offset=shadow_start_offset,
            enable_shadow=enable_shadow_confidence,
        )

    pred_norm = robust_normalize(pred)
    target_norm = robust_normalize(target)

    loss = pred.sum() * 0.0
    if laplacian_weight:
        pred_lap = multiscale_laplacian_pyramid(pred_norm, sigmas=sigmas, kernel_size=kernel_size)
        target_lap = multiscale_laplacian_pyramid(target_norm, sigmas=sigmas, kernel_size=kernel_size)
        lap_weight = confidence
        if content_normalize:
            lap_weight = combine_weights(
                confidence,
                content_weight_map(
                    target,
                    feature=target_lap,
                    intensity_threshold=content_intensity_threshold,
                    feature_threshold=content_feature_threshold,
                    background_weight=content_background_weight,
                ),
            )
        loss = loss + laplacian_weight * weighted_l1(pred_lap, target_lap, lap_weight)

    if edge_weight:
        pred_edges = ultrasound_edge_map(
            pred,
            sigmas=sigmas,
            sobel_weight=sobel_weight,
            kernel_size=kernel_size,
            sobel_blur_kernel_size=sobel_blur_kernel_size,
            sobel_blur_sigma=sobel_blur_sigma,
        )
        target_edges = ultrasound_edge_map(
            target,
            sigmas=sigmas,
            sobel_weight=sobel_weight,
            kernel_size=kernel_size,
            sobel_blur_kernel_size=sobel_blur_kernel_size,
            sobel_blur_sigma=sobel_blur_sigma,
        )
        edge_loss_weight = confidence
        if content_normalize:
            edge_loss_weight = combine_weights(
                confidence,
                content_weight_map(
                    target,
                    feature=target_edges,
                    intensity_threshold=content_intensity_threshold,
                    feature_threshold=content_feature_threshold,
                    background_weight=content_background_weight,
                ),
            )
        loss = loss + edge_weight * weighted_l1(pred_edges, target_edges, edge_loss_weight)

    if intensity_weight:
        intensity_loss_weight = confidence
        if content_normalize:
            intensity_loss_weight = combine_weights(
                confidence,
                content_weight_map(
                    target,
                    intensity_threshold=content_intensity_threshold,
                    background_weight=content_background_weight,
                ),
            )
        loss = loss + intensity_weight * weighted_l1(pred_norm, target_norm, intensity_loss_weight)

    return loss
