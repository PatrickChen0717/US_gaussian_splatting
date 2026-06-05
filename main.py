import torch
from ultrasound_projection import render_ultrasound_gaussians
from ultrasound_losses import ultrasound_edge_loss

# 1. Initialize Gaussians (Positions, Scales, Rotations, Opacity, Colors)
# In your case, 'means' could be initialized from your SVR starting stack
num_points = 1000
means = torch.randn((num_points, 3), device="cuda", requires_grad=True)
scales = torch.full((num_points, 3), 2.0, device="cuda", requires_grad=True)
opacities = torch.ones((num_points, 1), device="cuda", requires_grad=True)
colors = torch.rand((num_points, 3), device="cuda", requires_grad=True)

# 2. Define Ultrasound Slice Parameters
# 'viewmat' is the Transformation Matrix T from your SVR paper
H, W = 512, 512
slice_to_world = torch.eye(4, device="cuda")
pixel_spacing = (0.3, 0.3)  # physical units per pixel: lateral, axial
slice_thickness = 1.0       # elevational PSF width / slice thickness

# 3. Forward Pass: Intersect 3D Gaussians with the ultrasound slice plane
out_img, projected = render_ultrasound_gaussians(
    means,
    scales,
    colors,
    opacities,
    slice_to_world,
    H,
    W,
    pixel_spacing=pixel_spacing,
    slice_thickness=slice_thickness,
    shadowing=True,
    shadow_strength=1.0,
)

# 4. Shadow-robust Loss & Optimization
# Compare local ultrasound detail/edge maps instead of raw intensity.
target_slice = torch.randn((H, W, 3), device="cuda") # Your actual US frame
target_slice = target_slice.permute(2, 0, 1)
loss = ultrasound_edge_loss(
    out_img,
    target_slice,
    laplacian_weight=1.0,
    edge_weight=1.0,
    intensity_weight=0.05,
    sobel_weight=1.5,
)
loss.backward()

print(f"Loss: {loss.item()}")

