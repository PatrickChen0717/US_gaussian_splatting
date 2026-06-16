from pathlib import Path
import io
import math

from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("outputs/method_slides")
W, H = 1600, 900

NAVY = (20, 48, 120)
TEAL = (0, 126, 148)
TEAL_DARK = (0, 92, 108)
INK = (24, 30, 42)
MUTED = (82, 95, 120)
LIGHT = (246, 248, 252)
LINE = (178, 188, 204)
PALE_TEAL = (226, 246, 249)
PALE_BLUE = (234, 240, 252)
ORANGE = (213, 117, 54)


def font(size, bold=False, italic=False):
    candidates = []
    if bold:
        candidates.extend([
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
        ])
    elif italic:
        candidates.extend([
            "C:/Windows/Fonts/ariali.ttf",
            "C:/Windows/Fonts/segoeuii.ttf",
        ])
    candidates.extend([
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ])
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(62, bold=True)
F_SUB = font(28, italic=True)
F_H1 = font(30, bold=True)
F_H2 = font(24, bold=True)
F_BODY = font(24)
F_SMALL = font(18)
F_TINY = font(15)
F_EQ = font(20)
F_NUM = font(22, bold=True)


def text(draw, xy, s, f=F_BODY, fill=INK, anchor=None):
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def wrap_text(draw, s, f, max_width):
    words = s.split()
    lines = []
    line = ""
    for word in words:
        candidate = word if not line else line + " " + word
        if draw.textbbox((0, 0), candidate, font=f)[2] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def paragraph(draw, x, y, s, f=F_BODY, fill=INK, max_width=400, leading=1.25):
    lines = wrap_text(draw, s, f, max_width)
    line_h = int(f.size * leading)
    for i, line in enumerate(lines):
        text(draw, (x, y + i * line_h), line, f, fill)
    return y + len(lines) * line_h


def header(draw, title, subtitle):
    text(draw, (44, 34), title, F_TITLE, NAVY)
    text(draw, (48, 116), subtitle, F_SUB, (73, 94, 145))
    draw.line((44, 172, W - 44, 172), fill=(140, 145, 155), width=2)


def badge(draw, x, y, n, r=22):
    draw.ellipse((x - r, y - r, x + r, y + r), fill=TEAL, outline=TEAL_DARK, width=2)
    text(draw, (x, y + 1), str(n), F_NUM, (255, 255, 255), anchor="mm")


def rounded(draw, box, radius=18, fill=(255, 255, 255), outline=TEAL, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill=(80, 80, 80), width=4):
    draw.line((start, end), fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    ang = math.atan2(y2 - y1, x2 - x1)
    length = 16
    spread = 0.45
    pts = [
        end,
        (x2 - length * math.cos(ang - spread), y2 - length * math.sin(ang - spread)),
        (x2 - length * math.cos(ang + spread), y2 - length * math.sin(ang + spread)),
    ]
    draw.polygon(pts, fill=fill)


def ultrasound_panel(draw, x, y, w, h, label=None):
    draw.rectangle((x, y, x + w, y + h), fill=(8, 10, 14), outline=(35, 42, 55), width=2)
    cx, cy = x + w / 2, y + h * 0.15
    for i in range(18):
        yy = y + 20 + i * h / 22
        amp = 12 + 10 * math.sin(i * 0.9)
        draw.arc((cx - amp - i * 3, yy - 8, cx + amp + i * 3, yy + 16), 190, 350, fill=(145, 155, 160), width=1)
    for i in range(140):
        px = x + 12 + (i * 37) % max(w - 24, 1)
        py = y + 20 + (i * 53) % max(h - 34, 1)
        val = 90 + (i * 31) % 155
        draw.point((px, py), fill=(val, val, val))
    draw.ellipse((x + w * 0.44, y + h * 0.56, x + w * 0.62, y + h * 0.78), outline=(230, 230, 230), width=3)
    if label:
        text(draw, (x, y - 28), label, F_SMALL, INK)


def gaussian_cloud(draw, x, y, w, h, disks=False, count=55):
    for i in range(count):
        px = x + (i * 71) % w
        py = y + (i * 43) % h
        rx = 8 + (i * 11) % 28
        ry = 5 + (i * 7) % 20
        shade = 170 + (i * 17) % 65
        box = (px - rx, py - ry, px + rx, py + ry)
        draw.ellipse(box, fill=(shade, shade, shade, 130), outline=(125, 135, 145))
        if disks:
            draw.line((px - rx, py, px + rx, py), fill=(90, 105, 120), width=1)


def equation_box(draw, box, title, eq_lines):
    rounded(draw, box, radius=14, fill=(255, 255, 255), outline=TEAL, width=2)
    x1, y1, x2, y2 = box
    text(draw, (x1 + 24, y1 + 18), title, F_H2, TEAL_DARK)
    yy = y1 + 64
    max_width = x2 - x1 - 56
    for line in eq_lines:
        wrapped = wrap_text(draw, line, F_EQ, max_width)
        for part in wrapped:
            text(draw, (x1 + 28, yy), part, F_EQ, INK)
            yy += int(F_EQ.size * 1.28)
        yy += 4


def render_math_image(formula, fontsize=24, color=INK):
    rgb = tuple(value / 255.0 for value in color)
    fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
    fig.patch.set_alpha(0.0)
    fig.text(0, 0, formula, fontsize=fontsize, color=rgb)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def paste_fit(base, overlay, x, y, max_width=None, max_height=None):
    if max_width is not None and overlay.width > max_width:
        scale = max_width / overlay.width
        overlay = overlay.resize((max(1, int(overlay.width * scale)), max(1, int(overlay.height * scale))), Image.LANCZOS)
    if max_height is not None and overlay.height > max_height:
        scale = max_height / overlay.height
        overlay = overlay.resize((max(1, int(overlay.width * scale)), max(1, int(overlay.height * scale))), Image.LANCZOS)
    base.paste(overlay, (int(x), int(y)), overlay)
    return overlay.height


def math_equation_box(img, draw, box, title, formulas, fontsize=23, line_gap=10):
    rounded(draw, box, radius=14, fill=(255, 255, 255), outline=TEAL, width=2)
    x1, y1, x2, y2 = box
    text(draw, (x1 + 24, y1 + 18), title, F_H2, TEAL_DARK)
    yy = y1 + 62
    max_width = x2 - x1 - 56
    max_height = y2 - yy - 18
    remaining = max_height
    for formula in formulas:
        overlay = render_math_image(formula, fontsize=fontsize)
        used = paste_fit(img, overlay, x1 + 28, yy, max_width=max_width, max_height=max(24, remaining))
        yy += used + line_gap
        remaining = y2 - yy - 14


def save(img, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    img.save(path)
    return path


def slide1():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "Tracked Ultrasound Gaussian Reconstruction", "Current code pipeline: pose-aware slice rendering rather than camera-view 3DGS")
    d.line((355, 205, 355, 850), fill=LINE, width=2)
    text(d, (44, 220), "Technical ingredients", F_H1, TEAL_DARK)
    items = [
        "Tracked ultrasound frames provide physical slice poses.",
        "SVRTK-style scatter initializes Gaussians from occupied anatomy.",
        "Probe-plane renderer intersects splats with each ultrasound slice.",
        "Content-normalized edge loss reduces dark-frame bias.",
        "Debug, checkpoint, PLY/SOG export support iteration.",
    ]
    y = 305
    for i, item in enumerate(items, 1):
        badge(d, 60, y + 4, i, 20)
        paragraph(d, 96, y - 10, item, F_BODY, INK, 220, 1.2)
        y += 106

    steps = [
        ("Input slices\n+ poses", 430),
        ("SVRTK\ninit", 650),
        ("Disk / dot\nGaussians", 870),
        ("Probe-plane\nrender", 1095),
        ("Optimize\n+ export", 1320),
    ]
    for i, (label, x) in enumerate(steps, 1):
        badge(d, x, 226, i, 22)
        for line_idx, line in enumerate(label.split("\n")):
            text(d, (x + 2, 266 + line_idx * 23), line, F_SMALL, NAVY, anchor="mm")
        if i < len(steps):
            arrow(d, (x + 34, 226), (steps[i][1] - 34, 226), fill=(120, 120, 120), width=3)

    for j in range(4):
        ultrasound_panel(d, 430, 315 + j * 112, 105, 78, label=None)
        text(d, (548, 345 + j * 112), f"pose P{j+1}", F_SMALL, INK)
    text(d, (485, 760), "calibrated tracked sequence", F_SMALL, MUTED, anchor="mm")
    gaussian_cloud(d, 670, 330, 250, 320, disks=True)
    text(d, (795, 690), "initialized disk cloud", F_SMALL, MUTED, anchor="mm")
    d.polygon([(1005, 350), (1190, 305), (1190, 610), (1005, 660)], fill=(235, 239, 245), outline=(135, 145, 160))
    gaussian_cloud(d, 1008, 365, 180, 230, disks=True, count=28)
    ultrasound_panel(d, 1260, 350, 220, 165, "rendered slice")
    ultrasound_panel(d, 1260, 585, 220, 165, "target slice")
    arrow(d, (1370, 530), (1370, 578), fill=TEAL_DARK, width=4)
    text(d, (1400, 552), "loss", F_H2, INK)
    return save(img, "01_method_overview.png")


def slide2():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "SVRTK-Style Initialization", "A pose-aware scattered volume seeds Gaussians near scanned anatomy")
    text(d, (56, 220), "What the code does", F_H1, TEAL_DARK)
    steps = [
        ("1", "Sample pixels from each tracked slice."),
        ("2", "Map pixels into probe/world coordinates using calibration and pose."),
        ("3", "Scatter samples into a coarse voxel grid X0."),
        ("4", "Sample Gaussian centers from occupied/high-intensity voxels."),
    ]
    y = 290
    for n, s in steps:
        badge(d, 72, y + 10, n, 19)
        paragraph(d, 112, y - 8, s, F_BODY, INK, 390, 1.2)
        y += 82
    ultrasound_panel(d, 610, 252, 180, 130, "slice I_k")
    d.rectangle((895, 255, 1090, 385), fill=PALE_BLUE, outline=TEAL, width=2)
    text(d, (992, 320), "world points", F_H2, NAVY, anchor="mm")
    gaussian_cloud(d, 1180, 235, 270, 180, disks=False, count=35)
    arrow(d, (805, 320), (880, 320), fill=(120, 120, 120), width=4)
    arrow(d, (1105, 320), (1165, 320), fill=(120, 120, 120), width=4)
    math_equation_box(img, d, (540, 455, 1490, 850), "Equations used in init_SVRTK()", [
        r"$p_{\mathrm{img}}(u,v)=\left[(u-o_x)s,\;(v-o_y)s,\;0,\;1\right]^T$",
        r"$p_{\mathrm{world}}=T_{\mathrm{world}\leftarrow\mathrm{probe}}\,T_{\mathrm{probe}\leftarrow\mathrm{image}}\,p_{\mathrm{img}}$",
        r"$X_0(x)=\frac{\sum_k W_k(x)\,I_k(T_k^{-1}x)}{\sum_k W_k(x)}$",
        r"$\mu_i\sim p(x),\quad p(x)\propto X_0(x)\ \mathrm{for}\ X_0(x)\geq\tau$",
    ], fontsize=14, line_gap=3)
    rounded(d, (55, 665, 455, 810), radius=14, fill=PALE_TEAL, outline=TEAL, width=2)
    text(d, (78, 690), "Important limitation", F_H2, TEAL_DARK)
    paragraph(d, 78, 730, "This is not full SVRTK registration. It assumes poses.npy and image-to-probe calibration are already correct.", F_SMALL, INK, 340)
    return save(img, "02_svrtk_initialization.png")


def slide3():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "Current Primitive Model", "The code can render volume blobs, oriented disks, or dot-like splats")
    cols = [(70, 240, 490, 750), (590, 240, 1010, 750), (1110, 240, 1530, 750)]
    titles = ["Volume mode", "Disk mode", "Dot-like setting"]
    subs = [
        "3D anisotropic Gaussian ellipsoid",
        "learned normal + flattened covariance",
        "small isotropic scales, dense centers",
    ]
    for idx, box in enumerate(cols):
        rounded(d, box, radius=18, fill=(255, 255, 255), outline=TEAL if idx == 1 else LINE, width=3 if idx == 1 else 2)
        x1, y1, x2, y2 = box
        text(d, (x1 + 28, y1 + 26), titles[idx], F_H1, NAVY)
        paragraph(d, x1 + 28, y1 + 76, subs[idx], F_BODY, MUTED, 340)
    gaussian_cloud(d, 140, 410, 260, 180, disks=False, count=22)
    gaussian_cloud(d, 660, 410, 260, 180, disks=True, count=22)
    gaussian_cloud(d, 1180, 430, 260, 120, disks=False, count=80)
    math_equation_box(img, d, (100, 610, 460, 725), "Scale", [
        r"$\Sigma_i=\mathrm{diag}(s_x^2,\;s_y^2,\;s_z^2)$",
    ], fontsize=20)
    math_equation_box(img, d, (620, 610, 980, 725), "Disk covariance", [
        r"$\Sigma_i=R(n_i)\,D_i\,R(n_i)^T$",
        r"$D_i=\mathrm{diag}(s_u^2,\;s_v^2,\;s_n^2)$",
    ], fontsize=18, line_gap=4)
    math_equation_box(img, d, (1140, 610, 1500, 725), "Dot setting", [
        r"$s_x\approx s_y\approx s_z,\quad s_i\leq s_{\max}$",
    ], fontsize=20)
    rounded(d, (230, 800, 1370, 855), radius=12, fill=PALE_TEAL, outline=TEAL, width=2)
    text(d, (800, 828), "All modes share learnable means, log_scales, opacity logits, color/intensity, and optional disk normals.", F_H2, TEAL_DARK, anchor="mm")
    return save(img, "03_primitives.png")


def slide4():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "Probe-Plane Rendering", "No pinhole camera: each tracked frame is a physical ultrasound slice")
    d.polygon([(130, 300), (610, 220), (610, 650), (130, 730)], fill=(238, 242, 248), outline=(125, 135, 150), width=3)
    gaussian_cloud(d, 175, 325, 390, 260, disks=True, count=50)
    ultrasound_panel(d, 735, 285, 285, 215, "rendered Î")
    ultrasound_panel(d, 735, 585, 285, 215, "target I")
    arrow(d, (620, 475), (720, 402), fill=(120, 120, 120), width=4)
    arrow(d, (877, 515), (877, 575), fill=TEAL_DARK, width=4)
    text(d, (910, 545), "loss", F_H2, INK)
    math_equation_box(img, d, (1085, 245, 1535, 500), "Projection", [
        r"$p_i^{\mathrm{img}}=T_{\mathrm{image}\leftarrow\mathrm{probe}}\,P_k^{-1}\,\mu_i$",
        r"$u_i=x_i^{\mathrm{img}}/s+o_x,\quad v_i=y_i^{\mathrm{img}}/s+o_y$",
        r"$d_i=z_i^{\mathrm{img}}$",
    ], fontsize=18, line_gap=8)
    math_equation_box(img, d, (1085, 535, 1535, 785), "Splat alpha", [
        r"$w_i^{\mathrm{plane}}=\exp\!\left[-\frac{1}{2}\left(\frac{d_i}{\sigma_z}\right)^2\right]$",
        r"$G_i(u,v)=\exp\!\left[-\frac{1}{2}\left(\frac{u-u_i}{\sigma_x}\right)^2-\frac{1}{2}\left(\frac{v-v_i}{\sigma_y}\right)^2\right]$",
        r"$\alpha_i(u,v)=\mathrm{sigmoid}(o_i)\,w_i^{\mathrm{plane}}\,G_i(u,v)$",
    ], fontsize=16, line_gap=6)
    rounded(d, (90, 790, 590, 850), radius=12, fill=PALE_TEAL, outline=TEAL, width=2)
    text(d, (340, 820), "Reject non-intersecting splats early.", F_H2, TEAL_DARK, anchor="mm")
    return save(img, "04_probe_plane_rendering.png")


def slide5():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "Training Objective", "Ultrasound-specific loss focuses on stable structure instead of dark background")
    ultrasound_panel(d, 90, 270, 260, 190, "target")
    ultrasound_panel(d, 90, 555, 260, 190, "rendered")
    arrow(d, (370, 365), (505, 365), fill=(120, 120, 120), width=4)
    arrow(d, (370, 650), (505, 650), fill=(120, 120, 120), width=4)
    rounded(d, (525, 255, 775, 460), radius=16, fill=PALE_BLUE, outline=TEAL, width=2)
    rounded(d, (525, 540, 775, 745), radius=16, fill=PALE_BLUE, outline=TEAL, width=2)
    text(d, (650, 335), "Laplacian +\nSobel edges", F_H2, NAVY, anchor="mm")
    text(d, (650, 620), "Content\nweights", F_H2, NAVY, anchor="mm")
    math_equation_box(img, d, (850, 245, 1505, 500), "Feature loss", [
        r"$L_\sigma(I)=I-G_\sigma * I$",
        r"$E(I)=|L_\sigma(I)|+\lambda_s\,\|\nabla(G_\sigma * I)\|$",
        r"$L_{\mathrm{edge}}=\frac{\sum_x c(x)\,|E(\hat I)(x)-E(I)(x)|}{\sum_x c(x)+\epsilon}$",
    ], fontsize=17, line_gap=6)
    math_equation_box(img, d, (850, 545, 1505, 785), "Total objective", [
        r"$L=w_{\mathrm{lap}}L_{\mathrm{lap}}+w_{\mathrm{edge}}L_{\mathrm{edge}}+w_{\mathrm{int}}L_{\mathrm{int}}+w_{\mathrm{scale}}L_{\mathrm{scale}}$",
        r"$c(x)=\max(c_{\mathrm{bg}},\,I(x),\,E(I)(x))$",
    ], fontsize=17, line_gap=10)
    rounded(d, (88, 790, 770, 850), radius=12, fill=PALE_TEAL, outline=TEAL, width=2)
    text(d, (430, 820), "Why: mostly dark frames no longer look artificially easy.", F_H2, TEAL_DARK, anchor="mm")
    return save(img, "05_loss_and_optimization.png")


def slide6():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "How This Differs From Vanilla 3DGS", "The scanner geometry replaces camera projection")
    headers = ["Vanilla 3DGS", "Current ultrasound code", "Why it matters"]
    xs = [80, 570, 1060]
    for x, h in zip(xs, headers):
        text(d, (x, 235), h, F_H1, NAVY)
    rows = [
        ("Pinhole camera view matrix + intrinsics", "Tracked probe pose + image-to-probe calibration", "Ultrasound frame is a physical slice, not a camera image."),
        ("Projects 3D covariance through perspective Jacobian", "Intersects splat with calibrated probe plane", "Avoids fake optical perspective for cross-section data."),
        ("Color/opacity surface-like radiance", "Echo intensity / reflectivity-like splats", "Closer to reconstruction than novel RGB view synthesis."),
        ("Photometric RGB/image loss", "Laplacian/Sobel + content-normalized loss", "Reduces shadow/dark-frame dominance."),
        ("Often starts from SfM point cloud", "SVRTK-style scattered slice volume", "Uses tracked ultrasound samples directly."),
    ]
    y = 295
    for i, row in enumerate(rows):
        fill = (252, 253, 255) if i % 2 == 0 else (242, 247, 250)
        d.rectangle((58, y - 12, 1542, y + 78), fill=fill)
        for x, s in zip(xs, row):
            paragraph(d, x, y, s, F_SMALL, INK, 390, 1.16)
        y += 96
    d.line((520, 250, 520, 780), fill=LINE, width=2)
    d.line((1010, 250, 1010, 780), fill=LINE, width=2)
    rounded(d, (365, 812, 1235, 860), radius=12, fill=PALE_TEAL, outline=TEAL, width=2)
    text(d, (800, 836), "Core change: render a tracked acoustic slice through the field, not a camera projection of a scene.", F_H2, TEAL_DARK, anchor="mm")
    return save(img, "06_vs_vanilla_3dgs.png")


def slide7():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "Relationship To UltraGauss / UltraGS", "Similar direction, but this code remains a tracked-reconstruction prototype")
    d.line((800, 230, 800, 805), fill=LINE, width=3)
    text(d, (120, 235), "UltraGauss / UltraGS direction", F_H1, NAVY)
    text(d, (920, 235), "Current code", F_H1, NAVY)
    left = [
        "Optimized ultrasound-aware Gaussian/disc representation.",
        "Physically decoupled acoustic terms for attenuation, reflection, scattering.",
        "Rendering acceleration with custom CUDA/rasterization strategy.",
        "Often framed as novel-view synthesis from ultrasound observations.",
    ]
    right = [
        "Uses tracked probe poses and calibration from acquisition.",
        "SVRTK-style initialization creates rough voxel support before GS.",
        "Disk/dot modes are lightweight PyTorch approximations, not full CUDA UltraGS.",
        "Exports learned splat cloud as checkpoint, PLY, and experimental SOG path.",
    ]
    y = 315
    for i, item in enumerate(left, 1):
        badge(d, 120, y + 8, i, 18)
        paragraph(d, 160, y - 5, item, F_BODY, INK, 520, 1.18)
        y += 96
    y = 315
    for i, item in enumerate(right, 1):
        badge(d, 920, y + 8, i, 18)
        paragraph(d, 960, y - 5, item, F_BODY, INK, 520, 1.18)
        y += 96
    math_equation_box(img, d, (120, 720, 705, 825), "Ultra-style acoustic idea", [
        r"$I_{\mathrm{final}}=c+w_{\mathrm{att}}I_{\mathrm{att}}+w_{\mathrm{refl}}I_{\mathrm{refl}}+w_{\mathrm{scat}}I_{\mathrm{scat}}$",
    ], fontsize=16)
    math_equation_box(img, d, (920, 720, 1505, 825), "Our practical renderer", [
        r"$\hat I(x)=\sum_i \alpha_i(x)\,c_i\quad+\quad\mathrm{optional\ acoustic\ terms}$",
    ], fontsize=16)
    return save(img, "07_vs_ultragauss.png")


def main():
    paths = [slide1(), slide2(), slide3(), slide4(), slide5(), slide6(), slide7()]
    print("Generated slide images:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
