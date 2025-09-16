from skimage.io import imread
import numpy as np

# Read the image (grayscale or RGB)
img = imread("campus.tiff")   # replace with your image filename

# Report information
height, width = img.shape[0], img.shape[1]

# Bits per pixel:
bits_per_pixel = img.dtype.itemsize * 8   # itemsize is in bytes, so multiply by 8

print("Width:", width)
print("Height:", height)
print("Bits per pixel:", bits_per_pixel)

# Convert to double precision (float64)
img_double = img.astype(np.float64)
print("Converted dtype:", img_double.dtype)

# Assuming `image` is your 2D NumPy array of pixel values
black = 150
white = 4095

# Shift and scale
linear = (img_double - black) / (white - black)

# Clip to [0, 1]
linear = np.clip(linear, 0, 1)

print(linear.shape,"Linear shape here")

from scipy.interpolate import interp2d

def demosaic_interp2d(cfa, pattern):

    H, W = cfa.shape
    yy = np.arange(H, dtype=float)
    xx = np.arange(W, dtype=float)

    # For each pattern, define where R,G,B live inside the 2x2 tile: (y0, x0)
    # (0-based, y is row, x is col)
    if pattern == 'rggb':
        offs = {'R': (0,0), 'G1': (0,1), 'G2': (1,0), 'B': (1,1)}
    elif pattern == 'grbg':
        offs = {'G1': (0,0), 'R': (0,1), 'B': (1,0), 'G2': (1,1)}
    elif pattern == 'bggr':
        offs = {'B': (0,0), 'G1': (0,1), 'G2': (1,0), 'R': (1,1)}
    elif pattern == 'gbrg':
        offs = {'G1': (0,0), 'B': (0,1), 'R': (1,0), 'G2': (1,1)}
    else:
        raise ValueError("Unknown Bayer pattern")

    def upsample_from_subgrid(y0, x0):

        # subgrid axes & values
        sub_y = np.arange(y0, H, 2, dtype=float)
        sub_x = np.arange(x0, W, 2, dtype=float)
        Z = cfa[y0::2, x0::2].astype(float)  # shape (len(sub_y), len(sub_x))

        # If subgrid is 1xN or Nx1, fall back to nearest to avoid Fitpack issues
        if Z.shape[0] < 2 or Z.shape[1] < 2:
            # nearest-neighbor expansion
            out = np.zeros((H, W), dtype=float)
            out[y0::2, x0::2] = Z
            # fill missing by nearest parity copy
            out = np.where(out==0, np.pad(out[::2,::2], ((y0, H%2==0), (x0, W%2==0)), mode='edge')[:H,:W], out)
            return out

        f = interp2d(sub_x, sub_y, Z, kind='linear')  # regular grid!
        return f(xx, yy)  # full HxW

    # Build each plane from its subgrid
    R = upsample_from_subgrid(*offs.get('R', (0,0)))
    G1 = upsample_from_subgrid(*offs.get('G1', (0,1)))
    G2 = upsample_from_subgrid(*offs.get('G2', (1,0)))
    B = upsample_from_subgrid(*offs.get('B', (1,1)))

    # Merge the two green estimates (simple average)
    G = 0.5 * (G1 + G2)

    rgb = np.stack([R, G, B], axis=-1).astype(np.float32)
    return np.clip(rgb, 0, 1)

def white_world(rgb):
    # Scale each channel so its max becomes 1
    maxes = np.maximum(rgb.reshape(-1,3).max(axis=0), 1e-6)
    gains = 1.0 / maxes
    return np.clip(rgb * gains, 0, 1), gains

def gray_world(rgb):
    # Scale channels so their means match the global mean
    means = np.maximum(rgb.reshape(-1,3).mean(axis=0), 1e-6)
    target = means.mean()
    gains = target / means
    return np.clip(rgb * gains, 0, 1), gains

def preset_wb(rgb, r_gain, g_gain, b_gain):
    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
    gmax = np.max(gains)
    # Normalize so overall scale stays in [0,1] as much as possible
    gains = gains / gmax if gmax > 0 else gains
    return np.clip(rgb * gains, 0, 1), gains

r_gain, g_gain, b_gain = 2.394531, 1.000000, 1.597656


patterns = ['grbg','rggb','bggr','gbrg']

# Manually chosen rggb for best demosiacing
pattern = 'rggb'
rgb = demosaic_interp2d(linear, pattern)

ww_img, ww_gains = white_world(rgb)
gw_img, gw_gains = gray_world(rgb)
pr_img, pr_gains = preset_wb(rgb, r_gain, g_gain, b_gain)



import matplotlib.pyplot as plt
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
plt.imshow(np.clip(ww_img,0,1))
plt.title("White-World WB")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(np.clip(gw_img,0,1))
plt.title("Gray-World WB")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(np.clip(pr_img,0,1))
plt.title("Preset WB")
plt.axis("off")

plt.tight_layout()
plt.show()

MXYZ_to_cam_raw = np.array([
    6988, -1384, -714,
    -5631, 13410, 2447,
    -1485, 2204, 7318
], dtype=np.float64) / 10000.0

MXYZ_to_cam = MXYZ_to_cam_raw.reshape(3, 3)

MsRGB_to_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

MsRGB_to_cam = MXYZ_to_cam @ MsRGB_to_XYZ
row_sums = MsRGB_to_cam.sum(axis=1, keepdims=True)
MsRGB_to_cam_norm = MsRGB_to_cam / row_sums

Mcam_to_sRGB = np.linalg.inv(MsRGB_to_cam_norm)

def apply_cam_to_srgb(rgb_cam, M):
    h, w, _ = rgb_cam.shape
    flat = rgb_cam.reshape(-1, 3)
    out = flat @ M.T          # multiply row vectors by matrix
    return np.clip(out.reshape(h, w, 3), 0, 1)

ww_img = apply_cam_to_srgb(ww_img, Mcam_to_sRGB)
gw_img = apply_cam_to_srgb(gw_img, Mcam_to_sRGB)
pr_img = apply_cam_to_srgb(pr_img, Mcam_to_sRGB)

plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
plt.imshow(np.clip(ww_img,0,1))
plt.title("White-World WB")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(np.clip(gw_img,0,1))
plt.title("Gray-World WB")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(np.clip(pr_img,0,1))
plt.title("Preset WB")
plt.axis("off")

plt.tight_layout()
plt.show()

from skimage.color import rgb2gray

def brighten_to_mean(rgb_linear_srgb, target_mean=0.25):

    # current mean grayscale (in linear light)
    gray = rgb2gray(rgb_linear_srgb)              # linear grayscale
    current = float(gray.mean())

    # avoid divide-by-zero; if all black, just return as-is
    if current <= 1e-12:
        scaled = rgb_linear_srgb.copy()
    else:
        scale = target_mean / current
        scaled = np.clip(rgb_linear_srgb * scale, 0.0, 1.0)

    return scaled

# Doing manual white balancing after color correction
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _patch_means(img, y0, x0, h, w):
    H, W, _ = img.shape
    y1 = np.clip(y0, 0, H-1); x1 = np.clip(x0, 0, W-1)
    y2 = np.clip(y0+h, 0, H); x2 = np.clip(x0+w, 0, W)
    patch = img[y1:y2, x1:x2, :]
    if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return patch.reshape(-1,3).mean(axis=0)

def _wb_from_means(rgb_linear, means):
    means = np.maximum(means, 1e-8)
    target = means.mean()
    gains = target / means
    gains = gains / gains.max()          # keep exposure similar
    out = np.clip(rgb_linear * gains[None,None,:], 0, 1).astype(np.float32)
    return out, gains

def manual_wb_two_points(rgb_linear):
    """
    Click TOP-LEFT and BOTTOM-RIGHT of a white/neutral patch.
    Returns (wb_image, gains). Operates on LINEAR sRGB.
    """
    # gamma preview ONLY for the click UI
    a = 0.055
    disp = np.clip(rgb_linear, 0, 1)
    disp = np.where(disp <= 0.0031308, 12.92*disp, (1+a)*np.power(disp,1/2.4)-a)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(disp); ax.set_title("Click top-left, then bottom-right of a neutral patch")
    ax.axis("off")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(pts) < 2:
        print("Not enough clicks; keeping original.")
        return rgb_linear, np.array([1.,1.,1.], dtype=np.float32)

    (x1,y1),(x2,y2) = pts
    x1,y1 ,x2 ,y2 = int(x1), int(y1), int(x2), int(y2)
    
    print(x1, y1, x2, y2, "points here")
    x0, y0 = int(min(x1,x2)), int(min(y1,y2))
    w,  h  = int(abs(x2-x1)), int(abs(y2-y1))

    print(x0, y0, w, h ,"boxes here")
    means = _patch_means(rgb_linear, y0, x0, h, w)
    wb_img, gains = _wb_from_means(rgb_linear, means)
    # quick preview of the selected box on the gamma view (optional)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(disp); ax.add_patch(Rectangle((x0,y0), w, h, fill=False, linewidth=2))
    ax.set_title(f"Patch means = {means}, gains = {gains}"); ax.axis("off"); plt.show()
    return wb_img, gains

# Uncomment for manual white balancing
# --------------------------------
# gw_img, manual_gains = manual_wb_two_points(gw_img)
# --------------------------------

candidates = [0.18, 0.22, 0.25, 0.30]
brightened_versions = {t: brighten_to_mean(gw_img, t) for t in candidates}
# Chose 0.25 for best result
rgb_bright = brightened_versions[0.25]

# sRGB gamma (tone reproduction)
def linear_to_srgb_gamma(x):

    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    low  = x <= 0.0031308
    high = ~low
    y = np.empty_like(x)
    y[low]  = 12.92 * x[low]
    y[high] = (1 + a) * np.power(x[high], 1/2.4) - a
    return y

rgb_display = linear_to_srgb_gamma(rgb_bright)

plt.figure(figsize=(8, 6))
plt.imshow(np.clip(rgb_display, 0, 1))
plt.title("Final image (brightened + sRGB gamma)")
plt.axis("off")
plt.show()

from skimage.io import imsave
import os

# Convert to 8-bit first (PNG/JPEG expect uint8)
img8 = (np.clip(rgb_display, 0, 1) * 255 + 0.5).astype(np.uint8)

imsave("result.png", img8)                # lossless (no compression artifacts)
# imsave("result_q95.jpg", img8, quality=95)   # JPEG with quality=95