"""Engineered visible-pixel encoding and spatial blade interface; no game state."""

import numpy as np
from scipy.ndimage import gaussian_filter

GRID_W, GRID_H = 12, 8


def encode(rgb):
    from PIL import Image

    x = (
        np.asarray(
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize(
                (96, 64), Image.Resampling.BOX
            ),
            dtype=np.float32,
        )
        / 255
    )
    mx = x.max(-1)
    mn = x.min(-1)
    # Palette-specific chromatic foreground and bright achromatic foreground.
    # This is engineered color segmentation, not fly phototransduction or learned vision.
    fruit = np.clip((mx - mn - 0.15) * 2, 0, 1)
    danger = np.clip((mn - 0.25) * 2, 0, 1) * (1 - np.clip((mx - mn) * 5, 0, 1))
    h, w = fruit.shape
    return (
        np.stack(
            [
                fruit.reshape(GRID_H, h // GRID_H, GRID_W, w // GRID_W).mean((1, 3)),
                danger.reshape(GRID_H, h // GRID_H, GRID_W, w // GRID_W).mean((1, 3)),
            ]
        )
        .ravel()
        .astype(np.float32)
    )


def motor(field, blade=(320, 430)):
    f = np.clip(np.asarray(field).reshape(2, GRID_H, GRID_W), 0, 1)
    fruit, danger = f
    # Select one reconstructed visual peak, then a sub-cell weighted centroid.
    # Spatial selection is engineered; it cannot access the observation directly.
    threat = gaussian_filter(danger, 0.6)
    utility = fruit - 2.8 * threat
    iy, ix = np.unravel_index(np.argmax(utility), utility.shape)
    if utility[iy, ix] < 0.035:
        return float(blade[0]), float(blade[1]), False
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
    local = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / 0.6) * np.maximum(utility, 0)
    s = max(local.sum(), 1e-8)
    tx = float(((xx + 0.5) * local).sum() / s * 640 / GRID_W)
    ty = float(((yy + 0.5) * local).sum() / s * 480 / GRID_H)
    # Suppress cutting if the straight blade path crosses perceived hazard.
    bx, by = blade
    xs = np.linspace(bx, tx, 12)
    ys = np.linspace(by, ty, 12)
    hx = np.clip((xs / 640 * GRID_W).astype(int), 0, GRID_W - 1)
    hy = np.clip((ys / 480 * GRID_H).astype(int), 0, GRID_H - 1)
    safe = bool(np.max(threat[hy, hx]) < 0.045)
    return tx, ty, safe


def sample_scene(rng):
    from PIL import Image, ImageDraw
    from .world import W, H, FRUIT_COLORS

    im = Image.new("RGB", (W, H), (7, 11, 24))
    d = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(0, 7))):
        x = float(rng.uniform(30, W - 30))
        y = float(rng.uniform(10, H + 20))
        r = float(rng.uniform(22, 29))
        bomb = rng.random() < 0.25
        d.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(174, 181, 197) if bomb else FRUIT_COLORS[int(rng.integers(4))],
        )
    return encode(np.asarray(im))
