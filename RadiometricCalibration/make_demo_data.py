"""Generate synthetic demo maps of a rover-like scene into demo_data/.

Distance map: ground-plane ramp from ~0.5 m at the image bottom (the ground
just in front of the rover) to ~20 m at the top, with a few NaN holes to
mimic LiDAR dropouts. Apparent-temperature map: uniform 30 deg C surface
with a warm 40 deg C patch, so the effect of the per-pixel correction
(stronger for the distant pixels at the top) is clearly visible.

Usage:
    py make_demo_data.py
"""

from pathlib import Path

import numpy as np

HEIGHT, WIDTH = 240, 320
NEAR_M, FAR_M = 0.5, 20.0

OUT_DIR = Path(__file__).resolve().parent / "demo_data"


def main():
    rng = np.random.default_rng(seed=0)

    # Ground-plane-like distance ramp: near at the bottom row, far at the top.
    # Quadratic ramp so distance grows faster towards the "horizon".
    t = np.linspace(1.0, 0.0, HEIGHT)[:, None]  # 1 at top row, 0 at bottom row
    distance = NEAR_M + (FAR_M - NEAR_M) * t**2
    distance = np.broadcast_to(distance, (HEIGHT, WIDTH)).copy()

    # A few LiDAR dropouts (NaN holes).
    holes = rng.random((HEIGHT, WIDTH)) < 0.002
    distance[holes] = np.nan

    # Uniform 30 deg C scene with a warm 40 deg C circular patch in the middle.
    apparent = np.full((HEIGHT, WIDTH), 30.0)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    patch = (yy - HEIGHT / 2) ** 2 + (xx - WIDTH / 2) ** 2 < 40**2
    apparent[patch] = 40.0
    apparent += rng.normal(0.0, 0.1, (HEIGHT, WIDTH))  # sensor noise

    OUT_DIR.mkdir(exist_ok=True)
    np.save(OUT_DIR / "distance.npy", distance)
    np.save(OUT_DIR / "apparent.npy", apparent)
    print(f"Wrote {OUT_DIR / 'distance.npy'} ({NEAR_M}-{FAR_M} m ramp, {holes.sum()} NaN holes)")
    print(f"Wrote {OUT_DIR / 'apparent.npy'} (30 deg C + 40 deg C patch)")


if __name__ == "__main__":
    main()
