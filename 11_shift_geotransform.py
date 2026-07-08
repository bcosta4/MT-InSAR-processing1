"""
11_shift_geotransform.py
Shifts the geotransform origin of all DS-InSAR output TIFs eastward (positive X)
by a fixed number of pixels. The pixel data is NOT moved -- only the coordinate
origin in the GeoTIFF metadata is updated.

Positive X = east, Positive Y = north, Negative Y = south.
Outputs written to WORK_DIR/shifted/ with the same filenames.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine


# ── Configuration ──────────────────────────────────────────────────────────────
WORK_DIR = Path(r"E:\Processing\InSAR\inputs")
OUT_DIR  = WORK_DIR / "shifted"

SHIFT_PIXELS_X =  3   # positive = east
SHIFT_PIXELS_Y =  0   # positive = north (use negative for south)

TIF_PATTERNS = [
    "*_velocity_cmyr.tif",
    "*_velocity_rad_per_day.tif",
    "*_incidence_angle_ds_pixels.tif",
    "*_coherence_mask.tif",
    "*_mean_coherence.tif",
    "*_velocity_map.tif",
    "*_velocity_ddt_map.tif",
    "*_velocity_gddt_map.tif",
    "*_velocity_ddt_gddt_diff.tif",
    "*_residuals_map.tif",
    "*_timeseries_stack.tif",
]


# ── Main ───────────────────────────────────────────────────────────────────────
def shift_geotransform(src_path, dst_path, shift_x_px, shift_y_px):
    with rasterio.open(src_path) as src:
        meta    = src.meta.copy()
        t       = src.transform
        data    = src.read()
        tags    = src.tags()
        # FIX: capture indexes and band tags while src is still open --
        # accessing src.indexes after the with-block closes raises ValueError
        indexes = list(src.indexes)
        btags   = {i: src.tags(i) for i in indexes}

    pixel_w =  t.a   # pixel width  (positive, metres)
    pixel_h =  t.e   # pixel height (negative, metres)

    shift_east  = shift_x_px * pixel_w
    shift_north = shift_y_px * pixel_h

    new_transform = Affine(
        t.a, t.b, t.c + shift_east,
        t.d, t.e, t.f + shift_north,
    )

    meta["transform"] = new_transform
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(data)
        dst.update_tags(**tags)
        # FIX: iterate the captured list, not the closed src object
        for i in indexes:
            if btags.get(i):
                dst.update_tags(i, **btags[i])

    print(f"  Shifted: {src_path.name}  "
          f"x_origin {t.c:.2f} -> {new_transform.c:.2f}  "
          f"(+{shift_east:.1f} m east)")


def main():
    tifs = []
    for pattern in TIF_PATTERNS:
        tifs.extend(WORK_DIR.glob(pattern))
    tifs = sorted(set(tifs))

    if not tifs:
        print(f"No TIFs found in {WORK_DIR} matching patterns.")
        return

    print("11_shift_geotransform.py")
    print(f"Shift: X={SHIFT_PIXELS_X} px, Y={SHIFT_PIXELS_Y} px")
    print(f"Found {len(tifs)} TIF(s) to process -> {OUT_DIR.name}/")
    print("=" * 60)

    for tif in tifs:
        dst = OUT_DIR / tif.name
        shift_geotransform(tif, dst, SHIFT_PIXELS_X, SHIFT_PIXELS_Y)

    print("=" * 60)
    print(f"Done. Shifted TIFs written to: {OUT_DIR}")
    print("Load from the 'shifted/' folder in QGIS to verify alignment.")
    print("If correct, update WORK_DIR in downstream scripts (12, 13).")


if __name__ == "__main__":
    main()
