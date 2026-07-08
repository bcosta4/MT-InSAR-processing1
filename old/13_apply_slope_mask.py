
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
import numpy as np
from pathlib import Path

FOLDER = Path(r"E:/Processing/InSAR/results")

INSAR_FILES = [
    "Network2_ifg_unw_Stack_TC1_velocity_map_masked.tif",
    "Network2_ifg_unw_Stack_TC1_velocity_gddt_map_masked.tif",
    "Network2_ifg_unw_Stack_TC1_velocity_ddt_map_masked.tif",
    "Network2_ifg_unw_Stack_TC1_velocity_ddt_gddt_diff_masked.tif",
]

SLOPE_MASKS = {
    2: "Reclass_Slope2.tif",
    4: "Reclass_Slope4.tif",
    5: "Reclass_Slope5.tif",
}

def apply_slope_mask(insar_path, mask_path, output_path):
    with rasterio.open(insar_path) as insar_src:
        insar_data = insar_src.read(1)
        meta = insar_src.meta.copy()
        nodata = insar_src.nodata

        # If nodata not set in metadata, detect from common sentinel values
        if nodata is None:
            for candidate in [-9999.0, -9999, 0.0]:
                if np.any(insar_data == candidate):
                    nodata = candidate
                    print(f"  Auto-detected nodata = {nodata} in {insar_path.name}")
                    break
            if nodata is None:
                nodata = -9999.0
                print(f"  WARNING: Could not detect nodata, defaulting to -9999 for {insar_path.name}")

        with rasterio.open(mask_path) as mask_src:
            # Reproject mask to exactly match InSAR grid
            mask_aligned = np.zeros(insar_data.shape, dtype=np.uint8)
            reproject(
                source=rasterio.band(mask_src, 1),
                destination=mask_aligned,
                src_transform=mask_src.transform,
                src_crs=mask_src.crs,
                dst_transform=insar_src.transform,
                dst_crs=insar_src.crs,
                resampling=Resampling.nearest,
            )

        # Apply mask: set to nodata wherever slope mask == 2
        masked = insar_data.copy()
        cut = (mask_aligned == 2)
        masked[cut] = nodata

        n_cut = int(cut.sum())
        n_total = int(np.prod(insar_data.shape))
        pct = 100 * n_cut / n_total
        print(f"  Masked out {n_cut:,} / {n_total:,} pixels ({pct:.1f}%)")

        meta.update(dtype=masked.dtype, nodata=nodata)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(masked, 1)

print("=" * 60)
print("Slope masking — InSAR velocity rasters")
print("=" * 60)

for thresh, mask_filename in SLOPE_MASKS.items():
    mask_path = FOLDER / mask_filename
    if not mask_path.exists():
        print(f"\nERROR: Mask file not found: {mask_path}")
        continue

    print(f"\n--- Slope threshold: {thresh}° ---")
    for insar_filename in INSAR_FILES:
        insar_path = FOLDER / insar_filename
        if not insar_path.exists():
            print(f"  SKIP (not found): {insar_filename}")
            continue

        # Output name: insert _slope{thresh}deg before .tif
        stem = insar_path.stem
        out_name = f"{stem}_slope{thresh}deg.tif"
        output_path = FOLDER / out_name

        print(f"  {insar_filename}")
        apply_slope_mask(insar_path, mask_path, output_path)
        print(f"  → Saved: {out_name}")

print("\nDone.")
