"""
11_apply_mask.py
Final pipeline step. Applies a low-quality mask shapefile to all shifted
DS-InSAR output TIFs. Pixels whose centres fall INSIDE the shapefile polygons
are set to NODATA. Outputs get a _masked suffix in the same shifted/ folder.

Pipeline position: runs after 10_master_pipeline.py, operates on shifted/ TIFs.

Inputs:
  SHIFTED_DIR          : tc/shifted/  (geotransform-corrected TIFs)
  MASK_SHP             : mask_low_quality.shp  (low quality / unconnected areas)

Outputs (written to SHIFTED_DIR, same folder, _masked suffix):
  {stem}_masked.tif    for every matched TIF pattern
  11_masked_report.txt summary with before/after pixel counts and velocity stats
"""

import os
from pathlib import Path

# ── PROJ path fix (must be before rasterio/fiona import) ──────────────────────
_conda_prefix = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\anaconda3"))
_proj_data    = _conda_prefix / "Library" / "share" / "proj"
if _proj_data.exists():
    os.environ["PROJ_DATA"]    = str(_proj_data)
    os.environ["PROJ_LIB"]     = str(_proj_data)
    os.environ["PROJ_NETWORK"] = "OFF"

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from rasterio.crs import CRS
import fiona
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
WORK_DIR    = Path(r"C:\bcosta\DS-InSAR\tc")
SHIFTED_DIR = WORK_DIR / "shifted"
OUT_DIR     = SHIFTED_DIR
MASK_SHP    = Path(r"G:\Data\DS-InSAR\outputs\mask_low_quality.shp")
NODATA      = -9999.0
SHP_EPSG    = 3155   # EPSG of the shapefile — used for CRS repair on TIFs

TIF_PATTERNS = [
    # ── 05b inversion outputs ──────────────────────────────────────────────────
    "*_velocity_cmyr.tif",
    "*_velocity_rad_per_day.tif",
    "*_incidence_angle_ds_pixels.tif",
    "*_mean_coherence.tif",
    "*_coherence_mask.tif",
    # ── 07 export_geotiff outputs ──────────────────────────────────────────────
    "*_velocity_map.tif",
    "*_residuals_map.tif",
    "*_timeseries_stack.tif",
    # ── 09 DDT outputs ────────────────────────────────────────────────────────
    "*_velocity_ddt_map.tif",
    "*_velocity_gddt_map.tif",
    "*_velocity_ddt_gddt_diff.tif",
]

VELOCITY_TIFS = {
    "_velocity_cmyr.tif":       "cm/yr",
    "_velocity_map.tif":        "cm/yr",
    "_velocity_ddt_map.tif":    "cm/DDT",
    "_velocity_gddt_map.tif":   "cm/GDDT",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def is_engineering_crs(crs_wkt):
    return "EngineeringCRS" in crs_wkt or "Unknown engineering datum" in crs_wkt


def repair_tif_crs(tif_path, epsg=SHP_EPSG):
    """Overwrite EngineeringCRS metadata in-place with correct EPSG."""
    correct_crs = CRS.from_epsg(epsg)
    with rasterio.open(tif_path, "r+") as src:
        if is_engineering_crs(src.crs.to_wkt()):
            src.crs = correct_crs
            print(f"    [crs-repair] {tif_path.name} -> EPSG:{epsg}")


def load_mask_geometries(shp_path, target_crs_wkt):
    with fiona.open(shp_path) as shp:
        shp_crs_wkt = shp.crs_wkt if hasattr(shp, "crs_wkt") else None
        if shp_crs_wkt is None:
            shp_crs_wkt = CRS.from_user_input(shp.crs).to_wkt()
        raw_geoms = [feat["geometry"] for feat in shp
                     if feat["geometry"] is not None]

    shp_crs    = CRS.from_wkt(shp_crs_wkt)
    target_crs = CRS.from_wkt(target_crs_wkt)
    shp_epsg    = shp_crs.to_epsg()
    target_epsg = target_crs.to_epsg()

    if is_engineering_crs(target_crs_wkt):
        # TIF has no geodetic datum — coordinates are coincident with shapefile,
        # skip reprojection entirely
        print(f"    [mask] TIF has EngineeringCRS — coordinates assumed "
              f"coincident with EPSG:{shp_epsg}, skipping reprojection")
        geoms = [shape(g) for g in raw_geoms]

    elif shp_epsg and target_epsg and shp_epsg == target_epsg:
        print(f"    [mask] CRS match (EPSG:{shp_epsg}) — no reprojection needed")
        geoms = [shape(g) for g in raw_geoms]

    else:
        print(f"    [mask] Reprojecting {len(raw_geoms)} polygon(s) "
              f"EPSG:{shp_epsg} -> EPSG:{target_epsg} via rasterio.warp ...")
        geoms = [
            shape(transform_geom(shp_crs_wkt, target_crs_wkt, g))
            for g in raw_geoms
        ]
        print(f"    [mask] Reprojection done")

    dissolved = unary_union(geoms)
    print(f"    [mask] {len(raw_geoms)} polygon(s) dissolved into 1 geometry")
    return [dissolved]


def apply_mask(src_path, dst_path, mask_geoms):
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else NODATA
        meta       = src.meta.copy()
        original   = src.read(1).astype(np.float32)

        burned, _ = rio_mask(
            src,
            [mapping(g) for g in mask_geoms],
            invert=True,       # keep OUTSIDE, nodata INSIDE mask polygons
            nodata=NODATA,
            filled=True,
            crop=False
        )

    meta.update(dtype="float32", nodata=NODATA)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(burned.astype(np.float32))

    before = int(np.sum((original != src_nodata) & np.isfinite(original)))
    after  = int(np.sum((burned[0] != NODATA)    & np.isfinite(burned[0])))
    return before, before - after, after


def velocity_stats(tif_path, units):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
    valid = data[(data != NODATA) & np.isfinite(data)]
    if len(valid) == 0:
        return None
    return {
        "units":  units,
        "n":      len(valid),
        "mean":   float(np.mean(valid)),
        "std":    float(np.std(valid)),
        "median": float(np.median(valid)),
        "min":    float(np.min(valid)),
        "max":    float(np.max(valid)),
        "n_sub":  int(np.sum(valid < 0)),
        "n_upl":  int(np.sum(valid > 0)),
    }


def format_vel_stats(s):
    n = s["n"]
    return [
        f"  Units          : {s['units']}\n",
        f"  Valid pixels   : {n:,}\n",
        f"  Mean           : {s['mean']:+.4f} {s['units']}\n",
        f"  Median         : {s['median']:+.4f} {s['units']}\n",
        f"  Std            : {s['std']:.4f} {s['units']}\n",
        f"  Range          : {s['min']:.4f} to {s['max']:.4f} {s['units']}\n",
        f"  Subsiding      : {s['n_sub']:,} ({s['n_sub']/max(n,1)*100:.1f}%)\n",
        f"  Uplifting      : {s['n_upl']:,} ({s['n_upl']/max(n,1)*100:.1f}%)\n",
    ]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    tifs = []
    for pattern in TIF_PATTERNS:
        for f in SHIFTED_DIR.glob(pattern):
            if "_masked" not in f.stem:
                tifs.append(f)
    tifs = sorted(set(tifs))

    if not tifs:
        print(f"No TIFs found in {SHIFTED_DIR} matching patterns.")
        print("Run shift_geotransform.py first.")
        return

    print("11_apply_mask.py")
    print(f"Mask SHP  : {MASK_SHP}")
    print(f"Input dir : {SHIFTED_DIR}")
    print(f"Output    : same folder, _masked suffix")
    print(f"Found {len(tifs)} TIF(s) to process")
    print("=" * 60)

    # Repair EngineeringCRS on all input TIFs before loading mask
    print("Checking/repairing TIF CRS metadata ...")
    for tif in tifs:
        repair_tif_crs(tif, epsg=SHP_EPSG)

    # Load mask geometries using (now repaired) first TIF CRS
    with rasterio.open(tifs[0]) as src:
        tif_crs_wkt = src.crs.to_wkt()
    mask_geoms = load_mask_geometries(MASK_SHP, tif_crs_wkt)
    print()

    report = [
        f"11_apply_mask report\n",
        f"Timestamp  : {datetime.now().isoformat(timespec='seconds')}\n",
        f"Mask SHP   : {MASK_SHP}\n",
        f"Input dir  : {SHIFTED_DIR}\n",
        "=" * 60 + "\n\n",
    ]

    for tif in tifs:
        dst_name = tif.stem + "_masked.tif"
        dst      = OUT_DIR / dst_name

        before, n_masked, after = apply_mask(tif, dst, mask_geoms)
        pct = n_masked / max(before, 1) * 100

        print(f"  {tif.name}")
        print(f"    Before: {before:,}  |  Masked: {n_masked:,} ({pct:.1f}%)  |  After: {after:,}")
        print(f"    -> {dst_name}")

        report += [
            f"{tif.name}\n",
            f"  Before valid : {before:,}\n",
            f"  Masked out   : {n_masked:,} ({pct:.1f}%)\n",
            f"  After valid  : {after:,}\n",
            f"  Output       : {dst_name}\n",
        ]

        for suffix, units in VELOCITY_TIFS.items():
            if tif.name.endswith(suffix):
                stats = velocity_stats(dst, units)
                if stats:
                    report += format_vel_stats(stats)
                    print(f"    mean={stats['mean']:+.4f} {units}  "
                          f"std={stats['std']:.4f}  "
                          f"sub={stats['n_sub']/max(stats['n'],1)*100:.1f}%  "
                          f"upl={stats['n_upl']/max(stats['n'],1)*100:.1f}%")
                break

        report.append("\n")

    out_rep = OUT_DIR / "11_masked_report.txt"
    Path(out_rep).write_text("".join(report), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Done. Masked TIFs written to: {OUT_DIR}")
    print(f"Report: {out_rep.name}")


if __name__ == "__main__":
    main()