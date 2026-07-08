"""
12_apply_quality_slope_mask.py
Applies two masks sequentially to all shifted DS-InSAR TIFs:
  1. Low-quality shapefile mask (pixels inside polygons -> NODATA)
  2. Slope mask (pixels where slope raster == 2 -> NODATA), applied at
     three thresholds (2 deg, 4 deg, 5 deg) to the four main velocity rasters.

Pipeline position: runs after 11_shift_geotransform.py (shifted/ TIFs exist).

Inputs:
  SHIFTED_DIR : inputs/shifted/  -- geotransform-corrected TIFs from script 11
  MASK_SHP    : mask_low_quality.shp
  SLOPE_MASKS : Reclass_Slope{2,4,5}.tif  in RESULTS_DIR

Outputs (written to SHIFTED_DIR):
  {stem}_masked.tif               -- quality-masked TIFs (all patterns)
  12_masked_report.txt            -- per-TIF pixel counts + velocity stats

Outputs (written to RESULTS_DIR):
  {stem}_slope{N}deg.tif         -- slope-masked velocity TIFs
"""

import os
import json
import importlib.util
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
from rasterio.warp import reproject, transform_geom, Resampling
from rasterio.crs import CRS
from datetime import datetime

_fiona_spec = importlib.util.find_spec("fiona")
fiona = importlib.import_module("fiona") if _fiona_spec else None
_ogr_spec = importlib.util.find_spec("osgeo.ogr")
ogr = importlib.import_module("osgeo.ogr") if _ogr_spec else None
osr = importlib.import_module("osgeo.osr") if _ogr_spec else None


# ==============================================================================
# Configuration
# ==============================================================================

WORK_DIR    = Path(r"E:\Processing\InSAR\inputs")
SHIFTED_DIR = WORK_DIR / "shifted"
OUT_DIR     = SHIFTED_DIR
MASK_SHP    = Path(r"G:\Data\DS-InSAR\outputs\mask_low_quality.shp")
NODATA      = -9999.0
SHP_EPSG    = 3155

RESULTS_DIR = Path(r"E:/Processing/InSAR/results")

TIF_PATTERNS = [
    "*_velocity_cmyr.tif",
    "*_velocity_rad_per_day.tif",
    "*_incidence_angle_ds_pixels.tif",
    "*_mean_coherence.tif",
    "*_coherence_mask.tif",
    "*_velocity_map.tif",
    "*_residuals_map.tif",
    "*_timeseries_stack.tif",
    "*_velocity_ddt_map.tif",
    "*_velocity_gddt_map.tif",
    "*_velocity_ddt_gddt_diff.tif",
]

VELOCITY_TIFS = {
    "_velocity_cmyr.tif":     "cm/yr",
    "_velocity_map.tif":      "cm/yr",
    "_velocity_ddt_map.tif":  "cm/DDT",
    "_velocity_gddt_map.tif": "cm/GDDT",
}

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


# ==============================================================================
# Part 1 -- Quality mask helpers
# ==============================================================================

def is_engineering_crs(crs_wkt):
    return "EngineeringCRS" in crs_wkt or "Unknown engineering datum" in crs_wkt


def repair_tif_crs(tif_path, epsg=SHP_EPSG):
    correct_crs = CRS.from_epsg(epsg)
    try:
        with rasterio.open(tif_path, "r+") as src:
            if is_engineering_crs(src.crs.to_wkt()):
                src.crs = correct_crs
                print(f"  [crs-repair] {tif_path.name} -> EPSG:{epsg}")
    except Exception as e:
        print(f"  [crs-repair] WARNING: Could not repair {tif_path.name}: {e}")


def load_mask_geometries(shp_path, target_crs_wkt):
    """
    Load shapefile geometries and reproject if needed.
    Returns a list of GeoJSON-like dicts compatible with rasterio.mask.mask.
    Dissolves all polygons into a single geometry to avoid redundant masking.
    """
    if fiona is not None:
        with fiona.open(shp_path) as shp:
            shp_crs_wkt = shp.crs_wkt if hasattr(shp, "crs_wkt") else None
            if shp_crs_wkt is None:
                shp_crs_wkt = CRS.from_user_input(shp.crs).to_wkt()
            # FIX: convert fiona geometry proxies to plain dicts for downstream use
            raw_geoms = [dict(feat["geometry"]) for feat in shp
                         if feat["geometry"] is not None]
    elif ogr is not None:
        driver = ogr.GetDriverByName("ESRI Shapefile")
        ds = driver.Open(str(shp_path))
        if ds is None:
            raise RuntimeError(f"Unable to open shapefile: {shp_path}")
        layer = ds.GetLayer()
        srs = layer.GetSpatialRef()
        shp_crs_wkt = srs.ExportToWkt() if srs is not None else None
        raw_geoms = []
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            raw_geoms.append(json.loads(geom.ExportToJson()))
        ds.Destroy()
    else:
        raise ImportError(
            "Neither fiona nor osgeo.ogr is available; cannot read shapefile."
        )

    if shp_crs_wkt is None:
        raise RuntimeError(f"Could not determine CRS for shapefile: {shp_path}")

    shp_crs     = CRS.from_wkt(shp_crs_wkt)
    target_crs  = CRS.from_wkt(target_crs_wkt)
    shp_epsg    = shp_crs.to_epsg()
    target_epsg = target_crs.to_epsg()

    if is_engineering_crs(target_crs_wkt):
        print(f"  [mask] TIF has EngineeringCRS -- coordinates assumed "
              f"coincident with EPSG:{shp_epsg}, skipping reprojection")
        geoms = raw_geoms
    elif shp_epsg and target_epsg and shp_epsg == target_epsg:
        print(f"  [mask] CRS match (EPSG:{shp_epsg}) -- no reprojection needed")
        geoms = raw_geoms
    else:
        print(f"  [mask] Reprojecting {len(raw_geoms)} polygon(s) "
              f"EPSG:{shp_epsg} -> EPSG:{target_epsg} ...")
        geoms = [transform_geom(shp_crs_wkt, target_crs_wkt, g) for g in raw_geoms]
        print(f"  [mask] Reprojection done")

    # FIX: dissolve into a single geometry so overlapping polygons don't double-mask.
    # Falls back to individual geometries if shapely is unavailable.
    try:
        from shapely.geometry import shape, mapping
        from shapely.ops import unary_union
        dissolved = unary_union([shape(g) for g in geoms])
        geoms = [mapping(dissolved)]
        print(f"  [mask] {len(raw_geoms)} polygon(s) dissolved into 1 geometry")
    except ImportError:
        print(f"  [mask] shapely not available -- using {len(geoms)} polygon(s) without dissolve")

    return geoms


def apply_quality_mask(src_path, dst_path, mask_geoms):
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else NODATA
        meta       = src.meta.copy()
        original   = src.read(1).astype(np.float32)
        burned, _  = rio_mask(
            src,
            mask_geoms,
            invert=True,
            nodata=NODATA,
            filled=True,
            crop=False,
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
        f"  Units        : {s['units']}\n",
        f"  Valid pixels : {n:,}\n",
        f"  Mean         : {s['mean']:+.4f} {s['units']}\n",
        f"  Median       : {s['median']:+.4f} {s['units']}\n",
        f"  Std          : {s['std']:.4f} {s['units']}\n",
        f"  Range        : {s['min']:.4f} to {s['max']:.4f} {s['units']}\n",
        f"  Subsiding    : {s['n_sub']:,} ({s['n_sub']/max(n,1)*100:.1f}%)\n",
        f"  Uplifting    : {s['n_upl']:,} ({s['n_upl']/max(n,1)*100:.1f}%)\n",
    ]


# ==============================================================================
# Part 2 -- Slope mask helpers
# ==============================================================================

def apply_slope_mask(insar_path, mask_path, output_path):
    # FIX: capture all insar_src attributes while the file is still open.
    # insar_src must NOT be referenced after its with-block closes.
    with rasterio.open(insar_path) as insar_src:
        insar_data      = insar_src.read(1)
        meta            = insar_src.meta.copy()
        nodata          = insar_src.nodata
        insar_shape     = insar_data.shape
        insar_transform = insar_src.transform
        insar_crs       = insar_src.crs

    if nodata is None:
        for candidate in [-9999.0, -9999, 0.0]:
            if np.any(insar_data == candidate):
                nodata = candidate
                print(f"  Auto-detected nodata = {nodata} in {insar_path.name}")
                break
    if nodata is None:
        nodata = -9999.0
        print(f"  WARNING: Could not detect nodata, defaulting to -9999 "
              f"for {insar_path.name}")

    # FIX: cast to float32 so nodata=-9999.0 never truncates on integer source dtypes
    insar_data = insar_data.astype(np.float32)

    with rasterio.open(mask_path) as mask_src:
        mask_aligned = np.zeros(insar_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(mask_src, 1),
            destination=mask_aligned,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=insar_transform,
            dst_crs=insar_crs,
            resampling=Resampling.nearest,
        )

    masked      = insar_data.copy()
    cut         = (mask_aligned == 2)
    masked[cut] = nodata

    n_cut   = int(cut.sum())
    n_total = int(np.prod(insar_shape))
    print(f"  Masked out {n_cut:,} / {n_total:,} pixels ({100*n_cut/n_total:.1f}%)")

    # Always write float32 so nodata round-trips correctly
    meta.update(dtype="float32", nodata=nodata)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(masked.astype(np.float32), 1)


# ==============================================================================
# Main
# ==============================================================================

def main():

    # ── Part 1: Quality mask ──────────────────────────────────────────────────
    tifs = []
    for pattern in TIF_PATTERNS:
        for f in SHIFTED_DIR.glob(pattern):
            if "_masked" not in f.stem:
                tifs.append(f)
    tifs = sorted(set(tifs))

    if not tifs:
        print(f"No TIFs found in {SHIFTED_DIR}. Run 11_shift_geotransform.py first.")
        return

    print("12_apply_quality_slope_mask.py")
    print("=" * 60)
    print("PART 1 -- Quality mask")
    print(f"  Mask SHP  : {MASK_SHP}")
    print(f"  Input dir : {SHIFTED_DIR}")
    print(f"  Found     : {len(tifs)} TIF(s)")
    print("=" * 60)

    print("Checking/repairing TIF CRS metadata ...")
    for tif in tifs:
        repair_tif_crs(tif, epsg=SHP_EPSG)

    with rasterio.open(tifs[0]) as src:
        tif_crs_wkt = src.crs.to_wkt()
    mask_geoms = load_mask_geometries(MASK_SHP, tif_crs_wkt)
    print()

    report = [
        "12_apply_quality_slope_mask report\n",
        f"Timestamp : {datetime.now().isoformat(timespec='seconds')}\n",
        f"Mask SHP  : {MASK_SHP}\n",
        f"Input dir : {SHIFTED_DIR}\n",
        "=" * 60 + "\n\n",
        "PART 1 -- Quality mask\n\n",
    ]

    for tif in tifs:
        dst    = OUT_DIR / (tif.stem + "_masked.tif")
        before, n_masked, after = apply_quality_mask(tif, dst, mask_geoms)
        pct    = n_masked / max(before, 1) * 100
        print(f"  {tif.name}")
        print(f"    Before: {before:,} | Masked: {n_masked:,} ({pct:.1f}%) | After: {after:,}")
        print(f"    -> {dst.name}")
        report += [
            f"{tif.name}\n",
            f"  Before valid : {before:,}\n",
            f"  Masked out   : {n_masked:,} ({pct:.1f}%)\n",
            f"  After valid  : {after:,}\n",
            f"  Output       : {dst.name}\n",
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

    # ── Part 2: Slope mask ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PART 2 -- Slope mask")
    print(f"  Results dir : {RESULTS_DIR}")
    print("=" * 60)

    report += ["\n" + "=" * 60 + "\n", "PART 2 -- Slope mask\n\n"]

    for thresh, mask_filename in SLOPE_MASKS.items():
        mask_path = RESULTS_DIR / mask_filename
        if not mask_path.exists():
            print(f"\n  ERROR: Mask not found: {mask_path}")
            report.append(f"ERROR: Mask not found: {mask_path}\n")
            continue
        print(f"\n  --- Slope threshold: {thresh} deg ---")
        report.append(f"Slope threshold: {thresh} deg\n")
        for insar_filename in INSAR_FILES:
            insar_path = RESULTS_DIR / insar_filename
            if not insar_path.exists():
                print(f"    SKIP (not found): {insar_filename}")
                continue
            out_name    = f"{insar_path.stem}_slope{thresh}deg.tif"
            output_path = RESULTS_DIR / out_name
            print(f"    {insar_filename}")
            apply_slope_mask(insar_path, mask_path, output_path)
            print(f"    -> {out_name}")
            report.append(f"  {insar_filename} -> {out_name}\n")
        report.append("\n")

    out_rep = OUT_DIR / "12_masked_report.txt"
    out_rep.write_text("".join(report), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Done. Quality-masked TIFs : {OUT_DIR}")
    print(f"      Slope-masked TIFs   : {RESULTS_DIR}")
    print(f"      Report              : {out_rep.name}")


if __name__ == "__main__":
    main()
