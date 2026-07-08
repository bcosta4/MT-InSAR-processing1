"""
07_export_geotiff.py
Maps 1D inversion result vectors back to 2D GeoTIFFs.

Spatial reference is derived from localIncidenceAngle.hdr inside the TC .data
folder — guaranteed to be on the exact same pixel grid as all coherence and
phase ENVI bands processed by scripts 02-06.

All physical units preserved from 05b/06 outputs:
  velocity   : cm/year  (negative = subsidence, positive = uplift)
  timeseries : cm

Exports per network:
  _velocity_map.tif      float32 single-band [cm/year]
  _timeseries_stack.tif  float32 multi-band  [cm], one band per date
  _residuals_map.tif     float32 single-band [rad]
  _qc_report.txt         quality control summary

Sign convention: negative = subsidence, positive = uplift.

Fix log:
v1.3  Geotransform source changed from localIncidenceAngle_TC.tif
      (separately exported from SNAP, different pixel grid) to
      localIncidenceAngle.hdr inside the TC .data folder.
      This fixes the ~2-3 pixel (~55-83 m) spatial offset observed in
      all GeoTIFF outputs relative to the Google Satellite basemap.
      _envi_hdr_to_rasterio_transform() implements the correct ENVI
      1-based pixel-centre to rasterio top-left-corner conversion.
"""

import csv
import re
from datetime import datetime
from pathlib import Path
import numpy as np


WORK_DIR               = Path(r"C:\bcosta\DS-InSAR\tc")
OUTPUT_NODATA          = -9999.0
VEL_MIN_CLIP           = -6.0
VEL_MAX_CLIP           =  3.0
UPLIFT_DOMINANCE_THRESH = 0.80
COMPRESS               = "deflate"

_EPSG3155_WKT = (
    'PROJCS["NAD83(CSRS) / UTM zone 8N",'
    'GEOGCS["NAD83(CSRS)",'
    'DATUM["NAD83_Canadian_Spatial_Reference_System",'
    'SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],'
    'AUTHORITY["EPSG","6140"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4617"]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",-135],'
    'PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AUTHORITY["EPSG","3155"]]'
)


# ── Geotransform from ENVI .hdr (v1.3) ────────────────────────────────────────

def _envi_hdr_to_rasterio_transform(hdr_path):
    """
    Build a rasterio Affine transform from an ENVI .hdr map info line.

    ENVI map info convention:
      {projection, ref_col, ref_row, ref_east, ref_north, pixel_x, pixel_y, ...}
      ref_col / ref_row are 1-based and refer to the CENTRE of that pixel.

    rasterio Affine convention:
      origin = top-left CORNER of pixel (0, 0)  [0-based]

    Conversion:
      centre of pixel (1,1):  east  = ref_east  - (ref_col - 1) * px
                               north = ref_north + (ref_row - 1) * py
      top-left corner (0,0):  east  = centre_east  - px/2
                               north = centre_north + py/2
    """
    from rasterio.transform import Affine
    text  = Path(hdr_path).read_text(encoding="utf-8", errors="ignore")
    m     = re.search(r"map info\s*=\s*\{(.+?)\}", text, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError(f"No 'map info' found in {hdr_path}")
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) < 7:
        raise ValueError(f"map info has fewer than 7 fields in {hdr_path}")
    ref_col   = float(parts[1])
    ref_row   = float(parts[2])
    ref_east  = float(parts[3])
    ref_north = float(parts[4])
    px        = float(parts[5])
    py        = float(parts[6])
    c1_east   = ref_east  - (ref_col - 1.0) * px
    c1_north  = ref_north + (ref_row  - 1.0) * py
    origin_east  = c1_east  - px / 2.0
    origin_north = c1_north + py / 2.0
    return Affine(px, 0.0, origin_east, 0.0, -py, origin_north)


def _envi_hdr_shape(hdr_path):
    """Return (nrows, ncols) from ENVI .hdr samples/lines fields."""
    text = Path(hdr_path).read_text(encoding="utf-8", errors="ignore")
    def _int(key):
        m = re.search(rf"^\s*{key}\s*=\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
        if not m:
            raise ValueError(f"'{key}' not found in {hdr_path}")
        return int(m.group(1))
    return _int("lines"), _int("samples")


def get_master_meta():
    """
    Return (transform, crs, shape) sourced from localIncidenceAngle.hdr
    inside the first TC .data folder found in WORK_DIR.

    The .hdr inside .data is guaranteed to be on the same pixel grid as
    all coherence, phase, and unwrapped interferogram bands (v1.3 fix).
    """
    from rasterio.crs import CRS

    dim_files = sorted(WORK_DIR.glob("*.dim"))
    if not dim_files:
        raise FileNotFoundError(f"No .dim files found in {WORK_DIR}")

    data_dir = dim_files[0].with_suffix(".data")
    hdr      = data_dir / "localIncidenceAngle.hdr"

    if not hdr.exists():
        raise FileNotFoundError(
            f"localIncidenceAngle.hdr not found: {hdr}\n"
            f"This file must exist inside the TC .data folder.\n"
            f"It should be present automatically as a SNAP output band.\n"
            f"Do NOT use the separately exported localIncidenceAngle_TC.tif —\n"
            f"that file has a different grid alignment."
        )

    transform        = _envi_hdr_to_rasterio_transform(hdr)
    nrows, ncols     = _envi_hdr_shape(hdr)
    crs              = CRS.from_wkt(_EPSG3155_WKT)
    shape            = (nrows, ncols)
    return transform, crs, shape


# ── I/O helpers ───────────────────────────────────────────────────────────────

def vector_to_2d(values_1d, pixel_coords, shape, nodata=OUTPUT_NODATA):
    out_2d = np.full(shape, nodata, dtype=np.float32)
    out_2d[pixel_coords[:, 0], pixel_coords[:, 1]] = values_1d
    return out_2d


def write_single_band_tif(array_2d, out_path, transform, crs, nodata, description=""):
    import rasterio
    nrows, ncols = array_2d.shape
    profile = dict(
        driver="GTiff", height=nrows, width=ncols, count=1,
        dtype="float32", crs=crs, transform=transform,
        nodata=nodata, compress=COMPRESS,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array_2d.astype(np.float32), 1)
        if description:
            dst.update_tags(1, description=description)


def write_multiband_tif(array_3d, out_path, transform, crs, nodata, band_descriptions=None):
    import rasterio
    n_bands, nrows, ncols = array_3d.shape
    profile = dict(
        driver="GTiff", height=nrows, width=ncols, count=n_bands,
        dtype="float32", crs=crs, transform=transform,
        nodata=nodata, compress=COMPRESS,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(n_bands):
            dst.write(array_3d[i].astype(np.float32), i + 1)
            if band_descriptions and i < len(band_descriptions):
                dst.set_band_description(i + 1, str(band_descriptions[i]))
                dst.update_tags(i + 1, date=str(band_descriptions[i]))
    return n_bands


# ── Per-network processing ─────────────────────────────────────────────────────

def process_network(stem):
    print("\n" + "=" * 60)
    print(f"Network : {stem}")

    vel_path    = WORK_DIR / f"{stem}_velocity_referenced.npy"
    ts_path     = WORK_DIR / f"{stem}_timeseries_referenced.npy"
    res_path    = WORK_DIR / f"{stem}_residuals.npy"
    coords_path = WORK_DIR / f"{stem}_pixel_coords.npy"
    dates_path  = WORK_DIR / f"{stem}_date_list.csv"

    for p in [vel_path, ts_path, coords_path, dates_path]:
        if not p.exists():
            print(f"  ERROR: missing {p.name} -- run scripts 05 and 06 first.")
            return

    velocity    = np.load(vel_path).astype(np.float32)
    timeseries  = np.load(ts_path).astype(np.float32)
    pixel_coords = np.load(coords_path)

    if timeseries.ndim != 2:
        print(f"  ERROR: timeseries has unexpected shape {timeseries.shape}")
        return

    date_strings = []
    days_list    = []
    with open(dates_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_strings.append(row["date"])
            days_list.append(float(row.get("days_since_first", 0)))
    days_arr = np.asarray(days_list, dtype=np.float32)

    n_ts_dates, n_pixels = timeseries.shape
    n_dates = len(date_strings)
    if n_dates != n_ts_dates:
        n_dates      = min(n_dates, n_ts_dates)
        date_strings = date_strings[:n_dates]
        days_arr     = days_arr[:n_dates]
        timeseries   = timeseries[:n_dates, :]

    # v1.3: geotransform from localIncidenceAngle.hdr inside .data
    src_transform, src_crs, shape = get_master_meta()
    nrows, ncols = shape
    crs_str = src_crs.to_string() if src_crs else "unknown"

    raw_min  = float(np.nanmin(velocity))
    raw_max  = float(np.nanmax(velocity))
    n_before = int(np.sum(np.isfinite(velocity)))

    print(f"  Units        : cm/year (velocity)  cm (timeseries)")
    print(f"  Master grid  : localIncidenceAngle.hdr in .data (v1.3)")
    print(f"  CRS          : {crs_str}")
    print(f"  Shape        : {shape[0]} x {shape[1]}")
    print(f"  Velocity     : {velocity.shape}  range {raw_min:.3f} - {raw_max:.3f} cm/year")
    print(f"  Time series  : {timeseries.shape}")
    print(f"  Pixel coords : {pixel_coords.shape}")
    print(f"  Dates        : {n_dates} ({date_strings[0]} -> {date_strings[-1]})")

    if pixel_coords.ndim != 2 or pixel_coords.shape[1] != 2:
        print(f"  ERROR: pixel_coords has unexpected shape {pixel_coords.shape}")
        return

    if velocity.shape[0] != pixel_coords.shape[0] or timeseries.shape[1] != pixel_coords.shape[0]:
        print("  ERROR: vector lengths do not match pixel_coords. Stop and verify scripts 05/06.")
        return

    # ── Velocity QC ──────────────────────────────────────────────────────────
    clip_mask  = (velocity < VEL_MIN_CLIP) | (velocity > VEL_MAX_CLIP)
    n_clipped  = int(np.sum(clip_mask))
    velocity   = velocity.copy()
    timeseries = timeseries.copy()
    velocity[clip_mask]      = np.nan
    timeseries[:, clip_mask] = np.nan
    if n_clipped > 0:
        print(f"  [QC] Clipped {n_clipped:,} velocity pixels outside [{VEL_MIN_CLIP}, {VEL_MAX_CLIP}] cm/year")

    valid_mask        = np.isfinite(velocity)
    n_valid           = int(np.sum(valid_mask))
    n_pos             = int(np.sum(velocity[valid_mask] > 0))
    pct_pos           = n_pos / n_valid if n_valid else 0.0
    sign_warning      = False
    offset_applied_cm = 0.0

    if pct_pos > UPLIFT_DOMINANCE_THRESH:
        sign_warning  = True
        median_vel    = float(np.nanmedian(velocity))
        print(f"  [QC] WARNING: {pct_pos*100:.1f}% positive velocity; subtracting median {median_vel:.3f} cm/year")
        velocity   -= median_vel
        timeseries -= (median_vel / 365.25) * days_arr[:, np.newaxis]
        offset_applied_cm = median_vel
        valid_mask  = np.isfinite(velocity)
        n_valid     = int(np.sum(valid_mask))

    n_subsiding = int(np.sum(velocity[valid_mask] < 0))
    n_uplifting = int(np.sum(velocity[valid_mask] > 0))
    vel_mean    = float(np.nanmean(velocity))
    vel_std     = float(np.nanstd(velocity))
    vel_min     = float(np.nanmin(velocity))
    vel_max     = float(np.nanmax(velocity))

    print(f"  Velocity QC  : mean={vel_mean:.3f} std={vel_std:.3f} range [{vel_min:.3f}, {vel_max:.3f}] cm/year")
    print(f"  Subsiding    : {n_subsiding:,} ({n_subsiding/max(n_valid,1)*100:.1f}%)")
    print(f"  Uplifting    : {n_uplifting:,} ({n_uplifting/max(n_valid,1)*100:.1f}%)")

    # ── [1/3] Velocity map ────────────────────────────────────────────────────
    print("  [1/3] Velocity map...")
    velocity_export = velocity.copy()
    velocity_export[~np.isfinite(velocity_export)] = OUTPUT_NODATA
    vel_2d   = vector_to_2d(velocity_export, pixel_coords, shape)
    out_vel  = WORK_DIR / f"{stem}_velocity_map.tif"
    write_single_band_tif(
        vel_2d, out_vel, src_transform, src_crs, OUTPUT_NODATA,
        description=(
            "Mean LOS velocity [cm/year]. Negative=subsidence (away from satellite), "
            "Positive=uplift (toward satellite). "
            f"Clipped [{VEL_MIN_CLIP}, {VEL_MAX_CLIP}] cm/year. "
            + (f"Zero-median correction: {offset_applied_cm:.3f} cm/year subtracted."
               if sign_warning else "No zero-median correction applied.")
        ),
    )
    valid_vel = vel_2d[vel_2d != OUTPUT_NODATA]
    if valid_vel.size:
        print(f"  Velocity map : {valid_vel.min():.3f} to {valid_vel.max():.3f} cm/year")
    print(f"  Written      : {out_vel.name}")

    # ── [2/3] Time series stack ───────────────────────────────────────────────
    print(f"  [2/3] Time series stack ({n_dates} bands)...")
    ts_3d = np.full((n_dates, nrows, ncols), OUTPUT_NODATA, dtype=np.float32)
    for epoch in range(n_dates):
        ep_vals = timeseries[epoch, :].copy()
        ep_vals[~np.isfinite(ep_vals)] = OUTPUT_NODATA
        ts_3d[epoch] = vector_to_2d(ep_vals, pixel_coords, shape)
        if epoch % 5 == 0 or epoch == n_dates - 1:
            print(f"    Epoch {epoch+1:3d}/{n_dates} ({date_strings[epoch]})")
    out_ts = WORK_DIR / f"{stem}_timeseries_stack.tif"
    write_multiband_tif(ts_3d, out_ts, src_transform, src_crs, OUTPUT_NODATA,
                        band_descriptions=date_strings)
    print(f"  Written      : {out_ts.name}")

    # ── [3/3] Residuals ───────────────────────────────────────────────────────
    res_mean = np.nan
    res_max  = np.nan
    if res_path.exists():
        print("  [3/3] Residuals map...")
        residuals     = np.load(res_path)
        mean_abs_res  = np.nanmean(np.abs(residuals), axis=0)
        res_2d        = vector_to_2d(mean_abs_res, pixel_coords, shape)
        out_res       = WORK_DIR / f"{stem}_residuals_map.tif"
        write_single_band_tif(res_2d, out_res, src_transform, src_crs, OUTPUT_NODATA,
                               description="Mean absolute inversion residual [rad].")
        valid_res = res_2d[res_2d != OUTPUT_NODATA]
        if valid_res.size:
            res_mean = float(valid_res.mean())
            res_max  = float(valid_res.max())
            print(f"  Residuals    : {valid_res.min():.4f} - {res_max:.4f} rad (mean {res_mean:.4f})")
        print(f"  Written      : {out_res.name}")

    # ── QC report ─────────────────────────────────────────────────────────────
    out_qc = WORK_DIR / f"{stem}_qc_report.txt"
    if np.isfinite(res_mean):
        res_quality = "POOR (mean >1 rad)" if res_mean > 1.0 else "ACCEPTABLE"
        res_lines = (
            f"  Mean abs residual : {res_mean:.4f} rad\n"
            f"  Max abs residual  : {res_max:.4f} rad\n"
            f"  Quality flag      : {res_quality}\n"
        )
    else:
        res_lines = "  Residuals not available\n"

    qc_lines = [
        f"QC report   : {stem}\n",
        f"Timestamp   : {datetime.now().isoformat(timespec='seconds')}\n",
        f"\n[Grid]\n",
        f"  Master source : localIncidenceAngle.hdr in .data (v1.3)\n",
        f"  CRS           : {crs_str}\n",
        f"  Transform     : {src_transform}\n",
        f"  Shape         : {nrows} x {ncols}\n",
        f"\n[Velocity after QC]\n",
        f"  Raw range before clip  : {raw_min:.4f} - {raw_max:.4f} cm/year\n",
        f"  Clip thresholds        : [{VEL_MIN_CLIP}, {VEL_MAX_CLIP}] cm/year\n",
        f"  Pixels clipped         : {n_clipped:,} ({n_clipped/max(n_before,1)*100:.2f}%)\n",
        f"  Zero-median correction : " + (f"YES -- {offset_applied_cm:.4f} cm/year subtracted\n"
                                          if sign_warning else "NO\n"),
        f"  Final mean    : {vel_mean:.4f} cm/year\n",
        f"  Final std     : {vel_std:.4f} cm/year\n",
        f"  Final range   : {vel_min:.4f} - {vel_max:.4f} cm/year\n",
        f"  Subsiding     : {n_subsiding:,} ({n_subsiding/max(n_valid,1)*100:.1f}%)\n",
        f"  Uplifting     : {n_uplifting:,} ({n_uplifting/max(n_valid,1)*100:.1f}%)\n",
        f"\n[Residuals]\n",
        res_lines,
        f"\n[Sign convention]\n",
        f"  negative velocity = subsidence (moving away from satellite)\n",
        f"  positive velocity = uplift (moving toward satellite)\n",
    ]
    out_qc.write_text("".join(qc_lines), encoding="utf-8")
    print(f"  Written      : {out_qc.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("07_export_geotiff.py  (v1.3 — georeference from .data/localIncidenceAngle.hdr)")
    vel_files = sorted(WORK_DIR.glob("*_velocity_referenced.npy"))
    if not vel_files:
        print(f"No *_velocity_referenced.npy found in {WORK_DIR}")
        return
    stems = [f.stem.replace("_velocity_referenced", "") for f in vel_files]
    print(f"Found {len(stems)} network(s): {stems}")
    print(f"Vel clip : [{VEL_MIN_CLIP}, {VEL_MAX_CLIP}] cm/year")
    for stem in stems:
        process_network(stem)
    print("\n" + "=" * 60)
    print("GeoTIFF export complete.")
    print(f"All outputs written to: {WORK_DIR}")


if __name__ == "__main__":
    main()
