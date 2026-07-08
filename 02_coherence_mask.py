"""
02_coherence_mask.py
Builds a per-pixel coherence mask from a SNAP BEAM-DIMAP unwrapped stack.
All IFGs in the stack are used regardless of their scene-mean coherence.

Threshold strategy:
- Per-pixel mean coherence >= 0.40 (DS-InSAR candidate selection threshold)
- No IFG-level mean threshold applied (all IFGs contribute to covariance estimation)

Memory strategy:
- Bands are streamed one at a time into running sum/count accumulators
- Peak RAM = 2 x (rows x cols x float64) + 1 x (rows x cols x float32) per band

Geotransform strategy (v1.3 FIX):
- Transform derived directly from localIncidenceAngle.hdr inside the TC .data
  folder — guaranteed to be on the exact same pixel grid as all other ENVI bands.
- Replaces v1.2 which used the separately exported localIncidenceAngle_TC.tif.
  That file was exported from SNAP with a different grid alignment (snapped
  differently), causing a ~2-3 pixel (~55-83 m) offset in all outputs.
- _envi_hdr_to_rasterio_transform() computes the rasterio Affine from the
  ENVI map info reference pixel, correctly converting from 1-based pixel
  centre convention (ENVI) to top-left corner convention (rasterio/GeoTIFF).

PROJ conflict fix:
- Uses pyproj.datadir.get_data_dir() to locate the correct proj.db
- CRS assigned via hardcoded WKT - never calls CRS.from_epsg(), so
  proj.db version mismatches cannot cause a failure here

Outputs (per network, GeoTIFF written to WORK_DIR):
- _mean_coherence.tif : float32 mean coherence across all IFGs
- _coherence_mask.tif : binary mask (1 = valid DS candidate, 0 = masked)
- _coherence_report.txt: per-IFG statistics

Fix log:
v1.1 WKT body described BC Albers but carried AUTHORITY["EPSG","3155"].
     Replaced with correct EPSG:3155 NAD83(CSRS)/UTM zone 8N definition.
v1.2 Geotransform source changed from full-scene SNAP ENVI .hdr map info
     to localIncidenceAngle_TC.tif to fix ~95 km SE offset.
v1.3 Geotransform source changed from localIncidenceAngle_TC.tif (separately
     exported, different grid) to localIncidenceAngle.hdr inside the TC .data
     folder (same grid as all coherence/phase bands). Fixes ~2-3 pixel offset.
"""

import os
import re
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path


# ── Fix PROJ conflict before any geo imports ───────────────────────────────────
def _fix_proj():
    try:
        from pyproj.datadir import get_data_dir
        proj_dir = str(get_data_dir())
        os.environ["PROJ_DATA"] = proj_dir
        os.environ["PROJ_LIB"]  = proj_dir
        os.environ["PROJ_NETWORK"] = "OFF"
        return True
    except Exception:
        pass
    _conda = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\anaconda3"))
    _p = _conda / "Library" / "share" / "proj"
    if _p.exists():
        os.environ["PROJ_DATA"] = str(_p)
        os.environ["PROJ_LIB"]  = str(_p)
        os.environ["PROJ_NETWORK"] = "OFF"
        return True
    return False

_fix_proj()


# ── Configuration ──────────────────────────────────────────────────────────────
WORK_DIR            = Path(r"C:\bcosta\DS-InSAR\tc")
PIXEL_COH_THRESHOLD = 0.40
OUTPUT_NODATA       = -9999.0

# ENVI data type codes -> numpy dtype
ENVI_DTYPE = {
    1:  "u1",   # uint8
    2:  "i2",   # int16
    3:  "i4",   # int32
    4:  "f4",   # float32
    5:  "f8",   # float64
    12: "u2",   # uint16
}

# EPSG:3155 NAD83(CSRS) / UTM zone 8N hardcoded WKT
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
    text = Path(hdr_path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"map info\s*=\s*\{(.+?)\}", text, re.IGNORECASE | re.DOTALL)
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
    # Centre of pixel (1,1) in map coordinates
    c1_east  = ref_east  - (ref_col - 1.0) * px
    c1_north = ref_north + (ref_row - 1.0) * py
    # Top-left corner of pixel (0,0) = rasterio origin
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


def _get_master_transform(data_dir):
    """
    Return (transform, crs, shape) sourced from localIncidenceAngle.hdr
    inside the TC .data folder.  This is the authoritative grid — identical
    to the pixel grid of all coherence and phase bands in the same folder.

    Falls back gracefully with a clear error if the file is missing.
    """
    from rasterio.crs import CRS
    hdr = Path(data_dir) / "localIncidenceAngle.hdr"
    if not hdr.exists():
        raise FileNotFoundError(
            f"localIncidenceAngle.hdr not found in {data_dir}\n"
            f"This file must exist to derive the correct geotransform.\n"
            f"Export localIncidenceAngle from the same TC product in SNAP\n"
            f"(it should already be present as a band in the .data folder)."
        )
    transform = _envi_hdr_to_rasterio_transform(hdr)
    nrows, ncols = _envi_hdr_shape(hdr)
    crs = CRS.from_wkt(_EPSG3155_WKT)
    print(f"  [GeoTIFF] master grid  : {hdr.name} (v1.3 — same pixel grid as data)")
    print(f"  [GeoTIFF] origin       : ({transform.c:.4f}, {transform.f:.4f})")
    print(f"  [GeoTIFF] pixel size   : {transform.a:.4f} x {abs(transform.e):.4f} m")
    print(f"  [GeoTIFF] shape        : {ncols} x {nrows} px")
    return transform, crs, (nrows, ncols)


# ── Helpers ────────────────────────────────────────────────────────────────────
def find_coherence_band_elements(root):
    """Return list of band name dicts for all coh_* bands in the DIMAP."""
    bands = []
    for sb in root.iter("Spectral_Band_Info"):
        name_el = sb.find("BAND_NAME")
        idx_el  = sb.find("BAND_INDEX")
        if name_el is not None and name_el.text and name_el.text.strip().startswith("coh_"):
            bands.append({
                "name":  name_el.text.strip(),
                "index": int(idx_el.text.strip()) if idx_el is not None else -1,
            })
    return bands


def read_envi_img(hdr_path, img_path):
    """
    Read a flat binary ENVI .img file.
    Parses byte order and data type from the .hdr so big-endian files
    (SNAP default: byte order = 1) are read correctly.
    Clamps output to [0, 1] — coherence is physically bounded.
    """
    text = Path(hdr_path).read_text(encoding="utf-8", errors="ignore")

    def parse_int(key):
        m = re.search(rf"^\s*{key}\s*=\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
        return int(m.group(1)) if m else None

    samples   = parse_int("samples")
    lines     = parse_int("lines")
    data_type = parse_int("data type")

    if samples is None or lines is None:
        raise ValueError(f"Cannot read dimensions from {hdr_path}")

    bo_match   = re.search(r"^\s*byte order\s*=\s*(\d)", text, re.MULTILINE | re.IGNORECASE)
    byte_order = int(bo_match.group(1)) if bo_match else 0
    endian     = ">" if byte_order == 1 else "<"

    base_dtype = ENVI_DTYPE.get(data_type, "f4")
    np_dtype   = np.dtype(f"{endian}{base_dtype}")

    data     = np.fromfile(img_path, dtype=np_dtype)
    expected = samples * lines
    if data.size != expected:
        raise ValueError(f"{Path(img_path).name}: expected {expected} values, got {data.size}")

    arr = data.reshape((lines, samples)).astype(np.float32)
    arr = np.where(np.isfinite(arr) & (arr >= 0) & (arr <= 1), arr, np.nan)
    return arr


def write_geotiff(array, out_path, nodata, transform, crs, master_shape):
    """
    Write a float32 GeoTIFF using the transform and CRS derived from
    localIncidenceAngle.hdr inside the TC .data folder (v1.3).

    Raises ValueError if array shape does not match master_shape.
    """
    import rasterio

    nrows, ncols = array.shape
    if (nrows, ncols) != master_shape:
        raise ValueError(
            f"Array shape {(nrows, ncols)} does not match master shape {master_shape}.\n"
            f"Check that multilooking parameters match across all bands."
        )

    with rasterio.open(
        out_path, "w",
        driver    = "GTiff",
        height    = nrows,
        width     = ncols,
        count     = 1,
        dtype     = np.float32,
        crs       = crs,
        transform = transform,
        nodata    = nodata,
        compress  = "deflate",
    ) as dst:
        dst.write(array.astype(np.float32), 1)


# ── Per-stack processing ───────────────────────────────────────────────────────
def process_stack(dim_path, pixel_thr=PIXEL_COH_THRESHOLD):
    dim_path = Path(dim_path)
    data_dir = dim_path.with_suffix(".data")
    stem     = dim_path.stem

    print("\n" + "=" * 60)
    print(f"Stack : {stem}")

    if not data_dir.is_dir():
        print(f"  ERROR: .data folder not found: {data_dir}")
        return

    # Derive geotransform from localIncidenceAngle.hdr inside .data (v1.3)
    try:
        transform, crs, master_shape = _get_master_transform(data_dir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    tree = ET.parse(dim_path)
    root = tree.getroot()

    coh_bands = find_coherence_band_elements(root)
    print(f"  Coherence bands found : {len(coh_bands)}")

    if not coh_bands:
        print("  ERROR: no coh_* bands found. Check .dim file. Aborting.")
        return

    report_lines = [
        f"Coherence report : {stem}\n",
        f"Pixel threshold  : {pixel_thr} (DS-InSAR candidate selection)\n",
        f"IFG mean filter  : DISABLED - all IFGs contribute\n",
        f"CRS              : EPSG:3155 (NAD83(CSRS) / UTM zone 8N)\n",
        f"Geotransform     : derived from localIncidenceAngle.hdr in .data (v1.3)\n\n",
        f"{'Band':<55} {'Mean':>6} {'Median':>7} {'<thr':>7} {'NaN%':>6}\n",
        "-" * 85 + "\n",
    ]

    sum_coh   = None
    count_coh = None
    n_loaded  = 0

    for b in coh_bands:
        name     = b["name"]
        hdr_path = data_dir / (name + ".hdr")
        img_path = data_dir / (name + ".img")

        if not img_path.exists():
            print(f"  WARNING: missing {img_path.name} - skipping")
            continue

        arr = read_envi_img(hdr_path, img_path)

        if sum_coh is None:
            sum_coh   = np.zeros(arr.shape, dtype=np.float64)
            count_coh = np.zeros(arr.shape, dtype=np.int32)

        finite_mask = np.isfinite(arr)
        nan_pct     = float(np.sum(~finite_mask) / arr.size * 100)
        mean        = float(np.nanmean(arr))
        med         = float(np.nanmedian(arr))
        pct_low     = float(np.sum(arr[finite_mask] < pixel_thr) / arr.size * 100)

        report_lines.append(
            f"  {name:<55} {mean:6.3f} {med:7.3f} {pct_low:7.1f}% {nan_pct:5.1f}%\n"
        )
        print(f"  {name[:50]:<50} mean={mean:.3f} <{pixel_thr}={pct_low:.1f}% NaN={nan_pct:.1f}%")

        sum_coh[finite_mask]   += arr[finite_mask].astype(np.float64)
        count_coh[finite_mask] += 1
        del arr
        n_loaded += 1

    if sum_coh is None or n_loaded == 0:
        print("  ERROR: no coherence bands could be loaded.")
        return

    all_nan  = count_coh == 0
    mean_coh = np.where(
        ~all_nan,
        (sum_coh / np.where(count_coh > 0, count_coh, 1)).astype(np.float32),
        np.nan,
    )
    del sum_coh, count_coh

    mask         = (mean_coh >= pixel_thr).astype(np.float32)
    mean_coh_out = mean_coh.copy()
    mean_coh_out[all_nan] = OUTPUT_NODATA
    mask[all_nan]         = OUTPUT_NODATA

    valid_pct  = float(np.sum(mask == 1) / mask.size * 100)
    nodata_pct = float(np.sum(all_nan)   / all_nan.size * 100)

    print(f"\n  IFGs used             : {n_loaded}")
    print(f"  Mean coherence range  : {float(np.nanmin(mean_coh)):.3f} - {float(np.nanmax(mean_coh)):.3f}")
    print(f"  Valid DS pixels (>={pixel_thr}) : {valid_pct:.1f}%")
    print(f"  All-NaN pixels (nodata): {nodata_pct:.1f}%")

    out_mean = dim_path.parent / f"{stem}_mean_coherence.tif"
    out_mask = dim_path.parent / f"{stem}_coherence_mask.tif"
    out_rep  = dim_path.parent / f"{stem}_coherence_report.txt"

    write_geotiff(mean_coh_out, out_mean, OUTPUT_NODATA, transform, crs, master_shape)
    write_geotiff(mask,         out_mask, OUTPUT_NODATA, transform, crs, master_shape)

    report_lines += [
        f"\nSummary\n",
        f"  IFGs used      : {n_loaded}\n",
        f"  Valid pixels   : {valid_pct:.1f}%\n",
        f"  Nodata pixels  : {nodata_pct:.1f}%\n",
        f"  CRS            : EPSG:3155 (NAD83(CSRS) / UTM zone 8N)\n",
        f"  Geotransform   : derived from localIncidenceAngle.hdr (v1.3 fix)\n",
    ]
    out_rep.write_text("".join(report_lines), encoding="utf-8")

    print(f"\n  Written: {out_mean.name}")
    print(f"  Written: {out_mask.name}")
    print(f"  Written: {out_rep.name}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    dim_files = sorted(WORK_DIR.glob("*.dim"))
    if not dim_files:
        print(f"No .dim files found in {WORK_DIR}")
        return
    print(f"Found {len(dim_files)} stack(s) in {WORK_DIR}. Processing each independently.")
    for dim in dim_files:
        process_stack(dim)
    print("\n" + "=" * 60)
    print("Coherence masking complete.")
    print("Run 03_load_phases.py next.")


if __name__ == "__main__":
    main()
