"""
Coherence-threshold masking for SNAP InSAR ENVI rasters.

Replaces the lake-masking strategy in mask_lakes_insar_v3.py with a
coherence-based mask: pixels where coh < COH_THRESHOLD are set to NaN
across all target .img files in every .data directory.

Per-folder pipeline
-------------------
1. Discover .data dir(s) inside each yyyymmdd_yyyymmdd folder.
2. For each .data dir:
   a. Locate the coherence raster (filename starts with "coh", ends ".img").
   b. Read it as float32 using the ENVI header.
   c. Build a boolean mask:  mask = (coh_array < COH_THRESHOLD)
   d. List target .img files (i_ifg_*, q_ifg_*, Phase_ifg_*, coh_*, etc.;
      skip elevation.*).
   e. For every target file: apply mask → NaN, atomic write with .bak backup.

Design notes
------------
- Coherence and all interferogram bands are written by a single SNAP Write
  node into one BEAM-DIMAP .data directory, so every band shares the same
  pixel grid.  No georeferencing or reprojection is needed to align the
  coherence mask with the targets — a direct array-shape check is sufficient.
- The full georeferencing infrastructure (hdr / dim / tie-point) is retained
  unchanged because it may be needed by downstream scripts.
- The south-to-north flipud applied in v3 for lake masking is NOT applied
  here: the coherence array and the target arrays are in the same native
  SNAP orientation, so the mask is already aligned.
- Parallelism: one ProcessPoolExecutor worker per top-level folder.
  No shared state is needed (unlike the lake version which required a
  per-worker shapefile cache), so no pool initializer is required.
"""

from __future__ import annotations

import re
import os
import sys
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
from tqdm import tqdm

import rasterio
from rasterio.transform import from_origin, Affine
from rasterio.crs import CRS

from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Global threshold — exposed as a module-level constant so it can be
# overridden programmatically and is clearly visible in the source.
# ---------------------------------------------------------------------------
COH_THRESHOLD: float = 0.3


# ============================================================
# ENVI header parsing  (unchanged from v3)
# ============================================================

def parse_envi_hdr(hdr_path: str) -> dict:
    """
    Parse an ENVI .hdr file into a metadata dict.

    Handles multi-line brace-delimited values by collapsing internal
    whitespace so they appear as a single value string.
    """
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()

    # Collapse { ... } blocks to a single line so split("=",1) works.
    content = re.sub(
        r"\{([^}]*)\}",
        lambda m: "{" + " ".join(m.group(1).split()) + "}",
        content,
        flags=re.DOTALL,
    )

    meta: dict = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k.strip().lower()] = v.strip().strip("{}")

    meta["samples"]    = int(meta["samples"])
    meta["lines"]      = int(meta["lines"])
    meta["bands"]      = int(meta.get("bands", 1))
    meta["datatype"]   = int(meta.get("data type", meta.get("datatype", 4)))
    meta["byte order"] = int(meta.get("byte order", 0))   # 0 = little-endian
    return meta


def envi_dtype_to_numpy(dt_code: int) -> type:
    """Map ENVI data-type code to a NumPy scalar type."""
    table = {
        1: np.uint8,  2: np.int16,   3: np.int32,  4: np.float32,
        5: np.float64, 12: np.uint16, 13: np.uint32, 14: np.int64,
    }
    return table.get(dt_code, np.float32)


def read_envi_array(img_path: str, hdr_meta: dict) -> np.ndarray:
    """
    Read an ENVI flat-binary raster and return a float32 NumPy array.

    Shapes:
        single-band  → (lines, samples)
        multi-band   → (bands, lines, samples)

    Skips the redundant .astype() copy when the file is already
    float32 little-endian (the common SNAP output format).
    """
    dt_code = hdr_meta["datatype"]
    base    = envi_dtype_to_numpy(dt_code)
    bo      = hdr_meta.get("byte order", 0)
    dt      = np.dtype(base).newbyteorder(">" if bo == 1 else "<")
    samples = hdr_meta["samples"]
    lines   = hdr_meta["lines"]
    bands   = hdr_meta.get("bands", 1)
    count   = samples * lines * bands
    shape   = (lines, samples) if bands == 1 else (bands, lines, samples)

    with open(img_path, "rb") as fh:
        arr = np.fromfile(fh, dtype=dt, count=count).reshape(shape)

    # Fast path: avoid redundant allocation when already float32 LE
    if base == np.float32 and bo == 0:
        return arr.view(np.float32)
    return arr.astype(np.float32)


# ============================================================
# Map-info / CRS helpers  (unchanged from v3, kept for completeness)
# ============================================================

def parse_map_info(map_info_str: str) -> dict:
    parts = [p.strip() for p in map_info_str.strip("{}").split(",")]
    if len(parts) < 6:
        raise ValueError("map info has fewer than 6 fields.")
    d: dict = {
        "projection_name": parts[0],
        "ref_x":  float(parts[1]),
        "ref_y":  float(parts[2]),
        "tie_x":  float(parts[3]),
        "tie_y":  float(parts[4]),
        "px_x":   float(parts[5]),
        "px_y":   float(parts[6]) if len(parts) >= 7 else float(parts[5]),
        "zone":       None,
        "hemisphere": None,
        "datum":      None,
    }
    if len(parts) >= 8:
        try:
            d["zone"] = int(parts[7])
        except ValueError:
            pass
    if len(parts) >= 9:
        d["hemisphere"] = parts[8]
    if len(parts) >= 10:
        d["datum"] = parts[9]
    return d


def transform_from_map_info(mi: dict) -> Affine:
    west  = mi["tie_x"] - (mi["ref_x"] - 0.5) * mi["px_x"]
    north = mi["tie_y"] + (mi["ref_y"] - 0.5) * mi["px_y"]
    return from_origin(west, north, mi["px_x"], mi["px_y"])


def crs_from_hdr(meta: dict, mi: dict) -> Optional[CRS]:
    for key in ("coordinate system string", "projection info"):
        wkt = meta.get(key, "").strip()
        if wkt:
            try:
                return CRS.from_wkt(wkt)
            except Exception:
                pass
    proj_name = (mi.get("projection_name") or "").lower()
    datum     = (mi.get("datum") or "").lower()
    zone      = mi.get("zone")
    hemi      = (mi.get("hemisphere") or "").lower()
    if "utm" in proj_name and zone is not None and "wgs" in datum:
        epsg = (32700 + zone) if "south" in hemi else (32600 + zone)
        try:
            return CRS.from_epsg(epsg)
        except Exception:
            pass
    return None


def build_grid_from_hdr(hdr_path: str):
    meta   = parse_envi_hdr(hdr_path)
    mi_str = meta.get("map info")
    if not mi_str:
        raise RuntimeError("Header has no 'map info' → cannot build transform.")
    mi        = parse_map_info(mi_str)
    transform = transform_from_map_info(mi)
    crs       = crs_from_hdr(meta, mi)
    if crs is None:
        raise RuntimeError(
            "Unable to determine CRS (no WKT and cannot infer UTM EPSG)."
        )
    return meta["samples"], meta["lines"], transform, crs, meta


# ============================================================
# .dim fallback  (unchanged from v3, kept for completeness)
# ============================================================

def guess_dim_for_data_dir(data_dir: str) -> Optional[str]:
    parent = os.path.dirname(data_dir)
    base   = os.path.basename(data_dir)
    stem   = base[:-5] if base.lower().endswith(".data") else base
    cand   = os.path.join(parent, stem + ".dim")
    if os.path.isfile(cand):
        return cand
    dims = [
        os.path.join(parent, f)
        for f in os.listdir(parent)
        if f.lower().endswith(".dim")
    ]
    return dims[0] if dims else None


def grid_from_dim(dim_path: str):
    with rasterio.open(dim_path) as ds:
        transform, crs, width, height = ds.transform, ds.crs, ds.width, ds.height
    if crs is None:
        raise RuntimeError("DIM has no CRS.")
    if (
        isinstance(transform, Affine)
        and transform.a == 1 and transform.e == 1
        and transform.b == 0 and transform.c == 0
        and transform.d == 0 and transform.f == 0
    ):
        raise RuntimeError("DIM has identity transform (not georeferenced).")
    return width, height, transform, crs


# ============================================================
# Tie-point bilinear upsampling  (unchanged from v3, kept for completeness)
# ============================================================

def upsample_tiegrid_to_full(grid_tie: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Bilinear upsample a small tie grid (h_tie × w_tie) → (H × W).

    Uses RegularGridInterpolator for a single vectorised C call instead of
    the per-column np.interp loop used in older versions.
    """
    h_tie, w_tie = grid_tie.shape
    interp = RegularGridInterpolator(
        (np.linspace(0, H - 1, h_tie), np.linspace(0, W - 1, w_tie)),
        grid_tie.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64),
                         np.arange(W, dtype=np.float64), indexing="ij")
    return interp(np.stack([yy.ravel(), xx.ravel()], axis=-1)
                  ).reshape(H, W).astype(np.float32)


def read_lonlat_from_tiepoints(
    data_dir: str,
    ref_hdr_meta: dict,
    debug_lines: Optional[list] = None,
):
    """Read tie_point_grids/longitude.img + latitude.img, upsampling if needed."""
    tpg      = os.path.join(data_dir, "tie_point_grids")
    lon_path = os.path.join(tpg, "longitude.img")
    lat_path = os.path.join(tpg, "latitude.img")

    for p in (lon_path, lat_path, lon_path[:-4] + ".hdr", lat_path[:-4] + ".hdr"):
        if not os.path.isfile(p):
            raise RuntimeError(
                "Tie-point files latitude.img/longitude.img (+ .hdr) not found."
            )

    lon = read_envi_array(lon_path, parse_envi_hdr(lon_path[:-4] + ".hdr"))
    lat = read_envi_array(lat_path, parse_envi_hdr(lat_path[:-4] + ".hdr"))

    H, W = ref_hdr_meta["lines"], ref_hdr_meta["samples"]
    if debug_lines is not None:
        debug_lines.append(
            f"[tie] raw tie dims={lon.shape[0]}x{lon.shape[1]}, ref dims={H}x{W}"
        )

    if lon.shape != (H, W) or lat.shape != (H, W):
        lon = upsample_tiegrid_to_full(lon, H, W)
        lat = upsample_tiegrid_to_full(lat, H, W)
        if debug_lines is not None:
            debug_lines.append(f"[tie] upsampled to {lon.shape[0]}x{lon.shape[1]}")

    return lon, lat


# ============================================================
# Directory / file discovery
# ============================================================

def find_data_dirs(folder_path: str) -> list[str]:
    """
    Discover all .data directories inside *folder_path*.

    Search order mirrors v3:
      1. .data dir paired with the first .dim file found.
      2. Any *.data at the top level.
      3. Any *.data one level deeper (sub-directories).
    """
    found: list[str] = []
    dim_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dim")]
    if dim_files:
        d = os.path.join(folder_path, os.path.splitext(dim_files[0])[0] + ".data")
        if os.path.isdir(d):
            found.append(d)

    for d in glob.glob(os.path.join(folder_path, "*.data")):
        if os.path.isdir(d) and d not in found:
            found.append(d)

    for sub in (
        os.path.join(folder_path, s)
        for s in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, s))
    ):
        for d in glob.glob(os.path.join(sub, "*.data")):
            if os.path.isdir(d) and d not in found:
                found.append(d)

    return found


def find_coherence_img(data_dir: str) -> Optional[str]:
    """
    Return the path to the coherence .img file inside *data_dir*.

    Coherence files written by SNAP's Interferogram operator are named
    ``coh_IW<n>_VV_<date1>_<date2>.img`` (after Deburst + Multilook the
    subswath tag is retained).  We match any .img whose lowercase basename
    starts with "coh" and does not contain "elevation".

    Returns None when no coherence file is found.
    """
    candidates = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".img")
        and f.lower().startswith("coh")
        and "elevation" not in f.lower()
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        # Prefer IW3 VV (the subswath processed by the batch script).
        # Fall back to the first alphabetical match.
        preferred = [c for c in candidates if "iw3" in os.path.basename(c).lower()
                     and "vv" in os.path.basename(c).lower()]
        return preferred[0] if preferred else sorted(candidates)[0]
    return candidates[0]


def list_target_imgs(data_dir: str) -> list[str]:
    """
    Return all .img files in *data_dir* that should be coherence-masked.

    Targets: i_ifg_*, q_ifg_*, Phase_ifg_*, coh_*, Intensity_ifg_*
    Skipped: elevation.*
    The coherence file itself is also masked (pixels with coh < threshold
    carry unreliable coherence estimates and should be NaN).
    """
    imgs = []
    for f in os.listdir(data_dir):
        fl = f.lower()
        if not fl.endswith(".img"):
            continue
        if "elevation" in fl:
            continue
        # Accept any recognised interferogram band
        if (
            fl.startswith("coh_iw")
            or fl.startswith("i_ifg_iw")
            or fl.startswith("q_ifg_iw")
            or fl.startswith("phase_ifg_iw")
            or fl.startswith("intensity_ifg_iw")
            # Generic fallbacks for non-standard naming
            or ("coh"       in fl and "iw" in fl and "vv" in fl)
            or ("i_ifg"     in fl and "iw" in fl and "vv" in fl)
            or ("q_ifg"     in fl and "iw" in fl and "vv" in fl)
            or ("phase_ifg" in fl and "iw" in fl and "vv" in fl)
            or ("intensity_ifg" in fl and "iw" in fl and "vv" in fl)
        ):
            imgs.append(os.path.join(data_dir, f))
    return imgs


# ============================================================
# Coherence mask construction
# ============================================================

def build_coherence_mask(
    coh_path: str,
    threshold: float = COH_THRESHOLD,
    debug_lines: Optional[list] = None,
) -> tuple[np.ndarray, dict]:
    """
    Read the coherence raster and return a boolean mask.

    Parameters
    ----------
    coh_path : str
        Path to the coherence .img file.
    threshold : float
        Pixels with coh < threshold are masked (set to True in the mask).
    debug_lines : list, optional
        Accumulates diagnostic strings when provided.

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype bool
        True  → pixel will be set to NaN  (coherence below threshold)
        False → pixel is retained
    coh_hdr_meta : dict
        Parsed header metadata for the coherence file (used for dimension
        cross-checks against target files).

    Raises
    ------
    RuntimeError
        If the .hdr file is missing, unreadable, or the array is empty.
    """
    hdr_path = coh_path[:-4] + ".hdr"
    if not os.path.isfile(hdr_path):
        raise RuntimeError(
            f"Coherence header not found: {hdr_path}"
        )

    coh_hdr_meta = parse_envi_hdr(hdr_path)
    coh_arr      = read_envi_array(coh_path, coh_hdr_meta)

    # Coherence is single-band; drop a leading band dimension if present
    if coh_arr.ndim == 3:
        if coh_arr.shape[0] == 1:
            coh_arr = coh_arr[0]
        else:
            raise RuntimeError(
                f"Unexpected multi-band coherence file ({coh_arr.shape}): {coh_path}"
            )

    if coh_arr.size == 0:
        raise RuntimeError(f"Coherence array is empty: {coh_path}")

    # Pixels flagged by SNAP as no-data (exactly 0.0) are treated as low
    # coherence and will be masked.  NaN in the coherence array (rare but
    # possible after earlier masking passes) is also masked.
    mask = (coh_arr < threshold) | ~np.isfinite(coh_arr)

    if debug_lines is not None:
        n_total   = mask.size
        n_masked  = int(mask.sum())
        n_nodata  = int(~np.isfinite(coh_arr).sum())
        coh_min   = float(np.nanmin(coh_arr)) if np.any(np.isfinite(coh_arr)) else float("nan")
        coh_max   = float(np.nanmax(coh_arr)) if np.any(np.isfinite(coh_arr)) else float("nan")
        coh_mean  = float(np.nanmean(coh_arr)) if np.any(np.isfinite(coh_arr)) else float("nan")
        debug_lines.append(
            f"[coh] file={os.path.basename(coh_path)}"
        )
        debug_lines.append(
            f"[coh] dims={coh_arr.shape[0]}x{coh_arr.shape[1]}  "
            f"min={coh_min:.4f}  max={coh_max:.4f}  mean={coh_mean:.4f}"
        )
        debug_lines.append(
            f"[coh] threshold={threshold}  "
            f"masked={n_masked:,}/{n_total:,}  "
            f"nodata_nan={n_nodata:,}"
        )

    return mask.astype(np.bool_), coh_hdr_meta


# ============================================================
# Single-file masking  (atomic write, .bak backup)
# ============================================================

def mask_file(
    img_path: str,
    mask_array: np.ndarray,
    dry_run: bool = False,
) -> tuple[str, int, int, str]:
    """
    Apply *mask_array* to one SNAP ENVI .img raster in-place.

    Masked pixels (mask_array == True) are set to NaN.
    The original file is backed up to <img>.bak before the first write.
    The new data is written to <img>.tmp and then atomically renamed.
    The .hdr is updated to datatype=4 (float32) if the original was different.

    Parameters
    ----------
    img_path   : path to the target .img file.
    mask_array : 2-D boolean array; True → set to NaN.
    dry_run    : if True, skip all I/O and return statistics only.

    Returns
    -------
    (basename, n_masked_pixels, n_total_pixels, status_string)
    """
    basename = os.path.basename(img_path)
    hdr_path = img_path[:-4] + ".hdr"

    if not os.path.isfile(hdr_path):
        return (basename, 0, 0, "missing_hdr")

    # ---- read ----
    try:
        hdr_meta = parse_envi_hdr(hdr_path)
        data     = read_envi_array(img_path, hdr_meta)
    except Exception as exc:
        return (basename, 0, 0, f"read_error: {exc}")

    # ---- dimension check ----
    file_hw = (data.shape[-2], data.shape[-1])
    mask_hw = (mask_array.shape[-2], mask_array.shape[-1])
    if file_hw != mask_hw:
        return (
            basename, 0, 0,
            f"shape_mismatch: file={file_hw} mask={mask_hw}",
        )

    n_masked = int(mask_array.sum())
    n_total  = mask_array.size

    if dry_run:
        return (basename, n_masked, n_total, "dry_run")

    # ---- prepare writable float32 copy ----
    mask_bool = np.asarray(mask_array, dtype=np.bool_)

    if data.dtype != np.float32 or not data.flags.writeable:
        data = data.astype(np.float32, copy=True)

    # ---- apply mask (band-aware) ----
    if data.ndim == 2:
        data[mask_bool] = np.nan
        bands_n, H, W = 1, data.shape[0], data.shape[1]
    else:
        bands_n, H, W = data.shape
        for b in range(bands_n):
            data[b][mask_bool] = np.nan

    # ---- output dtype: preserve byte order, always float32 ----
    bo        = hdr_meta.get("byte order", 0)
    out_dtype = np.dtype(np.float32).newbyteorder(">" if bo == 1 else "<")
    expected_bytes = H * W * bands_n * 4   # float32 = 4 bytes

    bak_path = img_path + ".bak"
    tmp_path = img_path + ".tmp"
    status   = "ok"

    try:
        # Backup the original once (idempotent on re-runs)
        if not os.path.isfile(bak_path):
            with open(img_path, "rb") as src, open(bak_path, "wb") as dst:
                dst.write(src.read())

        # Update .hdr datatype to float32 BEFORE writing the data file so
        # that the header is never out of sync with the binary if the process
        # is interrupted between the two writes.
        original_dt = hdr_meta.get("datatype", 4)
        if original_dt != 4:
            try:
                with open(hdr_path, "r", encoding="utf-8") as fh:
                    hdr_text = fh.read()
                hdr_text = re.sub(
                    r"(?im)^(data\s+type\s*=\s*)\d+",
                    r"\g<1>4",
                    hdr_text,
                )
                with open(hdr_path, "w", encoding="utf-8") as fh:
                    fh.write(hdr_text)
            except Exception as exc_hdr:
                return (
                    basename, 0, 0,
                    f"hdr_update_failed_before_write: {exc_hdr}",
                )

        # Write → validate byte count → atomic rename
        data.astype(out_dtype).tofile(tmp_path)

        actual_bytes = os.path.getsize(tmp_path)
        if actual_bytes != expected_bytes:
            status = (
                f"write_size_mismatch: wrote {actual_bytes} "
                f"expected {expected_bytes}"
            )
        else:
            os.replace(tmp_path, img_path)
            tmp_path = None   # replaced successfully — nothing to clean up

    except Exception as exc:
        status = f"write_error: {exc}"

    finally:
        if tmp_path is not None and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if not status.startswith("ok"):
        return (basename, 0, 0, status)
    return (basename, n_masked, n_total, status)


# ============================================================
# Per-.data-directory orchestration
# ============================================================

def process_data_dir(
    data_dir: str,
    threshold: float = COH_THRESHOLD,
    debug: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Build a coherence mask for *data_dir* and apply it to all target .img files.

    Returns a result dict with keys:
        data_dir  : str  — the directory processed
        status    : str  — "ok" | "no_targets" | "missing_coherence" |
                           "missing_coh_hdr" | "coh_read_error:<msg>" |
                           "no_valid_targets"
        files     : list of per-file dicts (file, masked, total, status)
        debug     : str  — diagnostic log (empty when debug=False)
        coh_file  : str  — basename of the coherence file used (or "")
    """
    debug_lines: list[str] = []
    debug_lines.append(f"[data_dir] {data_dir}")

    # ---- discover target files ----
    targets = list_target_imgs(data_dir)
    if not targets:
        return {
            "data_dir": data_dir,
            "status":   "no_targets",
            "files":    [],
            "debug":    "",
            "coh_file": "",
        }

    debug_lines.append(f"[targets] {len(targets)} file(s) found")

    # ---- locate coherence file ----
    coh_path = find_coherence_img(data_dir)
    if coh_path is None:
        debug_lines.append("[coh] ERROR: no coherence file found")
        return {
            "data_dir": data_dir,
            "status":   "missing_coherence",
            "files":    [],
            "debug":    "\n".join(debug_lines),
            "coh_file": "",
        }

    debug_lines.append(f"[coh] using {os.path.basename(coh_path)}")

    # ---- build coherence mask ----
    try:
        coh_mask, coh_hdr_meta = build_coherence_mask(
            coh_path, threshold=threshold, debug_lines=debug_lines
        )
    except RuntimeError as exc:
        debug_lines.append(f"[coh] ERROR: {exc}")
        status_key = (
            "missing_coh_hdr"
            if "header not found" in str(exc)
            else f"coh_read_error: {exc}"
        )
        return {
            "data_dir": data_dir,
            "status":   status_key,
            "files":    [],
            "debug":    "\n".join(debug_lines),
            "coh_file": os.path.basename(coh_path),
        }

    coh_shape = (coh_hdr_meta["lines"], coh_hdr_meta["samples"])
    debug_lines.append(f"[mask] shape={coh_shape[0]}x{coh_shape[1]}  "
                       f"masked_px={int(coh_mask.sum()):,}")

    # ---- apply mask to every target file ----
    reports = []
    any_ok  = False

    for img_path in sorted(targets):
        rep = mask_file(img_path, coh_mask, dry_run=dry_run)
        fname, n_masked, n_total, file_status = rep

        # Dimension mismatch: log prominently but continue with other files
        if "shape_mismatch" in file_status:
            debug_lines.append(
                f"[file] SKIP {fname}: {file_status}"
            )
        else:
            debug_lines.append(
                f"[file] {fname}: masked={n_masked:,}/{n_total:,}  status={file_status}"
            )
            any_ok = True

        reports.append({
            "file":    fname,
            "masked":  n_masked,
            "total":   n_total,
            "status":  file_status,
        })

    final_status = "ok" if any_ok else "no_valid_targets"
    return {
        "data_dir": data_dir,
        "status":   final_status,
        "files":    reports,
        "debug":    "\n".join(debug_lines) if debug else "",
        "coh_file": os.path.basename(coh_path),
    }


# ============================================================
# Per-folder orchestration  (called by worker process)
# ============================================================

def process_folder(
    folder_path: str,
    threshold: float = COH_THRESHOLD,
    debug: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Process all .data directories inside one yyyymmdd_yyyymmdd folder.

    This is the top-level function submitted to ProcessPoolExecutor.
    It is entirely self-contained (no shared state, no global initializer
    needed) because coherence masking requires no external data files.
    """
    folder_name = os.path.basename(folder_path)
    data_dirs   = find_data_dirs(folder_path)

    if not data_dirs:
        return {"folder": folder_name, "status": "no_data_dirs", "details": []}

    details = [
        process_data_dir(d, threshold=threshold, debug=debug, dry_run=dry_run)
        for d in data_dirs
    ]
    return {"folder": folder_name, "status": "ok", "details": details}


# ============================================================
# Summary reporting helpers
# ============================================================

def _write_debug_file(debug_text: str, data_dir: str, written: set) -> None:
    """Write a per-folder debug log file (idempotent across multiple .data dirs)."""
    parent   = os.path.dirname(data_dir)
    dbg_path = os.path.join(
        parent, f"{os.path.basename(parent)}_coh_mask_debug.txt"
    )
    if dbg_path in written or not debug_text:
        return
    try:
        with open(dbg_path, "w", encoding="utf-8") as fh:
            fh.write(debug_text + "\n")
        written.add(dbg_path)
    except Exception:
        pass


def _print_summary(results: list, dry_run: bool) -> None:
    """Print a structured summary to stdout after all workers finish."""
    print("\n=== SUMMARY ===")
    total_files = total_masked = total_pixels = 0
    n_ok = n_empty = n_error = 0

    for res in sorted(results, key=lambda x: x["folder"]):
        folder = res["folder"]
        status = res["status"]

        if status != "ok":
            n_error += 1
            print(
                f"{folder}: no .data directories found."
                if status == "no_data_dirs"
                else f"{folder}: error: {status}"
            )
            continue

        folder_files = folder_masked = folder_total = 0
        had_targets  = False
        dbg_written: set[str] = set()

        for d in res["details"]:
            data_dir = d.get("data_dir", "")

            if d["status"] == "ok":
                _write_debug_file(d.get("debug", ""), data_dir, dbg_written)

                for fr in d["files"]:
                    if fr["status"] in ("ok", "dry_run"):
                        had_targets    = True
                        folder_files  += 1
                        total_files   += 1
                        folder_masked += fr["masked"]
                        folder_total  += fr["total"]
                        total_masked  += fr["masked"]
                        total_pixels  += fr["total"]
                    else:
                        print(f"  {folder}/{fr['file']}: {fr['status']}")

            elif d["status"] == "no_targets":
                print(
                    f"{folder}: {os.path.basename(data_dir)} "
                    f"has no matching .img files."
                )
            elif d["status"] == "missing_coherence":
                print(
                    f"{folder}: {os.path.basename(data_dir)} "
                    f"— no coherence file found (skipped)."
                )
            else:
                _write_debug_file(d.get("debug", ""), data_dir, dbg_written)
                print(
                    f"{folder}: {os.path.basename(data_dir)} → {d['status']}"
                )

        if had_targets and folder_total > 0:
            n_ok  += 1
            pct    = 100.0 * folder_masked / folder_total
            label  = "[DRY-RUN] " if dry_run else ""
            print(
                f"{label}{folder}: {folder_files} file(s), "
                f"masked {folder_masked:,} px ({pct:.2f}%)"
            )
        else:
            n_empty += 1
            print(f"{folder}: no data processed.")

    print("\n--- Totals ---")
    print(f"Folders OK       : {n_ok}")
    print(f"Folders empty    : {n_empty}")
    print(f"Folders error    : {n_error}")
    print(f"Files processed  : {total_files}")
    if total_pixels > 0:
        print(
            f"Total masked     : {total_masked:,} / {total_pixels:,} "
            f"({100.0 * total_masked / total_pixels:.2f}%)"
        )
    print(
        "\n*** DRY-RUN complete — no files were modified. ***"
        if dry_run
        else "\nCompleted coherence masking for all folders."
    )


# ============================================================
# CLI entry point
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Coherence-threshold masking for SNAP InSAR ENVI rasters.\n"
            "Sets pixels with coherence < THRESHOLD to NaN across all "
            "interferogram band files (.img) in each .data directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            r"  python mask_coh_insar.py --base C:\bcosta\DS-InSAR" "\n"
            r"  python mask_coh_insar.py --base C:\bcosta\DS-InSAR "
            "--threshold 0.4 --workers 4 --dry-run\n"
        ),
    )
    ap.add_argument(
        "--base",
        default=r"C:\bcosta\DS-InSAR",
        help="Root folder containing yyyymmdd_yyyymmdd sub-folders.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=COH_THRESHOLD,
        help=(
            f"Coherence threshold (default: {COH_THRESHOLD}). "
            "Pixels with coh < threshold are set to NaN."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N folders (useful for testing).",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Write per-folder diagnostic logs alongside the .data directories.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report masking statistics without modifying any files.",
    )
    args = ap.parse_args()

    # ---- validate arguments ----
    if not os.path.isdir(args.base):
        print(f"ERROR: Base folder not found: {args.base}")
        sys.exit(1)

    if not (0.0 < args.threshold < 1.0):
        print(
            f"ERROR: --threshold must be in the open interval (0, 1); "
            f"got {args.threshold}"
        )
        sys.exit(1)

    if args.dry_run:
        print("*** DRY-RUN MODE — no files will be written ***")

    print(f"Coherence threshold : {args.threshold}")

    # ---- discover folders ----
    folders = [
        os.path.join(args.base, f)
        for f in sorted(os.listdir(args.base))
        if "_" in f and os.path.isdir(os.path.join(args.base, f))
    ]
    if args.limit:
        folders = folders[: args.limit]

    print(f"Found {len(folders)} InSAR folder(s).")

    # ---- parallel processing ----
    # No pool initializer needed: coherence masking is self-contained.
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {
            exe.submit(
                process_folder,
                folder,
                args.threshold,
                args.debug,
                args.dry_run,
            ): folder
            for folder in folders
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing folders",
        ):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({
                    "folder":  os.path.basename(futures[fut]),
                    "status":  f"worker_error: {exc}",
                    "details": [],
                })

    _print_summary(results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()