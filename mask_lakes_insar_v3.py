"""
Lake masking for SNAP InSAR ENVI rasters (hybrid multiprocessing).

Per-folder pipeline:
1) Discover .data dir(s) and target .img rasters (skip elevation.*)
2) Build a per-folder mask ONCE, in priority:
   A) ENVI .hdr georef (map info + CRS)
   B) .dim georef (BEAM-DIMAP via GDAL)
   C) TIE-POINT geocoding (latitude/longitude) with BILINEAR UPSAMPLING
      + BBox clip of lakes to lon/lat scene extent
      + Optional fast mode downsampling (--fast)
3) Overwrite each target .img with NaN on lake pixels (atomic write)
"""

import re
import os
import sys
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from tqdm import tqdm

import rasterio
from rasterio.transform import from_origin, from_bounds, Affine
from rasterio.crs import CRS
from rasterio.features import geometry_mask, rasterize as rio_rasterize

from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# per-worker lake cache (populated by the pool initializer)
# ---------------------------------------------------------------------------
_lakes_cache: dict = {}  # maps EPSG int -> GeoDataFrame; 'wgs84' -> GDF(4326)


def _worker_init(lake_shp: str) -> None:
    """Load and cache the lake shapefile once per worker process."""
    global _lakes_cache
    _lakes_cache["wgs84"] = gpd.read_file(lake_shp).to_crs(epsg=4326)


def _get_lakes_for_crs(crs: CRS) -> gpd.GeoDataFrame:
    """Return the lake GDF reprojected to *crs*, caching by EPSG."""
    global _lakes_cache
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None

    if epsg is None:
        return _lakes_cache["wgs84"].to_crs(crs)

    if epsg not in _lakes_cache:
        _lakes_cache[epsg] = _lakes_cache["wgs84"].to_crs(epsg=epsg)
    return _lakes_cache[epsg]


# ---------------- ENVI header parsing ----------------
def parse_envi_hdr(hdr_path):
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # FIX-1: was r"\\{([^}]*)\\}" which matched literal \{...\} — corrected to \{...\}
    content = re.sub(
        r"\{([^}]*)\}",
        lambda m: "{" + " ".join(m.group(1).split()) + "}",
        content,
        flags=re.DOTALL,
    )

    meta = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().lower()
        v = v.strip().strip("{}")
        meta[k] = v

    meta["samples"] = int(meta["samples"])
    meta["lines"] = int(meta["lines"])
    meta["bands"] = int(meta.get("bands", 1))
    meta["datatype"] = int(meta.get("data type", meta.get("datatype", 4)))
    meta["byte order"] = int(meta.get("byte order", 0))  # 0=little, 1=big
    return meta


def envi_dtype_to_numpy(dt_code):
    table = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
    }
    return table.get(dt_code, np.float32)


def read_envi_array(img_path, hdr_meta):
    dt_code = hdr_meta["datatype"]
    base = envi_dtype_to_numpy(dt_code)
    bo = hdr_meta.get("byte order", 0)
    dt = np.dtype(base).newbyteorder(">" if bo == 1 else "<")
    samples = hdr_meta["samples"]
    lines = hdr_meta["lines"]
    bands = hdr_meta.get("bands", 1)
    count = samples * lines * bands
    shape = (lines, samples) if bands == 1 else (bands, lines, samples)

    with open(img_path, "rb") as f:
        arr = np.fromfile(f, dtype=dt, count=count).reshape(shape)

    if base == np.float32 and bo == 0:
        return arr.view(np.float32)
    return arr.astype(np.float32)


# ---------------- map info / CRS helpers ----------------
def parse_map_info(map_info_str):
    parts = [p.strip() for p in map_info_str.strip("{}").split(",")]
    if len(parts) < 6:
        raise ValueError("map info has fewer than 6 fields.")
    d = {}
    d["projection_name"] = parts[0]
    d["ref_x"] = float(parts[1])
    d["ref_y"] = float(parts[2])
    d["tie_x"] = float(parts[3])
    d["tie_y"] = float(parts[4])
    d["px_x"] = float(parts[5])
    d["px_y"] = float(parts[6]) if len(parts) >= 7 else d["px_x"]
    d["zone"] = None
    d["hemisphere"] = None
    d["datum"] = None
    if len(parts) >= 8:
        try:
            d["zone"] = int(parts[7])
        except Exception:
            pass
    if len(parts) >= 9:
        d["hemisphere"] = parts[8]
    if len(parts) >= 10:
        d["datum"] = parts[9]
    return d


def transform_from_map_info(mi):
    west = mi["tie_x"] - (mi["ref_x"] - 0.5) * mi["px_x"]
    north = mi["tie_y"] + (mi["ref_y"] - 0.5) * mi["px_y"]
    return from_origin(west, north, mi["px_x"], mi["px_y"])


def crs_from_hdr(meta, mi):
    css = meta.get("coordinate system string")
    if css and css.strip():
        try:
            return CRS.from_wkt(css)
        except Exception:
            pass
    pinfo = meta.get("projection info")
    if pinfo and pinfo.strip():
        try:
            return CRS.from_wkt(pinfo)
        except Exception:
            pass
    proj_name = (mi.get("projection_name") or "").lower()
    datum = (mi.get("datum") or "").lower()
    zone = mi.get("zone")
    hemi = (mi.get("hemisphere") or "").lower()
    if "utm" in proj_name and zone is not None and "wgs" in datum:
        epsg = (32700 + zone) if "south" in hemi else (32600 + zone)
        try:
            return CRS.from_epsg(epsg)
        except Exception:
            return None
    return None


def build_grid_from_hdr(hdr_path):
    meta = parse_envi_hdr(hdr_path)
    width = meta["samples"]
    height = meta["lines"]
    mi_str = meta.get("map info")
    if not mi_str:
        raise RuntimeError("Header has no 'map info' -> cannot build transform.")
    mi = parse_map_info(mi_str)
    transform = transform_from_map_info(mi)
    crs = crs_from_hdr(meta, mi)
    if crs is None:
        raise RuntimeError(
            "Unable to determine CRS from header (no WKT and cannot infer UTM EPSG)."
        )
    return width, height, transform, crs, meta


# ---------------- .dim fallback ----------------
def guess_dim_for_data_dir(data_dir):
    parent = os.path.dirname(data_dir)
    base = os.path.basename(data_dir)
    stem = base[:-5] if base.lower().endswith(".data") else base
    cand = os.path.join(parent, stem + ".dim")
    if os.path.isfile(cand):
        return cand
    dims = [
        os.path.join(parent, f)
        for f in os.listdir(parent)
        if f.lower().endswith(".dim")
    ]
    return dims[0] if dims else None


def grid_from_dim(dim_path):
    with rasterio.open(dim_path) as ds:
        transform = ds.transform
        crs = ds.crs
        width = ds.width
        height = ds.height
    if crs is None:
        raise RuntimeError("DIM has no CRS.")
    if (
        isinstance(transform, Affine)
        and transform.a == 1
        and transform.e == 1
        and transform.b == 0
        and transform.c == 0
        and transform.d == 0
        and transform.f == 0
    ):
        raise RuntimeError("DIM has identity transform (not georeferenced).")
    return width, height, transform, crs


# ---------------- tie-point fallback (bilinear upsample) ----------------
def upsample_tiegrid_to_full(grid_tie, H, W):
    """Bilinear upsample a small tie grid (h_tie x w_tie) to (H x W)."""
    h_tie, w_tie = grid_tie.shape
    y_tie = np.linspace(0, H - 1, h_tie)
    x_tie = np.linspace(0, W - 1, w_tie)

    interp = RegularGridInterpolator(
        (y_tie, x_tie),
        grid_tie.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    y_out = np.arange(H, dtype=np.float64)
    x_out = np.arange(W, dtype=np.float64)
    yy, xx = np.meshgrid(y_out, x_out, indexing="ij")
    pts = np.stack([yy.ravel(), xx.ravel()], axis=-1)

    return interp(pts).reshape(H, W).astype(np.float32)


def read_lonlat_from_tiepoints(data_dir, ref_hdr_meta, debug_lines=None):
    tpg = os.path.join(data_dir, "tie_point_grids")
    lon_path = os.path.join(tpg, "longitude.img")
    lat_path = os.path.join(tpg, "latitude.img")
    lon_hdr = lon_path[:-4] + ".hdr"
    lat_hdr = lat_path[:-4] + ".hdr"

    if not (
        os.path.isfile(lon_path)
        and os.path.isfile(lat_path)
        and os.path.isfile(lon_hdr)
        and os.path.isfile(lat_hdr)
    ):
        raise RuntimeError(
            "Tie-point files latitude.img/longitude.img (+ .hdr) not found."
        )

    lon_meta = parse_envi_hdr(lon_hdr)
    lat_meta = parse_envi_hdr(lat_hdr)

    lon = read_envi_array(lon_path, lon_meta)
    lat = read_envi_array(lat_path, lat_meta)

    H, W = ref_hdr_meta["lines"], ref_hdr_meta["samples"]
    h_tie, w_tie = lon.shape

    if debug_lines is not None:
        debug_lines.append(f"[tie] raw tie dims={h_tie}x{w_tie}, ref dims={H}x{W}")

    if lon.shape != (H, W) or lat.shape != (H, W):
        lon = upsample_tiegrid_to_full(lon, H, W)
        lat = upsample_tiegrid_to_full(lat, H, W)
        if debug_lines is not None:
            debug_lines.append(f"[tie] upsampled to {lon.shape[0]}x{lon.shape[1]}")

    return lon, lat


def rasterize_lakes_on_lonlat(
    lakes_wgs84: gpd.GeoDataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    debug_lines=None,
):
    H, W = lon.shape
    min_lon = float(np.nanmin(lon))
    max_lon = float(np.nanmax(lon))
    min_lat = float(np.nanmin(lat))
    max_lat = float(np.nanmax(lat))

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, W, H)

    shapes = [
        (geom, 1)
        for geom in lakes_wgs84.geometry
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        if debug_lines is not None:
            debug_lines.append("[tie] lake_pixels=0 (no shapes)")
        return np.zeros((H, W), dtype=bool)

    burned = rio_rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    mask = burned.astype(bool)

    if debug_lines is not None:
        debug_lines.append(f"[tie] lake_pixels={int(mask.sum())}")
    return mask


# ---------------- discovery & file lists ----------------
def find_data_dirs(folder_path):
    found = []
    dim_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dim")]
    if dim_files:
        base = os.path.splitext(dim_files[0])[0]
        d = os.path.join(folder_path, f"{base}.data")
        if os.path.isdir(d):
            found.append(d)
    for d in glob.glob(os.path.join(folder_path, "*.data")):
        if os.path.isdir(d) and d not in found:
            found.append(d)
    for sub in [os.path.join(folder_path, s) for s in os.listdir(folder_path)]:
        if os.path.isdir(sub):
            for d in glob.glob(os.path.join(sub, "*.data")):
                if os.path.isdir(d) and d not in found:
                    found.append(d)
    return found


def list_target_imgs(data_dir):
    imgs = []
    for f in os.listdir(data_dir):
        fl = f.lower()
        if not fl.endswith(".img"):
            continue
        if "elevation" in fl:
            continue
        if (
            fl.startswith("coh_iw")
            or fl.startswith("i_ifg_iw")
            or fl.startswith("q_ifg_iw")
            or ("coh" in fl and "iw" in fl and "vv" in fl)
            or ("i_ifg" in fl and "iw" in fl and "vv" in fl)
            or ("q_ifg" in fl and "iw" in fl and "vv" in fl)
        ):
            imgs.append(os.path.join(data_dir, f))
    return imgs


# ---------------- masking one file ----------------
def mask_file(img_path, mask_array, dry_run=False):
    """
    Overwrite a SNAP ENVI .img raster in-place, setting lake pixels to NaN.

    - Single-band (H, W) and multi-band (bands, H, W) layouts both supported.
    - Output is always float32 with original byte order preserved.
    - Original file is backed up to .bak before any write.
    - Atomic write via .tmp + os.replace().
    """
    hdr_path = img_path[:-4] + ".hdr"
    if not os.path.exists(hdr_path):
        return (os.path.basename(img_path), 0, 0, "missing_hdr")

    try:
        hdr_meta = parse_envi_hdr(hdr_path)
        data = read_envi_array(img_path, hdr_meta)
    except Exception as e:
        return (os.path.basename(img_path), 0, 0, f"read_error: {e}")

    # Shape check
    file_shape = data.shape[-2], data.shape[-1]
    mask_shape = mask_array.shape[-2], mask_array.shape[-1]
    if file_shape != mask_shape:
        return (
            os.path.basename(img_path),
            0,
            0,
            f"shape_mismatch: file={file_shape} mask={mask_shape}",
        )

    masked_pixels = int(mask_array.sum())
    total_pixels = int(mask_array.size)

    if dry_run:
        return (os.path.basename(img_path), masked_pixels, total_pixels, "dry_run")

    # Normalise mask to numpy bool_ (handles geometry_mask numpy.bool_ output)
    mask_array = np.asarray(mask_array, dtype=np.bool_)

    # Ensure float32 writable working array
    if data.dtype != np.float32 or not data.flags.writeable:
        data = data.astype(np.float32, copy=True)

    if not np.issubdtype(data.dtype, np.floating):
        return (
            os.path.basename(img_path),
            0,
            0,
            f"invalid_data_dtype_after_cast: {data.dtype}",
        )

    # Apply mask: lake pixels -> NaN (band-aware)
    if data.ndim == 2:
        data[mask_array] = np.nan
        bands_for_bytes = 1
        H, W = data.shape
    else:
        bands_for_bytes, H, W = data.shape
        for b in range(bands_for_bytes):
            data[b][mask_array] = np.nan

    # Preserve original endianness, always write float32
    bo = hdr_meta.get("byte order", 0)
    out_dtype = np.dtype(np.float32).newbyteorder(">" if bo == 1 else "<")

    if data.dtype != np.float32:
        return (
            os.path.basename(img_path),
            0,
            0,
            f"prewrite_data_dtype_mismatch: {data.dtype}",
        )

    expected_bytes = H * W * bands_for_bytes * 4  # float32 = 4 bytes per element

    tmp_path = img_path + ".tmp"
    status = "ok"

    bak_path = img_path + ".bak"
    try:
        # Backup original once
        if not os.path.exists(bak_path):
            with open(img_path, "rb") as src, open(bak_path, "wb") as dst:
                dst.write(src.read())

        # Update header datatype to 4 (float32) BEFORE writing data
        # FIX-2/3: corrected regex — \s and \d are proper escapes; \g<1> is back-ref
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
            except Exception as e_hdr:
                return (
                    os.path.basename(img_path),
                    0,
                    0,
                    f"hdr_update_failed_before_write: {e_hdr}",
                )

        # Write to tmp, validate byte count, then atomic replace
        data.astype(out_dtype).tofile(tmp_path)

        actual_bytes = os.path.getsize(tmp_path)
        if actual_bytes != expected_bytes:
            status = f"write_size_mismatch: wrote {actual_bytes} expected {expected_bytes}"
        else:
            os.replace(tmp_path, img_path)
            tmp_path = None

    except Exception as e:
        status = f"write_error: {e}"

    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if not status.startswith("ok"):
        return (os.path.basename(img_path), 0, 0, status)

    return (os.path.basename(img_path), masked_pixels, total_pixels, status)


# ---------------- process one .data dir ----------------
def process_data_dir(data_dir, debug=False, fast=1, dry_run=False):
    targets = list_target_imgs(data_dir)
    if not targets:
        return {"data_dir": data_dir, "status": "no_targets", "files": [], "debug": ""}

    ref_img = targets[0]
    ref_hdr = ref_img[:-4] + ".hdr"
    if not os.path.exists(ref_hdr):
        return {
            "data_dir": data_dir,
            "status": "missing_ref_hdr",
            "files": [],
            "debug": "",
        }

    debug_lines = []
    debug_lines.append(f"[data_dir] {data_dir}")
    debug_lines.append(f"[ref_img] {os.path.basename(ref_img)}")

    hdr_meta = parse_envi_hdr(ref_hdr)
    width_hdr = hdr_meta["samples"]
    height_hdr = hdr_meta["lines"]

    georef_source = None
    transform = None
    crs = None
    mask = None

    # A) ENVI header georef (highest priority)
    try:
        width, height, transform, crs, hdr_meta_checked = build_grid_from_hdr(ref_hdr)
        if width != width_hdr or height != height_hdr:
            debug_lines.append(
                f"[warn] header dims ({width}x{height}) != data dims "
                f"({width_hdr}x{height_hdr}); using data dims."
            )
            width, height = width_hdr, height_hdr
        hdr_meta = hdr_meta_checked
        georef_source = "hdr"
        debug_lines.append("[grid] source=hdr (map info present)")

        lakes = _get_lakes_for_crs(crs)
        mask = geometry_mask(
            lakes.geometry,
            out_shape=(height, width),
            transform=transform,
            invert=True,
        )
        debug_lines.append(f"[mask] lake_pixels={int(mask.sum())}")

    except Exception as e_hdr:
        debug_lines.append(f"[grid] header georef failed: {e_hdr}")

    # B) .dim georef fallback (only if A failed)
    # FIX-LOGIC: added "mask is None" guard so HDR result is not overwritten
    dim_path = guess_dim_for_data_dir(data_dir)
    if mask is None and dim_path and os.path.isfile(dim_path):
        debug_lines.append(
            f"[grid] trying DIM fallback: {os.path.basename(dim_path)}"
        )
        try:
            w_dim, h_dim, transform, crs = grid_from_dim(dim_path)
            width, height = width_hdr, height_hdr

            lakes = _get_lakes_for_crs(crs)
            mask = geometry_mask(
                lakes.geometry,
                out_shape=(height, width),
                transform=transform,
                invert=True,
            )
            georef_source = "dim"
            debug_lines.append("[grid] source=dim (BEAM-DIMAP)")
            debug_lines.append(
                f"[dim] dims={w_dim}x{h_dim}, [img] dims={width_hdr}x{height_hdr}"
            )
            debug_lines.append(f"[mask] lake_pixels={int(mask.sum())}")
        except Exception as e_dim:
            debug_lines.append(f"[grid] DIM fallback failed: {e_dim}")

    # C) Tie-point fallback (only if A and B both failed)
    if mask is None:
        try:
            lakes_all = _lakes_cache["wgs84"]

            lon, lat = read_lonlat_from_tiepoints(
                data_dir, hdr_meta, debug_lines=debug_lines
            )
            georef_source = "tie"
            debug_lines.append(f"[tie] lon/lat shape={lon.shape}")

            min_lon = float(np.nanmin(lon))
            max_lon = float(np.nanmax(lon))
            min_lat = float(np.nanmin(lat))
            max_lat = float(np.nanmax(lat))
            aoi_bbox = box(min_lon, min_lat, max_lon, max_lat)

            idx = list(lakes_all.sindex.intersection(aoi_bbox.bounds))
            lakes = lakes_all.iloc[idx]
            debug_lines.append(f"[tie] lakes after bbox clip: {len(lakes)} polys")

            f = max(1, int(fast))
            if f > 1:
                lon_c = lon[::f, ::f]
                lat_c = lat[::f, ::f]
                debug_lines.append(f"[tie] fast={f} -> coarse grid {lon_c.shape}")
                mask_c = rasterize_lakes_on_lonlat(
                    lakes, lon_c, lat_c, debug_lines=debug_lines
                )
                mask = np.repeat(
                    np.repeat(mask_c, f, axis=0), f, axis=1
                )[: lon.shape[0], : lon.shape[1]]
            else:
                mask = rasterize_lakes_on_lonlat(
                    lakes, lon, lat, debug_lines=debug_lines
                )

        except Exception as e_tie:
            debug_lines.append(f"[tie] fallback failed: {e_tie}")
            return {
                "data_dir": data_dir,
                "status": f"georef_error: {e_tie}",
                "files": [],
                "debug": "\n".join(debug_lines),
            }

    if crs is not None:
        try:
            crs_epsg = crs.to_epsg()
        except Exception:
            crs_epsg = None
        debug_lines.append(f"[crs] {crs.to_string()} (EPSG={crs_epsg})")
    else:
        debug_lines.append("[crs] (tie-point mode)")

    debug_lines.append(f"[dims] width={width_hdr}, height={height_hdr}")
    debug_lines.append(f"[source] georef={georef_source}")

    # SNAP stores rasters south-to-north (row 0 = southernmost lat).
    # from_bounds and geometry_mask both produce masks north-to-south,
    # so the mask must be flipped vertically to align with the raster.
    mask = np.flipud(mask)

    reports = []
    for img in targets:
        rep = mask_file(img, mask, dry_run=dry_run)
        reports.append(
            {
                "file": rep[0],
                "masked": rep[1],
                "total": rep[2],
                "status": rep[3],
                "source": georef_source,
            }
        )

    return {
        "data_dir": data_dir,
        "status": "ok",
        "files": reports,
        "debug": "\n".join(debug_lines),
    }


# ---------------- process one high-level folder ----------------
def process_folder(folder_path, debug=False, fast=1, dry_run=False):
    folder_name = os.path.basename(folder_path)
    data_dirs = find_data_dirs(folder_path)
    if not data_dirs:
        return {"folder": folder_name, "status": "no_data_dirs", "details": []}

    details = []
    for d in data_dirs:
        details.append(
            process_data_dir(d, debug=debug, fast=fast, dry_run=dry_run)
        )
    return {"folder": folder_name, "status": "ok", "details": details}


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Mask lakes in SNAP InSAR ENVI rasters (hdr->dim->tie fallback; "
            "upsampling; bbox clip; fast mode)."
        )
    )
    ap.add_argument(
        "--base",
        default=r"C:\bcosta\DS-InSAR",       # FIX-4: single backslash in r-string
        help="Root folder with yyyymmdd_yyyymmdd subfolders",
    )
    ap.add_argument(
        "--lakes",
        default=r"E:\GIS\canvec_50K_NT_Hydro_shp\waterbody_2_2.shp",  # FIX-5
        help="Lake shapefile (polygons)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: all logical CPUs)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N folders",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Write per-folder debug diagnostics",
    )
    ap.add_argument(
        "--fast",
        type=int,
        default=1,
        help=(
            "Downsample factor for tie-point rasterization (1=no downsample; "
            "try 2/4/8). Note: introduces up to FACTOR pixels of fringe error "
            "at lake edges."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate masking without writing any files (safe preview)",
    )
    args = ap.parse_args()

    base_dir = args.base
    lake_shp = args.lakes
    fast = max(1, int(args.fast))
    dry_run = args.dry_run

    if not os.path.isdir(base_dir):
        print(f"ERROR: Base folder not found: {base_dir}")
        sys.exit(1)
    if not os.path.isfile(lake_shp):
        print(f"ERROR: Lake shapefile not found: {lake_shp}")
        sys.exit(1)

    if dry_run:
        print("*** DRY-RUN MODE — no files will be written ***")

    folders = []
    for f in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, f)
        if os.path.isdir(full) and "_" in f:
            folders.append(full)
    if args.limit:
        folders = folders[: args.limit]

    print(f"Found {len(folders)} InSAR folders.")

    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(lake_shp,),
    ) as exe:
        futures = {
            exe.submit(process_folder, f, args.debug, fast, dry_run): f
            for f in folders
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Processing folders"
        ):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append(
                    {
                        "folder": os.path.basename(futures[fut]),
                        "status": f"worker_error: {e}",
                        "details": [],
                    }
                )

    # ----- summary + debug logs -----
    print("\n=== SUMMARY ===")
    total_files = total_masked = total_pixels = 0
    n_ok = n_empty = n_error = 0

    for res in sorted(results, key=lambda x: x["folder"]):
        folder = res["folder"]
        status = res["status"]

        if status != "ok":
            n_error += 1
            if status == "no_data_dirs":
                print(f"{folder}: no .data directories found.")
            else:
                print(f"{folder}: error: {status}")
            continue

        folder_files = folder_masked = folder_total = 0
        had_targets = False
        dbg_written = set()

        for d in res["details"]:
            data_dir = d.get("data_dir", "")
            parent = os.path.dirname(data_dir)

            def _write_debug(dbg_text):
                dbg_path = os.path.join(
                    parent, f"{os.path.basename(parent)}_mask_debug.txt"
                )
                if dbg_path not in dbg_written and dbg_text:
                    try:
                        with open(dbg_path, "w", encoding="utf-8") as df:
                            df.write(dbg_text + "\n")
                        dbg_written.add(dbg_path)
                    except Exception:
                        pass

            if d["status"] == "ok":
                if args.debug:
                    _write_debug(d.get("debug", ""))

                for fr in d["files"]:
                    had_targets = True
                    folder_files += 1
                    total_files += 1
                    folder_masked += fr["masked"]
                    folder_total += fr["total"]
                    total_masked += fr["masked"]
                    total_pixels += fr["total"]

                    if fr["status"] not in ("ok", "dry_run"):
                        print(f"  {folder}/{fr['file']}: {fr['status']}")

            elif d["status"] == "no_targets":
                print(
                    f"{folder}: {os.path.basename(data_dir)} has no matching .img files."
                )
            else:
                if args.debug:
                    _write_debug(d.get("debug", ""))
                print(f"{folder}: {os.path.basename(data_dir)} -> {d['status']}")

        if had_targets and folder_total > 0:
            n_ok += 1
            pct = 100.0 * folder_masked / folder_total
            label = "[DRY-RUN] " if dry_run else ""
            print(
                f"{label}{folder}: processed {folder_files} file(s), "
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
    if dry_run:
        print("\n*** DRY-RUN complete — no files were modified. ***")
    else:
        print("\nCompleted masking for all folders.")


if __name__ == "__main__":
    main()
