"""
03_load_phases.py
Loads all unwrapped phase bands and corresponding coherence bands from SNAP
BEAM-DIMAP stacks. Applies the coherence mask and builds phase and coherence
matrices ready for coherence-weighted SBAS inversion.

Networks are processed independently — outputs are prefixed with stack stem.

Outputs (per network, written to WORK_DIR):
- _phase_matrix.npy : float32 (n_ifgs × n_valid_pixels) — unwrapped phase [rad]
- _coh_matrix.npy   : float32 (n_ifgs × n_valid_pixels) — coherence [0-1]
- _pixel_coords.npy : int32   (n_valid_pixels × 2)       — row, col indices
- _ifg_pairs.csv    : IFG metadata — dates, baselines, NaN%
- _phase_load_report.txt : loading statistics and diagnostics

Phase sign convention (SNAP BEAM-DIMAP):
  phi_ifg = phi_secondary - phi_primary = (4*pi/lambda) * delta_R
  Positive phase => target moved AWAY from satellite => SUBSIDENCE.
  This is the standard SNAP/StaMPS radar-range convention.
  Sign conversion to geophysical (positive=uplift) is applied in
  script 05 (intervals_to_timeseries and compute_mean_velocity).
  Script 03 stores raw radar-convention phase — do NOT negate here.
"""

import os
import re
import csv
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from datetime import datetime

# -- Fix PROJ conflict ---------------------------------------------------------
_conda_prefix = Path(os.environ.get("CONDA_PREFIX", r"C:\\ProgramData\\anaconda3"))
_proj_data = _conda_prefix / "Library" / "share" / "proj"
if _proj_data.exists():
    os.environ["PROJ_DATA"] = str(_proj_data)
    os.environ["PROJ_LIB"] = str(_proj_data)
    os.environ["PROJ_NETWORK"] = "OFF"

# -- Paths ---------------------------------------------------------------------
WORK_DIR = Path(r"C:\\bcosta\\DS-InSAR\\tc")

# -- Configuration -------------------------------------------------------------
MASK_SUFFIX = "_coherence_mask.tif"
UNW_PREFIX = "Unw_Phase_ifg_"
COH_PREFIX = "coh_"
OUTPUT_NODATA = -9999.0

ENVI_DTYPE = {
    1: "u1",
    2: "i2",
    3: "i4",
    4: "f4",
    5: "f8",
    12: "u2",
}

SNAP_DATE_FMT = "%d%b%Y"

# -- Helpers -------------------------------------------------------------------

def read_envi_img(hdr_path, img_path, clamp_01=False):
    """
    Read ENVI .img using metadata from .hdr.
    Robust to spacing/case differences in SNAP-written headers.
    Returns float32 array.
    """
    meta = {}

    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            meta[key] = value

    try:
        samples = int(meta["samples"])
        lines = int(meta["lines"])
        data_type = int(meta.get("data type", 4))
        byte_order = int(meta.get("byte order", 0))
    except KeyError as e:
        raise ValueError(f"Missing ENVI header field {e} in {hdr_path}")

    endian = ">" if byte_order == 1 else "<"
    base_dtype = ENVI_DTYPE.get(data_type, "f4")
    np_dtype = np.dtype(f"{endian}{base_dtype}")

    data = np.fromfile(img_path, dtype=np_dtype)
    expected = samples * lines
    if data.size != expected:
        raise ValueError(f"{img_path.name}: expected {expected} values, got {data.size}")

    arr = data.reshape((lines, samples)).astype(np.float32)

    if clamp_01:
        arr = np.where(np.isfinite(arr) & (arr >= 0) & (arr <= 1), arr, np.nan)
    else:
        arr[~np.isfinite(arr)] = np.nan

    return arr


def load_mask(mask_path):
    """Load coherence mask GeoTIFF as boolean array using rasterio."""
    import rasterio
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask == 1


def find_bands(root, prefix):
    """Return list of {name, index} dicts for bands starting with prefix."""
    bands = []
    for sb in root.iter("Spectral_Band_Info"):
        name_el = sb.find("BAND_NAME")
        idx_el = sb.find("BAND_INDEX")
        if name_el is not None and name_el.text:
            name = name_el.text.strip()
            if name.startswith(prefix):
                bands.append({
                    "name": name,
                    "index": int(idx_el.text.strip()) if idx_el is not None else -1,
                })
    return bands


def parse_dates_from_band_name(band_name):
    """
    Extract first two dates from a SNAP band name.
    Works for both Unw_Phase_ifg_01Jun2019_13Jun2019
    and coh_IW3_VV_01Jun2019_13Jun2019 patterns.
    Returns (datetime, datetime) or (None, None).
    """
    matches = re.findall(r"\d{2}[A-Za-z]{3}\d{4}", band_name)
    if len(matches) >= 2:
        try:
            return (datetime.strptime(matches[0], SNAP_DATE_FMT),
                    datetime.strptime(matches[1], SNAP_DATE_FMT))
        except ValueError:
            pass
    return None, None


def match_coh_band(unw_name, coh_bands):
    """
    Find the coherence band corresponding to an unwrapped phase band by
    matching the primary and secondary dates.
    Returns the matching coh band dict or None.
    """
    d1, d2 = parse_dates_from_band_name(unw_name)
    if d1 is None:
        return None

    target_d1 = d1.strftime(SNAP_DATE_FMT)
    target_d2 = d2.strftime(SNAP_DATE_FMT)

    for cb in coh_bands:
        cd1, cd2 = parse_dates_from_band_name(cb["name"])
        if cd1 is None:
            continue
        if (cd1.strftime(SNAP_DATE_FMT) == target_d1 and
                cd2.strftime(SNAP_DATE_FMT) == target_d2):
            return cb
    return None


def validate_pixel_coords(valid_mask, pixel_coords, stem):
    nrows, ncols = valid_mask.shape
    ok = True
    messages = []

    if pixel_coords.ndim != 2 or pixel_coords.shape[1] != 2:
        return False, [f"pixel_coords has invalid shape {pixel_coords.shape}; expected (n_valid, 2)."]

    rows = pixel_coords[:, 0]
    cols = pixel_coords[:, 1]

    if np.any(rows < 0) or np.any(rows >= nrows):
        ok = False
        messages.append("row indices fall outside mask extent")
    if np.any(cols < 0) or np.any(cols >= ncols):
        ok = False
        messages.append("col indices fall outside mask extent")

    reconstructed = np.zeros_like(valid_mask, dtype=bool)
    reconstructed[rows, cols] = True

    same_mask = np.array_equal(reconstructed, valid_mask)
    mismatch_count = int(np.sum(reconstructed != valid_mask))
    if not same_mask:
        ok = False
        messages.append(f"reconstructed mask does not match valid_mask; mismatch pixels={mismatch_count}")

    messages.append(f"row min/max: {rows.min()} / {rows.max()}")
    messages.append(f"col min/max: {cols.min()} / {cols.max()}")
    messages.append(f"valid pixels in mask: {int(valid_mask.sum())}")
    messages.append(f"rows in pixel_coords: {pixel_coords.shape[0]}")
    messages.append(f"exact mask match [row,col]: {same_mask}")
    messages.append(f"mask diagnostic for {stem}: {'PASS' if ok else 'FAIL'}")

    return ok, messages


# -- Per-stack processing ------------------------------------------------------

def process_stack(dim_path):
    dim_path = Path(dim_path)
    data_dir = dim_path.with_suffix(".data")
    stem = dim_path.stem

    print("\n" + "=" * 60)
    print(f"Stack : {stem}")

    # -- Load coherence mask ---------------------------------------------------
    mask_path = WORK_DIR / f"{stem}{MASK_SUFFIX}"
    if not mask_path.exists():
        print(f"  ERROR: mask not found: {mask_path}")
        print("  Run 02_coherence_mask.py first. Aborting.")
        return

    valid_mask = load_mask(mask_path)
    nrows, ncols = valid_mask.shape
    n_valid = int(np.sum(valid_mask))
    print(f"  Mask loaded : {nrows} x {ncols} | valid pixels: {n_valid} ({n_valid/valid_mask.size*100:.1f}%)")

    pixel_rows, pixel_cols = np.where(valid_mask)
    pixel_coords = np.stack([pixel_rows, pixel_cols], axis=1).astype(np.int32)

    coords_ok, coord_msgs = validate_pixel_coords(valid_mask, pixel_coords, stem)
    print("  Pixel-coord diagnostic:")
    for msg in coord_msgs:
        print(f"    - {msg}")
    if not coords_ok:
        print("  ERROR: pixel coordinate validation failed. Aborting before export of corrupted coordinates.")
        return

    # -- Find bands ------------------------------------------------------------
    tree = ET.parse(dim_path)
    root = tree.getroot()
    unw_bands = find_bands(root, UNW_PREFIX)
    coh_bands = find_bands(root, COH_PREFIX)
    print(f"  UnwPhase bands  : {len(unw_bands)}")
    print(f"  Coherence bands : {len(coh_bands)}")

    if not unw_bands:
        print(f"  ERROR: no '{UNW_PREFIX}*' bands found. Aborting.")
        return

    # -- Load phases and coherence, build matrices -----------------------------
    phase_matrix = np.full((len(unw_bands), n_valid), np.nan, dtype=np.float32)
    coh_matrix = np.full((len(unw_bands), n_valid), np.nan, dtype=np.float32)
    ifg_records = []
    skipped = 0
    coh_missing = 0

    for i, b in enumerate(unw_bands):
        name = b["name"]
        hdr_path = data_dir / (name + ".hdr")
        img_path = data_dir / (name + ".img")

        if not img_path.exists():
            print(f"  WARNING: missing {img_path.name} - row {i} set to NaN")
            skipped += 1
            continue

        print(f"  Reading HDR     : {hdr_path.name}")
        phase_2d = read_envi_img(hdr_path, img_path, clamp_01=False)

        if phase_2d.shape != (nrows, ncols):
            print(f"  WARNING: shape mismatch {name} - skipping")
            skipped += 1
            continue

        phase_matrix[i, :] = phase_2d[valid_mask]

        cb = match_coh_band(name, coh_bands)
        if cb is not None:
            c_hdr = data_dir / (cb["name"] + ".hdr")
            c_img = data_dir / (cb["name"] + ".img")
            if c_img.exists():
                coh_2d = read_envi_img(c_hdr, c_img, clamp_01=True)
                if coh_2d.shape == (nrows, ncols):
                    coh_matrix[i, :] = coh_2d[valid_mask]
                else:
                    coh_missing += 1
            else:
                coh_missing += 1
        else:
            coh_missing += 1

        d1, d2 = parse_dates_from_band_name(name)
        temporal_baseline = (d2 - d1).days if d1 and d2 else None
        n_nan_phase = int(np.sum(np.isnan(phase_matrix[i, :])))
        n_nan_coh = int(np.sum(np.isnan(coh_matrix[i, :])))
        coh_mean = float(np.nanmean(coh_matrix[i, :]))

        ifg_records.append({
            "index": i,
            "band_name": name,
            "date_primary": d1.strftime("%Y%m%d") if d1 else "unknown",
            "date_secondary": d2.strftime("%Y%m%d") if d2 else "unknown",
            "temporal_baseline": temporal_baseline,
            "phase_nan_pct": round(n_nan_phase / n_valid * 100, 2),
            "coh_mean": round(coh_mean, 4),
            "coh_nan_pct": round(n_nan_coh / n_valid * 100, 2),
        })

        print(f"  [{i+1:02d}/{len(unw_bands)}] {name[:48]:<48} "
              f"dt={str(temporal_baseline)+'d':>6} "
              f"coh={coh_mean:.3f} "
              f"NaN={n_nan_phase/n_valid*100:.1f}%")

    # -- Summary ---------------------------------------------------------------
    loaded = len(unw_bands) - skipped
    phase_mean = float(np.nanmean(phase_matrix))
    phase_std = float(np.nanstd(phase_matrix))
    coh_mean_all = float(np.nanmean(coh_matrix))
    total_nan = float(np.sum(np.isnan(phase_matrix)) / phase_matrix.size * 100)

    print(f"\n  IFGs loaded        : {loaded} / {len(unw_bands)}")
    print(f"  Coh bands matched  : {loaded - coh_missing} / {loaded}")
    print(f"  Phase matrix       : {phase_matrix.shape} (IFGs x valid pixels)")
    print(f"  Phase range        : {np.nanmin(phase_matrix):.2f} - {np.nanmax(phase_matrix):.2f} rad")
    print(f"  Phase mean +/- std : {phase_mean:.3f} +/- {phase_std:.3f} rad")
    print(f"  Mean coherence     : {coh_mean_all:.3f}")
    print(f"  NaN in phase matrix: {total_nan:.1f}%")
    print("  Convention note    : positive phase = subsidence (SNAP radar convention)")
    print("                       sign flip to geophysical convention in script 05")

    # -- Save outputs ----------------------------------------------------------
    out_phase = WORK_DIR / f"{stem}_phase_matrix.npy"
    out_coh = WORK_DIR / f"{stem}_coh_matrix.npy"
    out_coords = WORK_DIR / f"{stem}_pixel_coords.npy"
    out_pairs = WORK_DIR / f"{stem}_ifg_pairs.csv"
    out_rep = WORK_DIR / f"{stem}_phase_load_report.txt"

    np.save(out_phase, phase_matrix)
    np.save(out_coh, coh_matrix)
    np.save(out_coords, pixel_coords)

    if ifg_records:
        with open(out_pairs, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ifg_records[0].keys()))
            writer.writeheader()
            writer.writerows(ifg_records)

    report = [
        f"Phase load report  : {stem}\n",
        f"Mask               : {mask_path.name}\n",
        f"Valid pixels       : {n_valid} ({n_valid/valid_mask.size*100:.1f}%)\n",
        f"Pixel coord check  : {'PASS' if coords_ok else 'FAIL'}\n",
    ]
    for msg in coord_msgs:
        report.append(f"  - {msg}\n")
    report.extend([
        f"IFGs loaded        : {loaded} / {len(unw_bands)}\n",
        f"Coh bands matched  : {loaded - coh_missing} / {loaded}\n",
        f"Phase matrix       : {phase_matrix.shape}\n",
        f"Phase range        : {np.nanmin(phase_matrix):.3f} - {np.nanmax(phase_matrix):.3f} rad\n",
        f"Phase mean +/- std : {phase_mean:.3f} +/- {phase_std:.3f} rad\n",
        f"Mean coherence     : {coh_mean_all:.3f}\n",
        f"NaN in phase matrix: {total_nan:.1f}%\n",
        f"\n",
        f"Sign convention    : SNAP radar convention stored in phase_matrix\n",
        f"                     positive phase = increased slant range = subsidence\n",
        f"                     Geophysical sign flip (positive=uplift) applied in 05_inversion.py\n",
    ])
    Path(out_rep).write_text("".join(report), encoding="utf-8")

    print(f"\n  Written: {out_phase.name}  ({phase_matrix.nbytes/1e6:.1f} MB)")
    print(f"  Written: {out_coh.name}    ({coh_matrix.nbytes/1e6:.1f} MB)")
    print(f"  Written: {out_coords.name}")
    print(f"  Written: {out_pairs.name}")
    print(f"  Written: {out_rep.name}")


# -- Entry point ---------------------------------------------------------------

def main():
    dim_files = sorted(WORK_DIR.glob("*.dim"))

    if not dim_files:
        print(f"No .dim files found in {WORK_DIR}")
        return

    print(f"Found {len(dim_files)} stack(s) in {WORK_DIR}")
    for dim in dim_files:
        process_stack(dim)

    print("\n" + "=" * 60)
    print("Phase loading complete.")


if __name__ == "__main__":
    main()
