"""
00_prepare_temperature.py  (v2.3)

FIXES vs v2.2:
  - ERA5-Land CDS files use "valid_time" as the time coordinate, not "time".
    xr.concat on "time" when the actual dim is "valid_time" creates a
    RangeIndex -> resample fails with TypeError.
    Fix: detect the actual time coordinate name, rename to "time" before concat,
    then assign a proper DatetimeIndex so resample works correctly.
  - xr.concat FutureWarning about join: now passes join="override" explicitly.
  - Handles both "time" and "valid_time" coordinate names automatically.
  - Handles single-level and multi-level coordinate structures from CDS.
"""

import csv
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────
ERA5_DIR = Path(r"C:\bcosta\DS-InSAR\era5")
NC_DIR   = ERA5_DIR / "nc_raw"
ERA5_DIR.mkdir(parents=True, exist_ok=True)

TUK_LAT =  69.44
TUK_LON = -133.03

DATE_START = datetime(2019, 1, 1)
DATE_END   = datetime(2024, 12, 31)

VARIABLES = [
    ("t2m",  "era5land_t2m",  "2m air temperature"),
    ("stl1", "era5land_stl1", "Soil temperature layer 1 (0-7 cm)"),
]

OUT_GRID_PATTERN = ERA5_DIR / "era5land_{var}_daily_grid.nc"
OUT_POINT_CSV    = ERA5_DIR / "tuk_t2m_daily.csv"
OUT_REPORT       = ERA5_DIR / "temperature_prep_report.txt"
SHORT_GAP_DAYS   = 5


# ── Format detection ───────────────────────────────────────────────────────────

def detect_file_format(nc_path):
    with open(nc_path, "rb") as f:
        magic = f.read(8)
    if len(magic) < 4:
        return "unknown"
    b = list(magic[:8])
    if b[0] == 0x89 and b[1] == 0x48 and b[2] == 0x44 and b[3] == 0x46:
        return "netcdf4"
    if b[0] == 0x43 and b[1] == 0x44 and b[2] == 0x46 and b[3] in (0x01, 0x02):
        return "netcdf3"
    if b[0] == 0x47 and b[1] == 0x52 and b[2] == 0x49 and b[3] == 0x42:
        return "grib"
    hex_str = " ".join(f"0x{x:02X}" for x in b)
    print(f"   [DIAG] First 8 bytes: {hex_str}")
    return "unknown"

def _list_xarray_engines():
    try:
        from xarray.backends import list_engines
        return list(list_engines().keys())
    except Exception:
        return []

def open_era5_dataset(nc_path):
    import xarray as xr
    fmt = detect_file_format(nc_path)
    for engine in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            ds = xr.open_dataset(nc_path, engine=engine)
            return ds, engine
        except Exception as e:
            continue
    raise ValueError(
        f"Cannot open {nc_path.name}.\n"
        f"Engines tried: netcdf4, h5netcdf, scipy\n"
        f"Install: pip install netCDF4 h5netcdf --user"
    )


# ── Date helpers ───────────────────────────────────────────────────────────────

def date_range(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def find_gaps(dates_present, start, end):
    all_dates = set(date_range(start, end))
    missing = sorted(all_dates - set(dates_present))
    if not missing:
        return []
    gaps, g_start, g_prev = [], missing[0], missing[0]
    for d in missing[1:]:
        if (d - g_prev).days == 1:
            g_prev = d
        else:
            gaps.append((g_start, g_prev, (g_prev - g_start).days + 1))
            g_start = g_prev = d
    gaps.append((g_start, g_prev, (g_prev - g_start).days + 1))
    return gaps

def find_nc_file(nc_dir, var_prefix, year):
    for d in [nc_dir, ERA5_DIR]:
        p = d / f"{var_prefix}_{year}.nc"
        if p.exists():
            return p
    return None


# ── Normalise a single-year dataset to a clean time-indexed DataArray ──────────

def normalise_dataset(ds, var_name):
    """
    ERA5-Land CDS files have quirks:
      - Time coordinate may be "valid_time" instead of "time"
      - After xr.concat the time axis may be a RangeIndex
      - Variable names may be uppercase (T2M, STL1) or have expver dimension
      - There may be a spurious "expver" or "number" dimension to drop

    Returns a DataArray with a clean DatetimeIndex on dim "time".
    """
    import pandas as pd

    # ── 1. Find the actual time coordinate ────────────────────────────────────
    time_coord = None
    for candidate in ["time", "valid_time", "forecast_reference_time"]:
        if candidate in ds.coords or candidate in ds.dims:
            time_coord = candidate
            break
    if time_coord is None:
        # Last resort: find any dimension whose values look like datetimes
        for dim in ds.dims:
            vals = ds[dim].values
            if len(vals) > 1:
                try:
                    pd.to_datetime(vals[0])
                    time_coord = dim
                    break
                except Exception:
                    pass
    if time_coord is None:
        raise KeyError(f"Cannot find time coordinate in dataset. "
                       f"Coords: {list(ds.coords)}  Dims: {list(ds.dims)}")

    # ── 2. Find the data variable ──────────────────────────────────────────────
    t_var = None
    for candidate in [var_name, var_name.upper(),
                      "VAR_2T", "VAR_STL1", "t2m", "stl1",
                      "2m_temperature", "soil_temperature_level_1"]:
        if candidate in ds:
            t_var = candidate
            break
    if t_var is None:
        # Take the first non-coordinate variable
        data_vars = [v for v in ds.data_vars
                     if not v.lower().startswith("expver")]
        if not data_vars:
            raise KeyError(f"No data variable found. "
                           f"Available: {list(ds.data_vars)}")
        t_var = data_vars[0]
        print(f"   Using variable: {t_var} (auto-selected from {list(ds.data_vars)})")

    da = ds[t_var]

    # ── 3. Drop nuisance dimensions (expver, number) ───────────────────────────
    for dim in ["expver", "number"]:
        if dim in da.dims:
            da = da.isel({dim: 0}).drop_vars(dim, errors="ignore")

    # ── 4. Rename time coordinate to "time" and ensure DatetimeIndex ──────────
    if time_coord != "time":
        da = da.rename({time_coord: "time"})

    # Force the time axis to a proper DatetimeIndex
    try:
        time_vals = pd.to_datetime(da.time.values)
        da = da.assign_coords(time=time_vals)
    except Exception as e:
        print(f"   [WARN] Could not convert time to DatetimeIndex: {e}")

    return da, t_var


# ── Core loader ────────────────────────────────────────────────────────────────

def load_era5land_spatial(nc_dir, var_prefix, var_name, years, date_start, date_end):
    """
    Load ERA5-Land files year by year, normalise each to a clean time-indexed
    DataArray, concatenate, resample to daily mean, convert K -> degC.
    """
    try:
        import xarray as xr
        import pandas as pd
    except ImportError:
        raise ImportError("pip install xarray --user")

    arrays = []   # list of normalised DataArrays, one per year
    for year in years:
        nc_path = find_nc_file(nc_dir, var_prefix, year)
        if nc_path is None:
            print(f"   WARNING: {var_prefix}_{year}.nc not found — skipping {year}")
            continue
        ds, engine = open_era5_dataset(nc_path)
        print(f"   {nc_path.name}  engine={engine}  "
              f"dims={dict(ds.dims)}  vars={list(ds.data_vars)}")
        da_year, t_var = normalise_dataset(ds, var_name)
        arrays.append(da_year)
        ds.close()

    if not arrays:
        raise FileNotFoundError(
            f"No files found for {var_prefix}.\n"
            f"Run 00b_download_era5land.py + 00_unzip_era5land.py first."
        )

    # Concatenate along time — each array already has a clean DatetimeIndex
    if len(arrays) == 1:
        da = arrays[0]
    else:
        da = xr.concat(arrays, dim="time", join="override")

    # Sort by time in case years are out of order
    da = da.sortby("time")

    # ── Coordinate names ───────────────────────────────────────────────────────
    lat_name = "latitude"  if "latitude"  in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    # ── Resample to daily mean ─────────────────────────────────────────────────
    # Check time spacing
    if len(da.time) > 1:
        t0 = pd.Timestamp(da.time.values[0])
        t1 = pd.Timestamp(da.time.values[1])
        dt_hours = (t1 - t0).total_seconds() / 3600.0
    else:
        dt_hours = 24.0

    if dt_hours < 20:
        print(f"   Sub-daily data ({dt_hours:.0f}h steps) -> resampling to daily mean")
        da = da.resample(time="1D").mean()
    else:
        print(f"   Already daily ({dt_hours:.0f}h steps)")

    # ── Extract values ─────────────────────────────────────────────────────────
    values = da.values   # (n_days, n_lat, n_lon)

    # K -> degC
    mean_val = float(np.nanmean(values))
    if mean_val > 150:
        values = (values - 273.15).astype(np.float32)
        print(f"   Kelvin -> degC  (was {mean_val:.1f} K)")
    else:
        values = values.astype(np.float32)
        print(f"   Already degC  (mean={mean_val:.1f})")

    lats = da[lat_name].values.astype(np.float64)
    lons = da[lon_name].values.astype(np.float64)
    lons = np.where(lons > 180, lons - 360, lons)

    # Parse dates and clip to target range
    import pandas as pd
    dates_all = [datetime(pd.Timestamp(t).year,
                          pd.Timestamp(t).month,
                          pd.Timestamp(t).day)
                 for t in da.time.values]

    mask      = [date_start <= d <= date_end for d in dates_all]
    dates_out = [d for d, m in zip(dates_all, mask) if m]
    data_3d   = values[mask, :, :]

    print(f"   Variable   : {t_var}")
    print(f"   Grid shape : {data_3d.shape}  "
          f"(days={len(dates_out)}, lat={len(lats)}, lon={len(lons)})")
    print(f"   Lat range  : {lats.min():.3f} – {lats.max():.3f} N")
    print(f"   Lon range  : {lons.min():.3f} – {lons.max():.3f} E")
    print(f"   Temp range : {np.nanmin(data_3d):.1f} – {np.nanmax(data_3d):.1f} degC")

    return dates_out, data_3d, lats, lons


# ── Gap fill ───────────────────────────────────────────────────────────────────

def fill_temporal_gaps_3d(dates_out, data_3d, date_start, date_end):
    date_set = {d: i for i, d in enumerate(dates_out)}
    all_days = list(date_range(date_start, date_end))
    n_total  = len(all_days)
    n_lat, n_lon = data_3d.shape[1], data_3d.shape[2]

    full = np.full((n_total, n_lat, n_lon), np.nan, dtype=np.float32)
    for i, d in enumerate(all_days):
        if d in date_set:
            full[i] = data_3d[date_set[d]]

    gaps     = find_gaps(list(date_set.keys()), date_start, date_end)
    n_filled = 0
    unfilled = []

    for g_start, g_end, g_len in gaps:
        if g_len > SHORT_GAP_DAYS:
            unfilled.append((g_start, g_end, g_len))
            continue
        try:
            i_before = all_days.index(g_start) - 1
            i_after  = all_days.index(g_end) + 1
        except ValueError:
            unfilled.append((g_start, g_end, g_len))
            continue
        if i_before < 0 or i_after >= n_total:
            unfilled.append((g_start, g_end, g_len))
            continue
        n_steps = i_after - i_before
        for step in range(1, n_steps):
            alpha = step / n_steps
            full[i_before + step] = (
                (1 - alpha) * full[i_before] + alpha * full[i_after]
            )
        n_filled += g_len

    return all_days, full, gaps, unfilled, n_filled


# ── Output writers ─────────────────────────────────────────────────────────────

def save_daily_grid_nc(out_path, all_days, data_3d, lats, lons,
                       var_name, description):
    try:
        import xarray as xr
        import pandas as pd
        times = pd.to_datetime(all_days)
        da = xr.DataArray(
            data_3d,
            coords={"time": times, "latitude": lats, "longitude": lons},
            dims=["time", "latitude", "longitude"],
            name=var_name,
            attrs={"units": "degC", "long_name": description,
                   "source": "ERA5-Land (~9 km)",
                   "processing": "daily mean, K->degC, gaps filled"}
        )
        xr.Dataset({var_name: da}).to_netcdf(out_path, format="NETCDF4",
                                              engine="netcdf4")
        print(f"   Written: {out_path.name}")
    except Exception as e:
        npz_path = out_path.with_suffix(".npz")
        np.savez_compressed(
            npz_path, data=data_3d, lats=lats, lons=lons,
            dates=np.array([d.strftime("%Y%m%d") for d in all_days])
        )
        print(f"   [WARN] NetCDF write failed ({e})")
        print(f"   Saved as .npz: {npz_path.name}  (script 09 auto-loads this)")

def extract_point_series(all_days, data_3d, lats, lons, target_lat, target_lon):
    i_lat = int(np.argmin(np.abs(lats - target_lat)))
    i_lon = int(np.argmin(np.abs(lons - target_lon)))
    point = data_3d[:, i_lat, i_lon]
    print(f"   TUK nearest ERA5-Land cell: "
          f"lat={lats[i_lat]:.3f} lon={lons[i_lon]:.3f}")
    return {d: float(v) for d, v in zip(all_days, point) if np.isfinite(v)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ERA5-Land spatial temperature preparation  (v2.3)")
    print(f"Date range : {DATE_START.strftime('%Y-%m-%d')} -> "
          f"{DATE_END.strftime('%Y-%m-%d')}")
    print(f"Source dir : {NC_DIR}")
    print(f"Output dir : {ERA5_DIR}")
    print(f"xarray engines: {_list_xarray_engines()}")
    print()

    years = list(range(DATE_START.year, DATE_END.year + 1))
    report_lines = ["ERA5-Land preparation report (v2.3)\n",
                    f"Date range: {DATE_START} -> {DATE_END}\n\n"]

    for var_name, var_prefix, description in VARIABLES:
        print("=" * 60)
        print(f"Variable: {var_name}  ({description})")

        out_nc = Path(str(OUT_GRID_PATTERN).replace("{var}", var_name))
        if out_nc.exists():
            print(f"  Exists: {out_nc.name} — delete to reprocess.")
            continue

        dates_out, data_3d, lats, lons = load_era5land_spatial(
            NC_DIR, var_prefix, var_name, years, DATE_START, DATE_END
        )

        all_days, data_full, gaps, unfilled, n_filled = fill_temporal_gaps_3d(
            dates_out, data_3d, DATE_START, DATE_END
        )

        total  = (DATE_END - DATE_START).days + 1
        n_miss = int(np.sum(np.all(np.isnan(data_full), axis=(1, 2))))
        print(f"  Days total={total} / with_data={len(dates_out)} / "
              f"filled={n_filled} / missing={n_miss}")
        for g_s, g_e, g_l in unfilled:
            print(f"    UNFILLED GAP: {g_s.strftime('%Y-%m-%d')} -> "
                  f"{g_e.strftime('%Y-%m-%d')} ({g_l}d)")

        save_daily_grid_nc(out_nc, all_days, data_full, lats, lons,
                           var_name, description)

        if var_name == "t2m":
            pt  = extract_point_series(all_days, data_full, lats, lons,
                                       TUK_LAT, TUK_LON)
            n_w = 0
            with open(OUT_POINT_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["date", "t2m_mean_c"])
                for d in date_range(DATE_START, DATE_END):
                    t = pt.get(d)
                    if t is not None and np.isfinite(t):
                        w.writerow([d.strftime("%Y%m%d"), round(t, 3)])
                        n_w += 1
            print(f"  Written point CSV: {OUT_POINT_CSV.name}  ({n_w} days)")

        report_lines += [f"\nVariable: {var_name}\n",
                         f"  Shape: {data_full.shape}\n",
                         f"  Missing: {n_miss} days\n"]

    OUT_REPORT.write_text("".join(report_lines), encoding="utf-8")
    print(f"\nReport: {OUT_REPORT.name}")
    print("=" * 60)
    print("Done. Next: python 09_ddt_timeseries.py")

if __name__ == "__main__":
    main()
