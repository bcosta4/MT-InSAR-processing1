"""
04_build_design_matrix.py
Parses IFG date pairs from the ifg_pairs.csv files produced by 03_load_phases.py,
extracts the unique acquisition date list, and builds the SBAS design matrix A
(n_ifgs x n_intervals) for each network independently.

The design matrix encodes which time intervals each interferogram spans.
Each row corresponds to one IFG; each column to one time interval between
consecutive acquisition dates. A value of 1 means the IFG spans that interval,
0 means it does not.

Outputs (per network, written to WORK_DIR):
- _design_matrix.npy  : float32 (n_ifgs x n_intervals)
- _date_list.csv      : unique acquisition dates in order
- _network_report.txt : design matrix diagnostics
"""

import csv
import numpy as np
from pathlib import Path
from datetime import datetime

# -- Paths ---------------------------------------------------------------------
WORK_DIR = Path(r"C:\bcosta\DS-InSAR\tc")

# -- Configuration -------------------------------------------------------------
IFG_PAIRS_SUFFIX = "_ifg_pairs.csv"
DATE_FMT         = "%Y%m%d"

# -- Helpers -------------------------------------------------------------------

def load_ifg_pairs(csv_path):
    """Load IFG pairs from CSV. Returns list of dicts."""
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def build_design_matrix(ifg_records, date_list):
    """
    Build SBAS design matrix A (n_ifgs x n_intervals).

    date_list  : sorted list of unique datetime objects (n_dates)
    n_intervals = n_dates - 1

    For each IFG spanning date[i] -> date[j] (j > i), columns i through j-1
    are set to 1. This encodes that the IFG is the cumulative sum of phase
    increments across all intervals between the two dates.

    Returns A as float32 numpy array and list of skipped row indices.
    """
    n_ifgs      = len(ifg_records)
    n_intervals = len(date_list) - 1
    A           = np.zeros((n_ifgs, n_intervals), dtype=np.float32)
    date_index  = {d: i for i, d in enumerate(date_list)}
    skipped     = []

    for row_idx, rec in enumerate(ifg_records):
        try:
            d1 = datetime.strptime(rec["date_primary"],   DATE_FMT)
            d2 = datetime.strptime(rec["date_secondary"], DATE_FMT)
        except ValueError:
            skipped.append(row_idx)
            continue

        if d1 not in date_index or d2 not in date_index:
            skipped.append(row_idx)
            continue

        i1 = date_index[d1]
        i2 = date_index[d2]

        if i1 >= i2:
            skipped.append(row_idx)
            continue

        A[row_idx, i1:i2] = 1.0

    return A, skipped


def check_network_connectivity(A, date_list):
    """
    Check that every time interval is spanned by at least one IFG.
    Returns coverage array and list of gap interval indices.
    """
    coverage = np.sum(A, axis=0)
    gaps     = [i for i, c in enumerate(coverage) if c == 0]
    return coverage, gaps


def rank_check(A):
    """
    Check rank of design matrix. Full column rank means the system is
    uniquely solvable. Returns rank and whether it is full column rank.
    """
    rank      = np.linalg.matrix_rank(A)
    n_cols    = A.shape[1]
    full_rank = rank == n_cols
    return rank, full_rank


# -- Per-network processing ----------------------------------------------------
def process_network(csv_path):

    stem = csv_path.stem.replace("_ifg_pairs", "")

    print("\n" + "=" * 60)
    print(f"Network : {stem}")

    # -- Load IFG pairs --------------------------------------------------------
    ifg_records = load_ifg_pairs(csv_path)
    print(f"  IFG pairs loaded  : {len(ifg_records)}")

    # -- Extract unique sorted dates -------------------------------------------
    dates = set()
    for rec in ifg_records:
        try:
            dates.add(datetime.strptime(rec["date_primary"],   DATE_FMT))
            dates.add(datetime.strptime(rec["date_secondary"], DATE_FMT))
        except ValueError:
            continue

    date_list   = sorted(dates)
    n_dates     = len(date_list)
    n_intervals = n_dates - 1

    print(f"  Unique dates      : {n_dates}")
    print(f"  Time intervals    : {n_intervals}")
    print(f"  Date range        : {date_list[0].strftime('%Y-%m-%d')} -> "
          f"{date_list[-1].strftime('%Y-%m-%d')}")

    # -- Build design matrix ---------------------------------------------------
    A, skipped = build_design_matrix(ifg_records, date_list)

    if skipped:
        print(f"  WARNING: {len(skipped)} IFGs skipped (unparseable dates): rows {skipped}")

    # -- Network diagnostics ---------------------------------------------------
    coverage, gaps = check_network_connectivity(A, date_list)
    rank, full_rank = rank_check(A)

    print(f"\n  Design matrix     : {A.shape}  (IFGs x intervals)")
    print(f"  Matrix rank       : {rank} / {n_intervals}  "
          f"({'FULL RANK OK' if full_rank else 'RANK DEFICIENT!'})")

    if gaps:
        print(f"  WARNING: {len(gaps)} interval(s) with no IFG coverage:")
        for g in gaps:
            d_start = date_list[g].strftime("%Y-%m-%d")
            d_end   = date_list[g + 1].strftime("%Y-%m-%d")
            print(f"    Interval {g}: {d_start} -> {d_end}")
    else:
        print(f"  Network coverage  : all intervals covered OK")

    print(f"\n  IFGs per interval (min / mean / max): "
          f"{coverage.min():.0f} / {coverage.mean():.1f} / {coverage.max():.0f}")

    # -- Print date list -------------------------------------------------------
    print(f"\n  Acquisition dates:")
    for i, d in enumerate(date_list):
        if i < n_dates - 1:
            gap_days = (date_list[i + 1] - d).days
            print(f"    [{i:02d}] {d.strftime('%Y-%m-%d')}  +{gap_days}d  "
                  f"{date_list[i + 1].strftime('%Y-%m-%d')}")
        else:
            print(f"    [{i:02d}] {d.strftime('%Y-%m-%d')}  (last)")

    # -- Save outputs ----------------------------------------------------------
    out_matrix = WORK_DIR / f"{stem}_design_matrix.npy"
    out_dates  = WORK_DIR / f"{stem}_date_list.csv"
    out_rep    = WORK_DIR / f"{stem}_network_report.txt"

    np.save(out_matrix, A)

    with open(out_dates, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "date", "days_since_first"])
        t0 = date_list[0]
        for i, d in enumerate(date_list):
            writer.writerow([i, d.strftime(DATE_FMT), (d - t0).days])

    # Build report — ASCII only, UTF-8 encoded
    t0 = date_list[0]
    report_lines = [
        f"Network report : {stem}\n",
        f"IFG pairs      : {len(ifg_records)}\n",
        f"Unique dates   : {n_dates}\n",
        f"Time intervals : {n_intervals}\n",
        f"Date range     : {date_list[0].strftime('%Y-%m-%d')} -> "
        f"{date_list[-1].strftime('%Y-%m-%d')}\n",
        f"Design matrix  : {A.shape}\n",
        f"Matrix rank    : {rank} / {n_intervals} "
        f"({'full rank' if full_rank else 'RANK DEFICIENT'})\n",
        f"Coverage gaps  : {len(gaps)}\n",
        f"IFGs/interval  : min={coverage.min():.0f} "
        f"mean={coverage.mean():.1f} max={coverage.max():.0f}\n\n",
        f"Date list:\n",
    ]
    for i, d in enumerate(date_list):
        report_lines.append(
            f"  [{i:02d}] {d.strftime('%Y-%m-%d')}  (t+{(d - t0).days}d)\n"
        )

    if gaps:
        report_lines.append(f"\nCoverage gaps:\n")
        for g in gaps:
            report_lines.append(
                f"  Interval {g}: {date_list[g].strftime('%Y-%m-%d')} -> "
                f"{date_list[g + 1].strftime('%Y-%m-%d')}\n"
            )

    # FIX: explicit UTF-8 encoding prevents cp1252 crash on Windows
    Path(out_rep).write_text("".join(report_lines), encoding="utf-8")

    print(f"\n  Written: {out_matrix.name}")
    print(f"  Written: {out_dates.name}")
    print(f"  Written: {out_rep.name}")

    return stem, date_list, A


# -- Entry point ---------------------------------------------------------------
def main():
    csv_files = sorted(WORK_DIR.glob("*_ifg_pairs.csv"))

    if not csv_files:
        print(f"No *_ifg_pairs.csv files found in {WORK_DIR}")
        print(f"Run 03_load_phases.py first.")
        return

    print(f"Found {len(csv_files)} network(s) in {WORK_DIR}")

    results = []
    for csv_path in csv_files:
        result = process_network(csv_path)
        if result:
            results.append(result)

    # -- Cross-network summary -------------------------------------------------
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("Cross-network comparison:")
        for stem, date_list, A in results:
            rank, full_rank = rank_check(A)
            print(f"  {stem}")
            print(f"    Dates     : {len(date_list)}")
            print(f"    IFGs      : {A.shape[0]}")
            print(f"    Intervals : {A.shape[1]}")
            print(f"    Rank      : {rank}/{A.shape[1]}  "
                  f"({'OK' if full_rank else 'DEFICIENT'})")

        dates_1 = set(results[0][1])
        dates_2 = set(results[1][1])
        shared  = dates_1 & dates_2
        print(f"\n  Shared acquisition dates : {len(shared)}")
        if shared:
            for d in sorted(shared):
                print(f"    {d.strftime('%Y-%m-%d')}")

    print("\n" + "=" * 60)
    print("Design matrix build complete.")


if __name__ == "__main__":
    main()   