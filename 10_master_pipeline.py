"""
10_master_pipeline.py
DS-InSAR master orchestration script.
Runs all processing steps sequentially.
Stops only if a critical step fails.
Logs every step to logs/pipeline_YYYYMMDD_HHMMSS.log

Usage:
    python C:/bcosta/DS-InSAR/scripts/10_master_pipeline.py

Run from any directory — all paths are absolute.
"""

import os
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(r"C:\bcosta\DS-InSAR\scripts")
WORK_DIR    = Path(r"C:\bcosta\DS-InSAR\tc")
LOG_DIR     = SCRIPTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Pipeline definition ────────────────────────────────────────────────────────
PIPELINE = [
    ("00_prepare_temperature.py",       SCRIPTS_DIR, False),
    ("01_verify_unwrapped_data.py",     WORK_DIR,    True ),
    ("02_coherence_mask.py",            SCRIPTS_DIR, True ),
    ("03_load_phases.py",               SCRIPTS_DIR, True ),
    ("04_build_design_matrix.py",       SCRIPTS_DIR, True ),
    ("05a_atmospheric_correction.py",   SCRIPTS_DIR, False),
    ("05b_inversion.py",                SCRIPTS_DIR, True ),
    ("06_reference_point.py",           SCRIPTS_DIR, True ),
    ("07_export_geotiff.py",            SCRIPTS_DIR, True ),
    ("08_plot_results.py",              SCRIPTS_DIR, False),
    ("09_ddt_timeseries.py",            SCRIPTS_DIR, False),
]

# ── Runner ─────────────────────────────────────────────────────────────────────
def run_step(script: str, cwd: Path) -> bool:
    script_path = cwd / script if (cwd / script).exists() else SCRIPTS_DIR / script
    if not script_path.exists():
        log.error(f"Script not found: {script_path}")
        return False

    # Force UTF-8 stdout in all child processes
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            log.info(f"    {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            log.warning(f"    STDERR: {line}")

    return result.returncode == 0


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("DS-InSAR pipeline started")
    log.info(f"Log file : {log_file}")
    log.info(f"Scripts  : {SCRIPTS_DIR}")
    log.info("=" * 70)

    start_time = datetime.now()
    n_ok = n_warn = 0

    for script, cwd, critical in PIPELINE:
        log.info("-" * 70)
        log.info(f"▶  {script}  (cwd={cwd.name})")
        t0 = datetime.now()

        success = run_step(script, cwd)
        elapsed = (datetime.now() - t0).total_seconds()

        if success:
            log.info(f"✓  {script} completed in {elapsed:.1f}s")
            n_ok += 1
        else:
            if critical:
                log.critical(f"✗  CRITICAL FAILURE: {script} (elapsed {elapsed:.1f}s)")
                log.critical("Pipeline halted. Fix the error above and re-run.")
                log.critical(f"Total elapsed: {(datetime.now() - start_time).total_seconds():.1f}s")
                log.critical(f"Steps OK={n_ok}  WARN={n_warn}  FAILED=1")
                sys.exit(1)
            else:
                log.warning(f"⚠  Non-critical step failed: {script} (elapsed {elapsed:.1f}s)")
                log.warning("   Continuing to next step.")
                n_warn += 1

    total = (datetime.now() - start_time).total_seconds()
    log.info("=" * 70)
    log.info(f"Pipeline finished in {total:.1f}s")
    log.info(f"Steps completed: {n_ok} OK  |  {n_warn} warnings  |  0 failures")
    log.info(f"Full log: {log_file}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()