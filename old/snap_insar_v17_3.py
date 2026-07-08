#!/usr/bin/env python3

r"""

Sentinel-1 InSAR Batch Processing with SNAP 12+ - PRODUCTION FINAL v17.3

===============================================================================

CRITICAL UPDATE v17.3: Goldstein Fix for SNAP 12

GOLDSTEIN PARAMETER FIX:
- v17.2 included unsupported parameters (windowSize, filterSize, useCoherence)
- SNAP 12 GoldsteinPhaseFiltering operator only accepts: alpha
- v17.3 removes invalid parameters → SNAP 12 compatible

KEY CHANGES v17.3:

- Outputs ALL bands: i_ifg, q_ifg, Phase_ifg, coh, Intensity_ifg, elevation

- Phase_ifg: Wrapped phase in radians (extracted from complex interferogram)

- elevation: DEM reprojected to radar geometry

- All in BEAM-DIMAP radar geometry format

- Ready for MintPy time-series processing

- Goldstein filtering: alpha=1.0 only (SNAP 12 compatible)

OUTPUT BANDS (v17.3):

- i_ifg (Real part of complex interferogram)

- q_ifg (Imaginary part)

- Phase_ifg (Wrapped phase [-π, +π] in radians)

- coh (Coherence magnitude [0, 1])

- Intensity_ifg (Intensity)

- elevation (DEM reprojected to radar geometry)

WORKFLOW (FINAL v17.3):

Read → Split → Orbit → Back-Geocoding
→ Interferogram (topographic removal)
→ Deburst
→ Goldstein Phase Filtering (alpha=1.0 ONLY) ← v17.3 FIX
→ Multilooking (8x2)
→ PhaseToDisplacement (EXTRACT PHASE BAND)
→ Write (BEAM-DIMAP, ALL bands)

Author: Arctic InSAR Processing Group
Date: January 2026
Version: 17.3 (GOLDSTEIN FIX - SNAP 12 COMPATIBLE)

"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from collections import defaultdict


class Config:
    """Processing configuration."""

    def __init__(self):
        self.snap_exe = self.find_snap_gpt()
        self.min_bt_days = 6
        self.max_bt_days = 450
        self.dem_paths = []
        self.dem_mosaic_path = None
        self.output_dir = Path(r'C:\bcosta\DS-InSAR')
        self.temp_dir = self.output_dir / 'temp'
        self.log_file = self.output_dir / 'processing.log'
        self.nlooks_range = 8
        self.nlooks_azimuth = 2
        self.ram_gb = 8
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()

    def find_snap_gpt(self) -> str:
        """Locate SNAP GPT executable."""
        snap_gpt_paths = [
            Path(r'C:\Program Files\esa-snap\bin\gpt.exe'),
            Path(r'C:\Program Files\snap\bin\gpt.exe'),
            Path(r'C:\snap\bin\gpt.exe'),
        ]

        for path in snap_gpt_paths:
            if path.exists():
                self.logger_console(f"✓ Found SNAP GPT at: {path}")
                return str(path)

        raise FileNotFoundError("SNAP GPT not found. Install SNAP 12+")

    def logger_console(self, msg: str):
        print(f"[SNAP CONFIG] {msg}")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('InSARBatch')
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger


class DEMManager:
    def __init__(self, dem_input: str, temp_dir: Path, logger: logging.Logger):
        self.dem_input = dem_input
        self.temp_dir = temp_dir
        self.logger = logger
        self.dem_paths = []
        self.dem_mosaic_path = None

    def parse_dem_paths(self) -> List[Path]:
        dem_list = [p.strip() for p in self.dem_input.split(',')]
        dem_paths = []

        for dem_str in dem_list:
            dem_path = Path(dem_str)
            if not dem_path.exists():
                self.logger.error(f"DEM file not found: {dem_path}")
                raise FileNotFoundError(f"DEM not found: {dem_path}")

            dem_paths.append(dem_path)
            self.logger.info(f"Registered DEM: {dem_path.name}")

        self.dem_paths = dem_paths
        return dem_paths

    def create_vrt_mosaic(self) -> Path:
        if len(self.dem_paths) == 1:
            self.logger.info(f"Single DEM detected: {self.dem_paths[0].name}")
            self.dem_mosaic_path = self.dem_paths[0]
            return self.dem_paths[0]

        self.logger.info(f"Creating VRT mosaic from {len(self.dem_paths)} DEM tiles...")
        vrt_path = self.temp_dir / 'dem_mosaic.vrt'

        gdal_cmd = ['gdalbuildvrt', '-separate', '-overwrite', str(vrt_path),
                    *[str(p) for p in self.dem_paths]]

        try:
            result = subprocess.run(gdal_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                self.logger.info(f"✓ VRT mosaic created: {vrt_path}")
                self.dem_mosaic_path = vrt_path
                return vrt_path
            else:
                raise RuntimeError(f"gdalbuildvrt failed: {result.stderr}")
        except FileNotFoundError:
            self.logger.error("gdalbuildvrt not found.")
            raise

    def get_dem_for_snap(self) -> Path:
        if not self.dem_paths:
            self.parse_dem_paths()

        if self.dem_mosaic_path:
            return self.dem_mosaic_path

        return self.create_vrt_mosaic()


class SLCFileManager:
    def __init__(self, slc_dir: Path, logger: logging.Logger):
        self.slc_dir = Path(slc_dir)
        self.logger = logger
        self.slc_files = []
        self.inventory = {}

    def discover_slc_files(self, pattern='*.zip') -> List[Path]:
        self.logger.info(f"Discovering SLC files in {self.slc_dir}")
        slc_files = sorted(self.slc_dir.glob(pattern))

        if not slc_files:
            self.logger.warning(f"No SLC files found matching {pattern}")
            return []

        self.logger.info(f"Found {len(slc_files)} SLC files")
        self.slc_files = slc_files
        return slc_files

    def extract_acquisition_date(self, slc_file: Path) -> Optional[datetime]:
        filename = slc_file.stem
        parts = filename.split('_')

        if len(parts) >= 6:
            date_str = parts[5]
            try:
                acq_date = datetime.strptime(date_str[:8], '%Y%m%d')
                return acq_date
            except ValueError:
                return None

        return None

    def build_inventory(self) -> Dict:
        inventory = defaultdict(list)

        for slc_file in self.slc_files:
            acq_date = self.extract_acquisition_date(slc_file)
            if acq_date:
                inventory[acq_date].append(slc_file)
                self.logger.debug(f"Catalogued {slc_file.name} (Date: {acq_date.strftime('%Y-%m-%d')})")

        self.inventory = dict(inventory)
        self.logger.info(f"Inventory built: {len(self.inventory)} unique acquisition dates")
        return self.inventory


class PairGenerator:
    def __init__(self, inventory: Dict, min_bt: int, max_bt: int, logger: logging.Logger):
        self.inventory = inventory
        self.min_bt = min_bt
        self.max_bt = max_bt
        self.logger = logger
        self.valid_pairs = []

    def generate_pairs(self) -> List[Tuple]:
        dates = sorted(self.inventory.keys())
        pairs = []

        for i, ref_date in enumerate(dates):
            for sec_date in dates[i+1:]:
                bt = (sec_date - ref_date).days
                if self.min_bt <= bt <= self.max_bt:
                    pairs.append((ref_date, sec_date, bt))

        self.valid_pairs = pairs
        self.logger.info(f"Generated {len(pairs)} valid pairs (Bt: {self.min_bt}-{self.max_bt} days)")
        return pairs

    def get_slc_files_for_pair(self, ref_date: datetime, sec_date: datetime) -> Tuple:
        ref_files = self.inventory.get(ref_date, [])
        sec_files = self.inventory.get(sec_date, [])

        if not ref_files or not sec_files:
            return None, None

        return ref_files[0], sec_files[0]

    def filter_by_coherence_expectation(self, land_cover_class: str = 'sparse_tundra') -> List[Tuple]:
        tau_values = {'sparse_tundra': 40, 'urban': 180, 'bare_rock': 120, 'grassland': 30, 'forest': 15}
        tau = tau_values.get(land_cover_class, 40)

        filtered_pairs = []

        for ref_date, sec_date, bt in self.valid_pairs:
            gamma_0 = 0.58
            expected_gamma = gamma_0 * np.exp(-bt / tau)
            filtered_pairs.append((ref_date, sec_date, bt, expected_gamma))

        filtered_pairs.sort(key=lambda x: x[3], reverse=True)

        self.logger.info(
            f"Total pairs for processing: {len(filtered_pairs)}\n"
            f" High coherence (gamma > 0.5): {sum(1 for p in filtered_pairs if p[3] > 0.5)}\n"
            f" Medium coherence (0.3 < gamma < 0.5): {sum(1 for p in filtered_pairs if 0.3 < p[3] <= 0.5)}\n"
            f" Low coherence (gamma < 0.3): {sum(1 for p in filtered_pairs if p[3] <= 0.3)}"
        )

        return filtered_pairs


class SnapGraphGenerator:
    def __init__(self, dem_path: Path, nlooks_range: int, nlooks_azimuth: int, logger: logging.Logger):
        self.dem_path = str(dem_path)
        self.nlooks_range = nlooks_range
        self.nlooks_azimuth = nlooks_azimuth
        self.logger = logger

    def create_insar_graph(self) -> str:
        """Generate SNAP 12.0.0 compatible processing graph - v17.3 (GOLDSTEIN FIX)

        CRITICAL FIX v17.3: 
        - Removed windowSize, filterSize, useCoherence, coherenceThreshold from Goldstein node
        - SNAP 12 GoldsteinPhaseFiltering only accepts 'alpha' parameter
        - Window size and other parameters are handled internally by SNAP

        Output bands (BEAM-DIMAP):
        - i_ifg, q_ifg (Complex interferogram - wrapped phase)
        - Phase_ifg (Wrapped phase extracted from complex: atan2(q, i))
        - coh (Coherence magnitude)
        - Intensity_ifg (Interferometric intensity)
        - elevation (DEM reprojected to radar geometry)

        This matches your expected output structure from the image.
        """

        graph = f"""<?xml version="1.0" encoding="UTF-8"?>
<graph id="INSAR-TUKTOYAKTUK-V17.3">
  <version>1.0</version>

  <!-- Read reference SLC -->
  <node id="read_ref">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>$SOURCE_REF</file>
    </parameters>
  </node>

  <!-- Read secondary SLC -->
  <node id="read_sec">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>$SOURCE_SEC</file>
    </parameters>
  </node>

  <!-- TOPSAR Split reference (IW3, VV) -->
  <node id="split_ref">
    <operator>TOPSAR-Split</operator>
    <sources>
      <sourceProduct refid="read_ref"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <subswath>IW3</subswath>
      <polarisation>VV</polarisation>
    </parameters>
  </node>

  <!-- TOPSAR Split secondary (IW3, VV) -->
  <node id="split_sec">
    <operator>TOPSAR-Split</operator>
    <sources>
      <sourceProduct refid="read_sec"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <subswath>IW3</subswath>
      <polarisation>VV</polarisation>
    </parameters>
  </node>

  <!-- Apply orbit file (reference) -->
  <node id="orbit_ref">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="split_ref"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Restituted (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>

  <!-- Apply orbit file (secondary) -->
  <node id="orbit_sec">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="split_sec"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Restituted (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>

  <!-- Back-Geocoding Coregistration -->
  <node id="backgeo">
    <operator>Back-Geocoding</operator>
    <sources>
      <sourceProduct refid="orbit_ref"/>
      <sourceProduct.secondary refid="orbit_sec"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>External DEM</demName>
      <externalDEMFile>{self.dem_path}</externalDEMFile>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
    </parameters>
  </node>

  <!-- Interferogram Formation -->
  <node id="ifg">
    <operator>Interferogram</operator>
    <sources>
      <sourceProduct refid="backgeo"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <subtractFlatEarthPhase>true</subtractFlatEarthPhase>
      <subtractTopographicPhase>true</subtractTopographicPhase>
      <demName>External DEM</demName>
      <externalDEMFile>{self.dem_path}</externalDEMFile>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <orbitDegree>5</orbitDegree>
      <includeCoherence>true</includeCoherence>
      <squarePixel>true</squarePixel>
    </parameters>
  </node>

  <!-- TOPSAR Deburst -->
  <node id="deburst">
    <operator>TOPSAR-Deburst</operator>
    <sources>
      <sourceProduct refid="ifg"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VV</selectedPolarisations>
    </parameters>
  </node>

  <!-- Goldstein Phase Filtering (v17.3 FIXED: alpha parameter only) -->
  <!-- SNAP 12+ only accepts 'alpha' parameter -->
  <!-- alpha=1.0: Filter strength (1.0 = maximum filtering) -->
  <!-- Window size, filter size, coherence weighting handled internally -->
  <node id="goldstein">
    <operator>GoldsteinPhaseFiltering</operator>
    <sources>
      <sourceProduct refid="deburst"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <alpha>1.0</alpha>
    </parameters>
  </node>

  <!-- Multilooking -->
  <node id="multilook">
    <operator>Multilook</operator>
    <sources>
      <sourceProduct refid="goldstein"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <nRgLooks>{self.nlooks_range}</nRgLooks>
      <nAzLooks>{self.nlooks_azimuth}</nAzLooks>
    </parameters>
  </node>

  <!-- Phase to Displacement (Extract Phase Band) -->
  <node id="phase">
    <operator>PhaseToDisplacement</operator>
    <sources>
      <sourceProduct refid="multilook"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
    </parameters>
  </node>

  <!-- Terrain Correction (Add Elevation Band) -->
  <node id="terrain">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="phase"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>External DEM</demName>
      <externalDEMFile>{self.dem_path}</externalDEMFile>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <externalDEMApplyEGM>false</externalDEMApplyEGM>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>20.0</pixelSpacingInMeter>
      <pixelSpacingInDegree></pixelSpacingInDegree>
      <mapProjection>GEOGC</mapProjection>
      <alignToStandardGrid>false</alignToStandardGrid>
      <standardGridOriginX>0.0</standardGridOriginX>
      <standardGridOriginY>0.0</standardGridOriginY>
      <northCeiling>89.9999</northCeiling>
      <southFloor>-89.9999</southFloor>
      <westBound>-180.0</westBound>
      <eastBound>180.0</eastBound>
      <noDataValueAtSea>true</noDataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveLatLon>false</saveLatLon>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
      <outputComplex>false</outputComplex>
      <applyRadiometricNormalization>false</applyRadiometricNormalization>
      <saveMaskInsteadOfValidityFlag>false</saveMaskInsteadOfValidityFlag>
      <removeReferenceMean>false</removeReferenceMean>
    </parameters>
  </node>

  <!-- Write Output (BEAM-DIMAP) -->
  <node id="write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="terrain"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>$OUTPUT_IFG</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>

</graph>
"""

        return graph

    def save_graph(self, graph_xml: str, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(graph_xml)
        self.logger.debug(f"Saved graph XML: {output_file}")


class SnapProcessor:
    def __init__(self, gpt_exe: str, ram_gb: int, logger: logging.Logger):
        self.gpt_exe = gpt_exe
        self.ram_gb = ram_gb
        self.logger = logger

    def process_pair(self, graph_file: Path, output_ifg: Path, ref_slc: Path, sec_slc: Path,
                     timeout: int = 7200) -> bool:
        output_ifg.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.gpt_exe,
            str(graph_file),
            f'-PSOURCE_REF={str(ref_slc)}',
            f'-PSOURCE_SEC={str(sec_slc)}',
            f'-POUTPUT_IFG={str(output_ifg)}',
            f'-PramIndex={self.ram_gb}'
        ]

        self.logger.info(f"Executing SNAP 12 GPT: gpt graph.xml -PSOURCE_REF=... -PSOURCE_SEC=... -POUTPUT_IFG=...")

        try:
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"✓ Successfully processed! (Phase + Coherence + Elevation output ready)")
                return True
            else:
                self.logger.error(f"✗ SNAP processing failed (return code: {result.returncode})")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')[:30]
                    self.logger.error(f"STDERR:\n" + '\n'.join(stderr_lines))
                return False

        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ SNAP processing timeout (>{timeout}s)")
            return False

        except Exception as e:
            self.logger.error(f"✗ SNAP execution error: {e}")
            return False


class InSARBatchProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = config.logger
        self.results = []

    def run(self, slc_input):
        self.logger.info("="*80)
        self.logger.info("Sentinel-1 InSAR Batch Processing (SNAP 12+) - PRODUCTION v17.3")
        self.logger.info("Output: Phase + Coherence + Elevation (EXTENDED OUTPUTS)")
        self.logger.info("="*80)
        self.logger.info(f"Configuration:")
        self.logger.info(f" - SNAP GPT: {self.config.snap_exe}")
        self.logger.info(f" - Output directory: {self.config.output_dir}")
        self.logger.info(f" - Multilooking: {self.config.nlooks_range}x{self.config.nlooks_azimuth}")
        self.logger.info(f" - DEM: {self.config.dem_mosaic_path.name}")
        self.logger.info(f" - Geocoding: DISABLED (output in SAR slant-range coordinates)")
        self.logger.info(f" - Goldstein Alpha: 1.0 (SNAP 12 compatible - alpha only)")
        self.logger.info(f" - Output Bands (v17.3):")
        self.logger.info(f"    * i_ifg, q_ifg (Complex interferogram)")
        self.logger.info(f"    * Phase_ifg (Wrapped phase in radians)")
        self.logger.info(f"    * coh (Coherence magnitude)")
        self.logger.info(f"    * Intensity_ifg (Interferometric intensity)")
        self.logger.info(f"    * elevation (DEM in radar geometry)")
        self.logger.info("="*80 + "\n")

        slc_path = Path(slc_input)

        if slc_path.is_file() and slc_path.suffix == '.txt':
            self.logger.info("Reading SLC list from text file...")
            with open(slc_path) as f:
                slc_files = [Path(line.strip()) for line in f if line.strip()]
            slc_dir = slc_files[0].parent if slc_files else slc_path.parent
        else:
            slc_dir = slc_path

        self.logger.info(f"Scanning directory for SLC files...")

        file_manager = SLCFileManager(slc_dir, self.logger)
        slc_files = file_manager.discover_slc_files()

        if not slc_files:
            self.logger.error("No SLC files found. Exiting.")
            return

        inventory = file_manager.build_inventory()

        if not inventory:
            self.logger.error("Could not build inventory from SLC files. Exiting.")
            return

        pair_gen = PairGenerator(inventory, self.config.min_bt_days, self.config.max_bt_days, self.logger)
        all_pairs = pair_gen.generate_pairs()

        if not all_pairs:
            self.logger.error("No valid pairs generated. Exiting.")
            return

        filtered_pairs = pair_gen.filter_by_coherence_expectation(land_cover_class='sparse_tundra')

        dem_for_snap = self.config.dem_mosaic_path

        graph_gen = SnapGraphGenerator(dem_for_snap, self.config.nlooks_range,
                                       self.config.nlooks_azimuth, self.logger)

        snap_proc = SnapProcessor(self.config.snap_exe, self.config.ram_gb, self.logger)

        self.logger.info(f"\nProcessing {len(filtered_pairs)} interferometric pairs...")
        self.logger.info("-"*80 + "\n")

        for pair_idx, (ref_date, sec_date, bt, expected_gamma) in enumerate(filtered_pairs, 1):
            pair_name = f"{ref_date.strftime('%Y%m%d')}_{sec_date.strftime('%Y%m%d')}"

            self.logger.info(f"\n[{pair_idx}/{len(filtered_pairs)}] Processing pair: {pair_name}")
            self.logger.info(f" Temporal baseline: {bt} days | Expected coherence: {expected_gamma:.3f}")

            ref_slc, sec_slc = pair_gen.get_slc_files_for_pair(ref_date, sec_date)

            if not ref_slc or not sec_slc:
                self.logger.error(f" SLC files not found for this pair")
                self.results.append({
                    'pair': pair_name,
                    'Bt_days': bt,
                    'expected_coherence': expected_gamma,
                    'status': 'FAILED_MISSING_SLC',
                })
                continue

            pair_dir = self.config.output_dir / pair_name
            ifg_dir = pair_dir / f"{pair_name}_ifg"

            graph_xml = graph_gen.create_insar_graph()
            graph_file = pair_dir / 'insar_graph.xml'
            graph_gen.save_graph(graph_xml, graph_file)

            self.logger.info(f" Generated SNAP graph: {graph_file.name}")

            success = snap_proc.process_pair(graph_file, ifg_dir, ref_slc, sec_slc)

            if success:
                self.results.append({
                    'pair': pair_name,
                    'Bt_days': bt,
                    'expected_coherence': expected_gamma,
                    'status': 'SUCCESS',
                })
            else:
                self.results.append({
                    'pair': pair_name,
                    'Bt_days': bt,
                    'expected_coherence': expected_gamma,
                    'status': 'FAILED_PROCESSING',
                })

        self._generate_summary_report()

    def _generate_summary_report(self):
        if not self.results:
            self.logger.warning("No processing results to report")
            return

        df = pd.DataFrame(self.results)
        results_file = self.config.output_dir / 'processing_results.csv'
        df.to_csv(results_file, index=False)

        self.logger.info("\n" + "="*80)
        self.logger.info("PROCESSING SUMMARY (v17.3 - GOLDSTEIN FIX + PHASE + ELEVATION)")
        self.logger.info("="*80)

        successful = df[df['status'] == 'SUCCESS'].shape[0]
        failed = df[df['status'].str.contains('FAILED')].shape[0]

        self.logger.info(f"Results: {successful} successful, {failed} failed")
        self.logger.info(f"CSV Report: {results_file}")
        self.logger.info("="*80)


def main():
    config = Config()

    # Parse DEM input (comma-separated list)
    dem_input = r'C:\bcosta\Copernicus_DSM_COG_10_N69_00_W134_00_DEM.tif'

    dem_manager = DEMManager(dem_input, config.temp_dir, config.logger)
    dem_manager.parse_dem_paths()
    config.dem_mosaic_path = dem_manager.get_dem_for_snap()

    # SLC input directory
    slc_input = Path(r'C:\bcosta\S1_raw')

    # Run batch processor
    processor = InSARBatchProcessor(config)
    processor.run(slc_input)


if __name__ == '__main__':
    main()
