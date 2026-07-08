@echo off
REM Sentinel-1 InSAR Batch Processing with SNAP 12+ - FIXED
REM Removed Goldstein filter (not available in your SNAP)
REM Output: D:\bcosta4\DS-InSAR

echo.
echo ================================================================================
echo Sentinel-1 InSAR Batch Processing with SNAP 12+ - FIXED
echo ================================================================================
echo.

python snap_insar_batch_fixed.py ^
  --slc-dir "C:\bcosta\S1_raw" ^
  --dem "C:\bcosta\S1_raw\Copernicus_DSM_COG_10_N69_00_W134_00_DEM.tif" ^
  --min-bt 6 ^
  --max-bt 500 ^
  --nlooks-range 8 ^
  --nlooks-az 2 ^
  --ram 8 ^
  --output-dir "D:\bcosta4\DS-InSAR"

echo.
echo ================================================================================
if %ERRORLEVEL% EQU 0 (
    echo ✓ PROCESSING COMPLETE!
    echo ✓ Results saved to: D:\bcosta4\DS-InSAR
    echo ✓ Log file: D:\bcosta4\DS-InSAR\processing.log
    echo ✓ Summary: D:\bcosta4\DS-InSAR\processing_results.csv
) else (
    echo ✗ PROCESSING FAILED - Check log file for errors
)
echo ================================================================================
echo.
pause
