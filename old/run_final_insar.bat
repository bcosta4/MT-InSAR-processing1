@echo off
REM Sentinel-1 InSAR Batch Processing with SNAP 12+ - PRODUCTION FINAL v6
REM Fixed: ESD Window oversampling 2 → 64 (only [32,64,128,256] valid)
REM Output: D:\bcosta4\DS-InSAR

echo.
echo ================================================================================
echo Sentinel-1 InSAR Batch Processing with SNAP 12+ - PRODUCTION FINAL v6
echo ================================================================================
echo.

python snap_insar_v6.py ^
  --slc-dir "C:\bcosta\S1_raw" ^
  --dem "C:\bcosta\S1_raw\Copernicus_DSM_COG_10_N69_00_W134_00_DEM.tif" ^
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
