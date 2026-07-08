@echo off

REM Sentinel-1 InSAR Batch Processing with SNAP 12+ - PRODUCTION FINAL v8

echo.
echo ================================================================================
echo Sentinel-1 InSAR Batch Processing with SNAP 12+ - PRODUCTION FINAL v8
echo ================================================================================ 
echo.

python retry_failed_pairs.py ^
  --slc-dir C:\bcosta\S1_raw ^
  --output-dir C:\bcosta\DS-InSAR ^
  --results-csv C:\bcosta\DS-InSAR\processing_results.csv

echo.
echo ================================================================================

if %ERRORLEVEL% EQU 0 (
  echo ✓ PROCESSING COMPLETE!
  echo ✓ Results saved to: C:\bcosta\DS-InSAR
  echo ✓ Log file: C:\bcosta\DS-InSAR\processing.log
  echo ✓ Summary: C:\bcosta\DS-InSAR\processing_results.csv
) else (
  echo ✗ PROCESSING FAILED - Check log file for errors
)

echo ================================================================================
echo.
pause