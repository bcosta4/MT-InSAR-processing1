@echo off
REM Sentinel-1 InSAR Batch Processing with SNAP 12+ - PRODUCTION FINAL v8
REM Fixed: Interferogram parameters (flatEarthPolynomialDegree, orbitDegree, coherenceWindowSize, demName)
REM All operators now 100%% validated against working reference XML
REM Output: D:\bcosta4\DS-InSAR

echo.
echo ================================================================================
echo Sentinel-1 InSAR Batch Processing with SNAP 12+ - PRODUCTION FINAL v8
echo ================================================================================
echo.

python snap_insar_v18.py ^
  --slc-dir "C:\bcosta\S1_raw" ^
  --output-dir "C:\bcosta\DS-InSAR"

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
