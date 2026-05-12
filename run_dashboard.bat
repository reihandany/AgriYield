@echo off
echo.
echo ==========================================
echo  Dashboard Prediksi Produktivitas Pertanian
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python tidak terinstall atau tidak ada di PATH
    echo Silakan install Python terlebih dahulu dari https://www.python.org/
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
pip list | findstr /i "streamlit" >nul
if errorlevel 1 (
    echo.
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Gagal menginstall dependencies
        pause
        exit /b 1
    )
)

echo.
echo ✓ Dependencies OK
echo.
echo Memulai dashboard...
echo Dashboard akan dibuka di: http://localhost:8501
echo.
echo Press Ctrl+C untuk menghentikan dashboard
echo.

streamlit run app.py

pause
