@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv_run\Scripts\python.exe"

if not exist "%VENV_PY%" (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.10 -m venv "%~dp0.venv_run"
    ) else (
        where python >nul 2>&1
        if errorlevel 1 goto no_python
        python -m venv "%~dp0.venv_run"
    )
    if errorlevel 1 goto setup_failed
)

"%VENV_PY%" -c "import cv2, numpy, ultralytics" >nul 2>&1
if errorlevel 1 (
    echo Memasang dependensi aplikasi. Proses pertama kali mungkin memerlukan waktu...
    "%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 goto setup_failed
)

echo Menjalankan aplikasi...
"%VENV_PY%" "%~dp0test_roboflow.py"
if errorlevel 1 goto run_failed
echo Aplikasi sudah ditutup.
pause
exit /b 0

:no_python
echo Python tidak ditemukan. Install Python 3.10 atau lebih baru, lalu jalankan file ini lagi.
pause
exit /b 1

:setup_failed
echo Persiapan aplikasi gagal. Periksa koneksi internet dan pastikan Python 3.10 atau lebih baru terpasang.
pause
exit /b 1

:run_failed
echo Aplikasi berhenti karena terjadi error. Pesan error ditampilkan di atas.
pause
exit /b 1
