@echo off
rem Registration Studio v1 - Windows launcher. Double-click this file.
rem First start: creates the conda env "reg_env" (or a venv in %%USERPROFILE%%\.regstudio_venv without conda)
rem and installs PyTorch for this PC (CUDA if an NVIDIA driver is present, else CPU). Then opens http://127.0.0.1:8780/
rem Extra arguments go to launch.py, e.g.  start_windows.bat --port 8790
setlocal EnableExtensions
cd /d "%~dp0"
set "ENV=reg_env"
set "CONDA="
where conda >nul 2>nul && for /f "delims=" %%i in ('where conda') do if not defined CONDA set "CONDA=%%i"
for %%d in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%ProgramData%\miniforge3" "%ProgramData%\Anaconda3" "C:\miniforge3") do (
  if not defined CONDA if exist "%%~d\Scripts\conda.exe" set "CONDA=%%~d\Scripts\conda.exe"
  if not defined CONDA if exist "%%~d\condabin\conda.bat" set "CONDA=%%~d\condabin\conda.bat"
)

if defined CONDA goto :use_conda
echo conda not found - using a Python venv (%USERPROFILE%\.regstudio_venv). Miniforge is recommended: https://conda-forge.org/download/
set "PY=%USERPROFILE%\.regstudio_venv\Scripts\python.exe"
if exist "%PY%" goto :run
where py >nul 2>nul && (py -3 -m venv "%USERPROFILE%\.regstudio_venv") || (python -m venv "%USERPROFILE%\.regstudio_venv")
if not exist "%PY%" (echo Python 3.10+ not found. Install Miniforge or Python from python.org, then double-click again. & goto :fail)
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt || goto :fail
goto :run

:use_conda
call "%CONDA%" env list | findstr /B /C:"%ENV% " >nul
if errorlevel 1 (
  echo Creating conda environment %ENV% ^(first start, a few minutes^)...
  call "%CONDA%" env create -f environment.yml || goto :fail
)
for /f "delims=" %%p in ('call "%CONDA%" run -n %ENV% python -c "import sys; print(sys.executable)"') do set "PY=%%p"
if not defined PY goto :fail

:run
"%PY%" tools\setup_torch.py || goto :fail
set /p VER=<VERSION
echo Starting Registration Studio %VER% ... (keep this window open; close it or press Ctrl+C to stop)
"%PY%" launch.py %*
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo Registration Studio could not start (see the messages above).
pause
exit /b 1
