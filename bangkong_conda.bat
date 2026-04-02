@echo off
REM Bangkong LLM Training System - Conda Environment Launcher
REM This script activates conda environment and runs Bangkong

title Bangkong LLM Training System (Conda)

REM ==============================================================================
REM CONFIGURATION - Edit these settings
REM ==============================================================================

REM Set your conda installation path (usually one of these):
set "CONDA_PATH=C:\Users\%USERNAME%\miniconda3"
REM set "CONDA_PATH=C:\Users\%USERNAME%\Anaconda3"
REM set "CONDA_PATH=C:\ProgramData\miniconda3"

REM Set your conda environment name
set "CONDA_ENV_NAME=webui"
REM set "CONDA_ENV_NAME=bangkong"
REM set "CONDA_ENV_NAME=torch"

REM ==============================================================================
REM DO NOT EDIT BELOW THIS LINE
REM ==============================================================================

echo.
echo ================================================================================
echo   BANGKONG LLM TRAINING SYSTEM - CONDA LAUNCHER
echo ================================================================================
echo.
echo Conda Path: %CONDA_PATH%
echo Environment: %CONDA_ENV_NAME%
echo.

REM Check if conda path exists
if not exist "%CONDA_PATH%" (
    echo ERROR: Conda installation not found at: %CONDA_PATH%
    echo.
    echo Please edit this file and set the correct CONDA_PATH
    echo Common paths:
    echo   C:\Users\%%USERNAME%%\miniconda3
    echo   C:\Users\%%USERNAME%%\Anaconda3
    echo   C:\ProgramData\miniconda3
    echo.
    pause
    exit /b 1
)

REM Activate conda environment
echo Activating conda environment: %CONDA_ENV_NAME%...
call "%CONDA_PATH%\Scripts\activate.bat"

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    echo.
    echo Please ensure:
    echo   1. Conda is installed at: %CONDA_PATH%
    echo   2. Environment '%CONDA_ENV_NAME%' exists
    echo.
    echo To create the environment, run:
    echo   conda create -n %CONDA_ENV_NAME% python=3.10 pytorch torchvision torchaudio -c pytorch
    echo.
    pause
    exit /b 1
)

REM Activate the specific environment
call conda activate %CONDA_ENV_NAME%

if errorlevel 1 (
    echo ERROR: Failed to activate conda environment: %CONDA_ENV_NAME%
    echo.
    echo To list available environments:
    echo   conda env list
    echo.
    echo To create this environment:
    echo   conda create -n %CONDA_ENV_NAME% python=3.10
    echo   conda activate %CONDA_ENV_NAME%
    echo   pip install torch transformers pydantic pyyaml python-dotenv tqdm
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   CONDA ENVIRONMENT ACTIVATED SUCCESSFULLY!
echo ================================================================================
echo.
echo Python: 
python --version
echo.
echo Conda Environment: %CONDA_ENV_NAME%
echo.
echo Starting Bangkong LLM Training System...
echo.
echo ================================================================================
echo.

REM Run the main bangkong.bat script
call bangkong.bat

REM Deactivate conda environment when done
echo.
echo Deactivating conda environment...
call conda deactivate

echo.
echo ================================================================================
echo   BANGKONG SESSION ENDED
echo ================================================================================
echo.
pause
