@echo off
REM Bangkong LLM Training System - Windows Launcher
REM This script provides an interactive interface for running Bangkong operations

title Bangkong LLM Training System
setlocal enabledelayedexpansion

:main_menu
cls
echo ==================================================
echo        Bangkong LLM Training System
echo ==================================================
echo.
echo Welcome to the Bangkong LLM Training System!
echo.
echo Please select an operation:
echo.
echo 1. Train a model
echo 2. Evaluate a model
echo 3. Convert a model
echo 4. Deploy a model
echo 5. Process raw data
echo 6. Run tests
echo 7. Check system information
echo 8. View documentation
echo 9. Test installation
echo 10. Exit
echo.
echo ==================================================

set /p choice=Enter your choice (1-10): 

if "%choice%"=="1" goto train_menu
if "%choice%"=="2" goto evaluate_model
if "%choice%"=="3" goto convert_model
if "%choice%"=="4" goto deploy_model
if "%choice%"=="5" goto process_data
if "%choice%"=="6" goto run_tests
if "%choice%"=="7" goto system_info
if "%choice%"=="8" goto view_docs
if "%choice%"=="9" goto test_installation
if "%choice%"=="10" goto exit_script

echo.
echo Invalid choice. Please enter a number between 1 and 10.
echo.
pause
goto main_menu

:train_menu
cls
echo ==================================================
echo              Training Menu
echo ==================================================
echo.
echo Please select training mode:
echo.
echo 1. Fresh Training (Start new model from scratch)
echo 2. Resume Training (Continue from checkpoint)
echo 3. Continue Training (Add more epochs to completed model)
echo 4. Fine-tune Model (Adapt existing model to new data)
echo 5. Back to main menu
echo.
echo ==================================================

set /p train_choice=Enter your choice (1-5): 

if "%train_choice%"=="1" goto train_fresh
if "%train_choice%"=="2" goto train_resume
if "%train_choice%"=="3" goto train_continue
if "%train_choice%"=="4" goto train_fine_tune
if "%train_choice%"=="5" goto main_menu

echo.
echo Invalid choice. Please enter a number between 1 and 5.
echo.
pause
goto train_menu

:train_fresh
cls
echo ==================================================
echo              Fresh Training
echo ==================================================
echo.
echo This will start training a new model from scratch.
echo.
echo Current working directory: %CD%
echo.
echo Note: You will see progress bars during training.
echo.

set CONFIG_FILE=
set /p CUSTOM_CONFIG=Enter custom config path (or press Enter for auto-detect): 

if defined CUSTOM_CONFIG (
    if exist "%CUSTOM_CONFIG%" (
        set CONFIG_FILE=--config %CUSTOM_CONFIG%
        echo Using custom configuration: %CUSTOM_CONFIG%
    ) else (
        echo Configuration file not found: %CUSTOM_CONFIG%
        echo Using auto-detected configuration
    )
)

echo.
echo ============ Select data training ==================
echo.

REM Scan for available JSONL files in data/processed
set FILE_COUNTER=0
if exist "data\processed\*.jsonl" (
    for %%f in (data\processed\*.jsonl) do (
        set /a FILE_COUNTER+=1
        echo !FILE_COUNTER!. %%~nxf
    )
)

if %FILE_COUNTER% EQU 0 (
    echo No JSONL files found in data/processed
    echo Please process data first using option 5
    echo.
    pause
    goto train_menu
)

echo.
set /p DATA_CHOICE=Enter your choice (1-%FILE_COUNTER%): 

REM Validate choice and find the selected file
set SELECTED_FILE=
set DISPLAY_NAME=
set CURRENT_COUNT=0

REM First validate the choice
if "%DATA_CHOICE%"=="" set DATA_CHOICE=1
if %DATA_CHOICE% LSS 1 set DATA_CHOICE=1
if %DATA_CHOICE% GTR %FILE_COUNTER% set DATA_CHOICE=1

REM Now find the selected file
for %%f in (data\processed\*.jsonl) do (
    set /a CURRENT_COUNT+=1
    if !CURRENT_COUNT! EQU %DATA_CHOICE% (
        set SELECTED_FILE=%%f
        set DISPLAY_NAME=%%~nxf
    )
)

REM If for some reason we didn't find a file, use the first one
if not defined SELECTED_FILE (
    echo Invalid choice or error finding file. Using first available file.
    for %%f in (data\processed\*.jsonl) do (
        if not defined SELECTED_FILE (
            set SELECTED_FILE=%%f
            set DISPLAY_NAME=%%~nxf
        )
    )
)

echo.
echo Selected training data: %DISPLAY_NAME%
set DATA_PATH=--data-path %SELECTED_FILE%

echo.
set OUTPUT_PATH=
set /p CUSTOM_OUTPUT=Enter output path (or press Enter for auto-detect): 

if defined CUSTOM_OUTPUT (
    set OUTPUT_PATH=--output-path %CUSTOM_OUTPUT%
    echo Using custom output path: %CUSTOM_OUTPUT%
) else (
    echo Using auto-detected output path
)

echo.
echo Starting fresh training process...
echo.
echo You will see progress bars during training.
echo Press Ctrl+C to interrupt training if needed.
echo.
pause

python -u scripts/train.py %CONFIG_FILE% %DATA_PATH% %OUTPUT_PATH% --training-mode fresh

echo.
echo Training process completed.
echo.
pause
goto train_menu

:train_resume
cls
echo ==================================================
echo              Resume Training
echo ==================================================
echo.
echo This will resume training from a saved checkpoint.
echo.
echo Note: You will see progress bars during training.
echo.

REM First, let user select a model
call :select_model MODEL_PATH SELECTED_MODEL_NAME
if not defined MODEL_PATH goto train_menu

echo.
echo Selected model: %SELECTED_MODEL_NAME%

REM Now, let user select a checkpoint
call :select_checkpoint "%MODEL_PATH%" CHECKPOINT_PATH CHECKPOINT_NAME
if not defined CHECKPOINT_PATH goto train_menu

echo.
echo Selected checkpoint: %CHECKPOINT_NAME%

echo.
set CONFIG_FILE=
set /p CUSTOM_CONFIG=Enter custom config path (or press Enter for auto-detect): 

if defined CUSTOM_CONFIG (
    if exist "%CUSTOM_CONFIG%" (
        set CONFIG_FILE=--config %CUSTOM_CONFIG%
        echo Using custom configuration: %CUSTOM_CONFIG%
    ) else (
        echo Configuration file not found: %CUSTOM_CONFIG%
        echo Using auto-detected configuration
    )
)

echo.
set OUTPUT_PATH=
set /p CUSTOM_OUTPUT=Enter output path (or press Enter for auto-detect): 

if defined CUSTOM_OUTPUT (
    set OUTPUT_PATH=--output-path %CUSTOM_OUTPUT%
    echo Using custom output path: %CUSTOM_OUTPUT%
) else (
    echo Using auto-detected output path
)

echo.
echo Starting resume training process...
echo.
echo You will see progress bars during training.
echo Press Ctrl+C to interrupt training if needed.
echo.
pause

python -u scripts/train.py %CONFIG_FILE% %OUTPUT_PATH% --training-mode resume --model-path "%MODEL_PATH%" --checkpoint-path "%CHECKPOINT_PATH%"

echo.
echo Training process completed.
echo.
pause
goto train_menu

:train_continue
cls
echo ==================================================
echo              Continue Training
echo ==================================================
echo.
echo This will continue training a completed model for additional epochs.
echo.
echo Note: You will see progress bars during training.
echo.

REM Let user select a model
call :select_model MODEL_PATH SELECTED_MODEL_NAME
if not defined MODEL_PATH goto train_menu

echo.
echo Selected model: %SELECTED_MODEL_NAME%

echo.
set CONFIG_FILE=
set /p CUSTOM_CONFIG=Enter custom config path (or press Enter for auto-detect): 

if defined CUSTOM_CONFIG (
    if exist "%CUSTOM_CONFIG%" (
        set CONFIG_FILE=--config %CUSTOM_CONFIG%
        echo Using custom configuration: %CUSTOM_CONFIG%
    ) else (
        echo Configuration file not found: %CUSTOM_CONFIG%
        echo Using auto-detected configuration
    )
)

echo.
set OUTPUT_PATH=
set /p CUSTOM_OUTPUT=Enter output path (or press Enter for auto-detect): 

if defined CUSTOM_OUTPUT (
    set OUTPUT_PATH=--output-path %CUSTOM_OUTPUT%
    echo Using custom output path: %CUSTOM_OUTPUT%
) else (
    echo Using auto-detected output path
)

echo.
echo Starting continue training process...
echo.
echo You will see progress bars during training.
echo Press Ctrl+C to interrupt training if needed.
echo.
pause

python -u scripts/train.py %CONFIG_FILE% %OUTPUT_PATH% --training-mode continue --model-path "%MODEL_PATH%"

echo.
echo Training process completed.
echo.
pause
goto train_menu

:train_fine_tune
cls
echo ==================================================
echo              Fine-tune Model
echo ==================================================
echo.
echo This will fine-tune an existing model on new data.
echo.
echo Note: You will see progress bars during training.
echo.

REM Let user select a model
call :select_model MODEL_PATH SELECTED_MODEL_NAME
if not defined MODEL_PATH goto train_menu

echo.
echo Selected model: %SELECTED_MODEL_NAME%

echo.
echo ============ Select fine-tuning data ==================
echo.

REM Scan for available JSONL files in data/processed
set FILE_COUNTER=0
if exist "data\processed\*.jsonl" (
    for %%f in (data\processed\*.jsonl) do (
        set /a FILE_COUNTER+=1
        echo !FILE_COUNTER!. %%~nxf
    )
)

if %FILE_COUNTER% EQU 0 (
    echo No JSONL files found in data/processed
    echo Please process data first using option 5
    echo.
    pause
    goto train_menu
)

echo.
set /p DATA_CHOICE=Enter your choice (1-%FILE_COUNTER%): 

REM Validate choice and find the selected file
set SELECTED_FILE=
set DISPLAY_NAME=
set CURRENT_COUNT=0

REM First validate the choice
if "%DATA_CHOICE%"=="" set DATA_CHOICE=1
if %DATA_CHOICE% LSS 1 set DATA_CHOICE=1
if %DATA_CHOICE% GTR %FILE_COUNTER% set DATA_CHOICE=1

REM Now find the selected file
for %%f in (data\processed\*.jsonl) do (
    set /a CURRENT_COUNT+=1
    if !CURRENT_COUNT! EQU %DATA_CHOICE% (
        set SELECTED_FILE=%%f
        set DISPLAY_NAME=%%~nxf
    )
)

REM If for some reason we didn't find a file, use the first one
if not defined SELECTED_FILE (
    echo Invalid choice or error finding file. Using first available file.
    for %%f in (data\processed\*.jsonl) do (
        if not defined SELECTED_FILE (
            set SELECTED_FILE=%%f
            set DISPLAY_NAME=%%~nxf
        )
    )
)

echo.
echo Selected fine-tuning data: %DISPLAY_NAME%
set DATA_PATH=--data-path %SELECTED_FILE%

echo.
set CONFIG_FILE=
set /p CUSTOM_CONFIG=Enter custom config path (or press Enter for auto-detect): 

if defined CUSTOM_CONFIG (
    if exist "%CUSTOM_CONFIG%" (
        set CONFIG_FILE=--config %CUSTOM_CONFIG%
        echo Using custom configuration: %CUSTOM_CONFIG%
    ) else (
        echo Configuration file not found: %CUSTOM_CONFIG%
        echo Using auto-detected configuration
    )
)

echo.
set OUTPUT_PATH=
set /p CUSTOM_OUTPUT=Enter output path (or press Enter for auto-detect): 

if defined CUSTOM_OUTPUT (
    set OUTPUT_PATH=--output-path %CUSTOM_OUTPUT%
    echo Using custom output path: %CUSTOM_OUTPUT%
) else (
    echo Using auto-detected output path
)

echo.
echo Starting fine-tuning process...
echo.
echo You will see progress bars during training.
echo Press Ctrl+C to interrupt training if needed.
echo.
pause

python -u scripts/train.py %CONFIG_FILE% %DATA_PATH% %OUTPUT_PATH% --training-mode fine-tune --model-path "%MODEL_PATH%"

echo.
echo Training process completed.
echo.
pause
goto train_menu

:select_model
REM Function to select a model
REM Returns: MODEL_PATH and MODEL_NAME
cls
echo.
echo ============ Select Model ==================
echo.

REM Scan for available models
set MODEL_COUNTER=0
if exist "models\*" (
    for /d %%d in (models\*) do (
        if exist "%%d\config.json" (
            set /a MODEL_COUNTER+=1
            echo !MODEL_COUNTER!. %%~nxd
            set MODEL_NAME_!MODEL_COUNTER!=%%~nxd
            set MODEL_PATH_!MODEL_COUNTER!=%%d
        ) else if exist "%%d\pytorch_model.bin" (
            set /a MODEL_COUNTER+=1
            echo !MODEL_COUNTER!. %%~nxd
            set MODEL_NAME_!MODEL_COUNTER!=%%~nxd
            set MODEL_PATH_!MODEL_COUNTER!=%%d
        )
    )
)

if %MODEL_COUNTER% EQU 0 (
    echo No trained models found in models directory.
    echo Please train a model first.
    echo.
    set %1=
    set %2=
    exit /b
)

echo.
set /p MODEL_CHOICE=Enter your choice (1-%MODEL_COUNTER%): 

REM Validate choice
if "%MODEL_CHOICE%"=="" set MODEL_CHOICE=1
if %MODEL_CHOICE% LSS 1 set MODEL_CHOICE=1
if %MODEL_CHOICE% GTR %MODEL_COUNTER% set MODEL_CHOICE=1

REM Get selected model
set SELECTED_MODEL_NAME=!MODEL_NAME_%MODEL_CHOICE%!
set SELECTED_MODEL_PATH=!MODEL_PATH_%MODEL_CHOICE%!

echo.
echo Selected model: %SELECTED_MODEL_NAME%
set %1=%SELECTED_MODEL_PATH%
set %2=%SELECTED_MODEL_NAME%
exit /b

:select_checkpoint
REM Function to select a checkpoint
REM Parameters: MODEL_PATH
REM Returns: CHECKPOINT_PATH and CHECKPOINT_NAME
set MODEL_PATH=%~1
cls
echo.
echo ============ Select Checkpoint ==================
echo.

REM Scan for available checkpoints
set CHECKPOINT_COUNTER=0
if exist "%MODEL_PATH%\checkpoints\*.pt" (
    for %%f in ("%MODEL_PATH%\checkpoints\*.pt") do (
        set /a CHECKPOINT_COUNTER+=1
        echo !CHECKPOINT_COUNTER!. %%~nxf
        set CHECKPOINT_NAME_!CHECKPOINT_COUNTER!=%%~nxf
        set CHECKPOINT_PATH_!CHECKPOINT_COUNTER!=%%f
    )
)

if exist "%MODEL_PATH%\*.pt" (
    for %%f in ("%MODEL_PATH%\*.pt") do (
        if /i not "%%~nxf"=="final_checkpoint.pt" (
            set /a CHECKPOINT_COUNTER+=1
            echo !CHECKPOINT_COUNTER!. %%~nxf
            set CHECKPOINT_NAME_!CHECKPOINT_COUNTER!=%%~nxf
            set CHECKPOINT_PATH_!CHECKPOINT_COUNTER!=%%f
        )
    )
)

if %CHECKPOINT_COUNTER% EQU 0 (
    echo No checkpoints found for this model.
    echo.
    set %2=
    set %3=
    exit /b
)

echo.
set /p CHECKPOINT_CHOICE=Enter your choice (1-%CHECKPOINT_COUNTER%): 

REM Validate choice
if "%CHECKPOINT_CHOICE%"=="" set CHECKPOINT_CHOICE=1
if %CHECKPOINT_CHOICE% LSS 1 set CHECKPOINT_CHOICE=1
if %CHECKPOINT_CHOICE% GTR %CHECKPOINT_COUNTER% set CHECKPOINT_CHOICE=1

REM Get selected checkpoint
set SELECTED_CHECKPOINT_NAME=!CHECKPOINT_NAME_%CHECKPOINT_CHOICE%!
set SELECTED_CHECKPOINT_PATH=!CHECKPOINT_PATH_%CHECKPOINT_CHOICE%!

echo.
echo Selected checkpoint: %SELECTED_CHECKPOINT_NAME%
set %2=%SELECTED_CHECKPOINT_PATH%
set %3=%SELECTED_CHECKPOINT_NAME%
exit /b

:evaluate_model
cls
echo ==================================================
echo              Evaluate a Model
echo ==================================================
echo.
echo This will evaluate a trained model.
echo.
set /p CUSTOM_MODEL=Enter path to model for evaluation (or press Enter for auto-detect): 

if defined CUSTOM_MODEL (
    if exist "%CUSTOM_MODEL%" (
        set MODEL_PATH=--model-path %CUSTOM_MODEL%
        echo Using custom model path: %CUSTOM_MODEL%
    ) else (
        echo Model path not found: %CUSTOM_MODEL%
        echo Please provide a valid model path.
        echo.
        pause
        goto evaluate_model
    )
) else (
    echo Using auto-detected model path
    set MODEL_PATH=
)

echo.
set CONFIG_FILE=
set /p CUSTOM_CONFIG=Enter path to configuration file (or press Enter for auto-detect): 

if defined CUSTOM_CONFIG (
    if exist "%CUSTOM_CONFIG%" (
        set CONFIG_FILE=--config %CUSTOM_CONFIG%
        echo Using custom configuration: %CUSTOM_CONFIG%
    ) else (
        echo Configuration file not found: %CUSTOM_CONFIG%
        echo Using auto-detected configuration
    )
)

echo.
set DATA_PATH=
set /p CUSTOM_DATA=Enter path to evaluation data (or press Enter for auto-detect): 

if defined CUSTOM_DATA (
    if exist "%CUSTOM_DATA%" (
        set DATA_PATH=--data-path %CUSTOM_DATA%
        echo Using custom data path: %CUSTOM_DATA%
    ) else (
        echo Data path not found: %CUSTOM_DATA%
        echo Using auto-detected data path
    )
)

echo.
echo Starting evaluation process...
echo.
echo Press any key to start evaluation...
pause >nul

python -u scripts/evaluate.py %MODEL_PATH% %CONFIG_FILE% %DATA_PATH%

echo.
echo Evaluation process completed.
echo.
pause
goto main_menu

:convert_model
cls
echo ==================================================
echo              Convert a Model
echo ==================================================
echo.
echo This will convert a model to different formats.
echo.
set /p CUSTOM_MODEL=Enter path to model for conversion (or press Enter for auto-detect): 

if defined CUSTOM_MODEL (
    if exist "%CUSTOM_MODEL%" (
        set MODEL_PATH=--model-path %CUSTOM_MODEL%
        echo Using custom model path: %CUSTOM_MODEL%
    ) else (
        echo Model path not found: %CUSTOM_MODEL%
        echo Please provide a valid model path.
        echo.
        pause
        goto convert_model
    )
) else (
    echo Using auto-detected model path
    set MODEL_PATH=
)

echo.
set CONFIG_FILE=
set /p CUSTOM_CONFIG=Enter path to configuration file (or press Enter for auto-detect): 

if defined CUSTOM_CONFIG (
    if exist "%CUSTOM_CONFIG%" (
        set CONFIG_FILE=--config %CUSTOM_CONFIG%
        echo Using custom configuration: %CUSTOM_CONFIG%
    ) else (
        echo Configuration file not found: %CUSTOM_CONFIG%
        echo Using auto-detected configuration
    )
)

echo.
set OUTPUT_PATH=
set /p CUSTOM_OUTPUT=Enter path to save converted model (or press Enter for auto-detect): 

if defined CUSTOM_OUTPUT (
    set OUTPUT_PATH=--output-path %CUSTOM_OUTPUT%
    echo Using custom output path: %CUSTOM_OUTPUT%
) else (
    echo Using auto-detected output path
)

echo.
echo Available formats:
echo 1. PyTorch (default)
echo 2. ONNX
echo 3. SafeTensors
echo 4. All formats
echo.
set /p FORMAT_CHOICE=Select format (1-4, or press Enter for default): 

if "%FORMAT_CHOICE%"=="1" (
    set FORMATS=--formats pytorch
) else if "%FORMAT_CHOICE%"=="2" (
    set FORMATS=--formats onnx
) else if "%FORMAT_CHOICE%"=="3" (
    set FORMATS=--formats safetensors
) else if "%FORMAT_CHOICE%"=="4" (
    set FORMATS=--formats pytorch onnx safetensors
) else (
    set FORMATS=
)

echo.
echo Starting conversion process...
echo.
echo Press any key to start conversion...
pause >nul

python -u scripts/convert.py %MODEL_PATH% %CONFIG_FILE% %OUTPUT_PATH% %FORMATS%

echo.
echo Conversion process completed.
echo.
pause
goto main_menu

:deploy_model
cls
echo ==================================================
echo              Deploy a Model
echo ==================================================
echo.
echo This will deploy a model to a target environment.
echo.
set /p CUSTOM_MODEL=Enter path to model for deployment (or press Enter for auto-detect): 

if defined CUSTOM_MODEL (
    if exist "%CUSTOM_MODEL%" (
        set MODEL_PATH=--model-path %CUSTOM_MODEL%
        echo Using custom model path: %CUSTOM_MODEL%
    ) else (
        echo Model path not found: %CUSTOM_MODEL%
        echo Please provide a valid model path.
        echo.
        pause
        goto deploy_model
    )
) else (
    echo Using auto-detected model path
    set MODEL_PATH=
)

echo.
set CONFIG_FILE=
set /p CUSTOM_CONFIG=Enter path to configuration file (or press Enter for auto-detect): 

if defined CUSTOM_CONFIG (
    if exist "%CUSTOM_CONFIG%" (
        set CONFIG_FILE=--config %CUSTOM_CONFIG%
        echo Using custom configuration: %CUSTOM_CONFIG%
    ) else (
        echo Configuration file not found: %CUSTOM_CONFIG%
        echo Using auto-detected configuration
    )
)

echo.
echo Available deployment targets:
echo 1. Local (default)
echo 2. Cloud
echo 3. Hybrid
echo.
set /p TARGET_CHOICE=Select target (1-3, or press Enter for default): 

if "%TARGET_CHOICE%"=="1" (
    set TARGET=--target local
) else if "%TARGET_CHOICE%"=="2" (
    set TARGET=--target cloud
) else if "%TARGET_CHOICE%"=="3" (
    set TARGET=--target hybrid
) else (
    set TARGET=--target local
)

echo.
echo Starting deployment process...
echo.
echo Press any key to start deployment...
pause >nul

python -u scripts/deploy.py %MODEL_PATH% %CONFIG_FILE% %TARGET%

echo.
echo Deployment process completed.
echo.
pause
goto main_menu

:run_tests
cls
echo ==================================================
echo              Run Tests
echo ==================================================
echo.
echo This will run the test suite.
echo.
echo Test options:
echo 1. Run all tests (default)
echo 2. Run unit tests
echo 3. Run integration tests
echo 4. Run end-to-end tests
echo.
set /p TEST_CHOICE=Select test type (1-4, or press Enter for all): 

echo.
echo Starting tests...
echo.

if "%TEST_CHOICE%"=="1" (
    python -u -m pytest 2>nul
    if errorlevel 1 (
        echo Tests encountered issues or pytest is not installed.
        echo Please ensure pytest is installed: pip install pytest
    )
) else if "%TEST_CHOICE%"=="2" (
    if exist "tests\unit" (
        python -u -m pytest tests/unit/ 2>nul
        if errorlevel 1 (
            echo Unit tests encountered issues.
        )
    ) else (
        echo Unit tests directory not found.
    )
) else if "%TEST_CHOICE%"=="3" (
    if exist "tests\integration" (
        python -u -m pytest tests/integration/ 2>nul
        if errorlevel 1 (
            echo Integration tests encountered issues.
        )
    ) else (
        echo Integration tests directory not found.
    )
) else if "%TEST_CHOICE%"=="4" (
    if exist "tests\e2e" (
        python -u -m pytest tests/e2e/ 2>nul
        if errorlevel 1 (
            echo End-to-end tests encountered issues.
        )
    ) else (
        echo End-to-end tests directory not found.
    )
) else (
    python -u -m pytest 2>nul
    if errorlevel 1 (
        echo Tests encountered issues or pytest is not installed.
        echo Please ensure pytest is installed: pip install pytest
    )
)

echo.
echo Test execution attempt completed.
echo.
pause
goto main_menu

:system_info
cls
echo ==================================================
echo              System Information
echo ==================================================
echo.
echo Checking system information...
echo.

echo Python version:
python -u -c "import sys; print(sys.version)" 2>nul || echo Python not found or import error
echo.

echo PyTorch information:
python -u -c "import torch; print('Version:', torch.__version__)" 2>nul || echo PyTorch not installed
python -u -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul || echo CUDA check failed
python -u -c "import torch; print('CUDA version:', torch.version.cuda) if torch.cuda.is_available() else print('CUDA not available')" 2>nul || echo CUDA version check failed
echo.

echo System resources:
python -u -c "import psutil; print('CPU cores:', psutil.cpu_count()); print('Memory:', psutil.virtual_memory().total // (1024**3), 'GB')" 2>nul || echo psutil not installed
echo.

echo Environment variables:
set BANGKONG_ 2>nul || echo No BANGKONG_ environment variables found
echo.

echo Available configuration files:
if exist "configs\*.yaml" (
    dir configs\*.yaml 2>nul
) else (
    echo No configuration files found in configs directory
)
echo.

pause
goto main_menu

:view_docs
cls
echo ==================================================
echo              Documentation
echo ==================================================
echo.
echo Available documentation:
echo.
echo 1. README.md - Project overview
echo 2. docs/architecture.md - System architecture
echo 3. docs/configuration.md - Configuration guide
echo 4. docs/deployment.md - Deployment guide
echo 5. docs/development.md - Development guide
echo 6. docs/contributing.md - Contribution guide
echo 7. docs/api/openapi.yaml - API specification
echo.
echo 8. Back to main menu
echo.
set /p doc_choice=Select documentation to view (1-8): 

if "%doc_choice%"=="1" (
    if exist "README.md" (
        type README.md | more
    ) else (
        echo README.md not found.
    )
) else if "%doc_choice%"=="2" (
    if exist "docs/architecture.md" (
        type docs/architecture.md | more
    ) else (
        echo docs/architecture.md not found.
    )
) else if "%doc_choice%"=="3" (
    if exist "docs/configuration.md" (
        type docs/configuration.md | more
    ) else (
        echo docs/configuration.md not found.
    )
) else if "%doc_choice%"=="4" (
    if exist "docs/deployment.md" (
        type docs/deployment.md | more
    ) else (
        echo docs/deployment.md not found.
    )
) else if "%doc_choice%"=="5" (
    if exist "docs/development.md" (
        type docs/development.md | more
    ) else (
        echo docs/development.md not found.
    )
) else if "%doc_choice%"=="6" (
    if exist "docs/contributing.md" (
        type docs/contributing.md | more
    ) else (
        echo docs/contributing.md not found.
    )
) else if "%doc_choice%"=="7" (
    if exist "docs/api/openapi.yaml" (
        type docs/api/openapi.yaml | more
    ) else (
        echo docs/api/openapi.yaml not found.
    )
) else if "%doc_choice%"=="8" (
    goto main_menu
) else (
    echo Invalid choice.
)

echo.
pause
goto main_menu

:test_installation
cls
echo ==================================================
echo              Test Installation
echo ==================================================
echo.
echo This will test if Bangkong is properly installed.
echo.
echo Press any key to start the installation test...
pause >nul

python -u test_installation.py

echo.
echo Installation test completed.
echo.
pause
goto main_menu

:process_data
cls
echo ==================================================
echo              Process Raw Data
echo ==================================================
echo.
echo This will process raw data through the preprocessing pipeline.
echo.
echo Data flow:
echo 1. Process data for training
echo 2. Process and clean data
echo 3. Back to main menu
echo.
echo ==================================================
set /p choice=Enter your choice (1-3): 

if "%choice%"=="1" goto organize_data
if "%choice%"=="2" goto clean_data
if "%choice%"=="3" goto main_menu

echo.
echo Invalid choice. Please enter a number between 1 and 3.
echo.
pause
goto process_data

:organize_data
cls
echo ==================================================
echo              Organize Raw Data
echo ==================================================
echo.
echo This will scan all files in data/raw including subfolders,
echo categorize by file type, and organize into data/organized.
echo.
echo Press any key to start organizing...
pause >nul

echo.
echo Starting data organization...
python -u scripts/organize_data.py

echo.
echo Data organization completed.
echo You can now review and filter files in data/organized
echo before proceeding to cleaning step.
echo.
pause
goto process_data

:clean_data
cls
echo ==================================================
echo              Clean and Process Data
echo ==================================================
echo.
echo This will scan all files in data/organized including subfolders,
echo and convert them into clean JSONL files in data/processed.
echo.
echo Press any key to start cleaning...
pause >nul

echo.
echo Starting data cleaning and processing...
python -u scripts/clean_data.py

echo.
echo Data cleaning and processing completed.
echo Training-ready JSONL files are now in data/processed
echo.
pause
goto process_data

:exit_script
cls
echo ==================================================
echo        Bangkong LLM Training System
echo ==================================================
echo.
echo Thank you for using Bangkong!
echo.
echo Exiting...
echo.
timeout /t 2 /nobreak >nul
exit /b