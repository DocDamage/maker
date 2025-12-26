@echo off
REM MusicGen LoRA Training Pipeline - Quick Setup Script
REM Run this in the musicmaker directory

echo ========================================
echo MusicGen LoRA Training Pipeline Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10 or 3.11
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [4/5] Installing dependencies...
pip install -r requirements.txt

echo [5/5] Verifying installation...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Add your music files to: data\raw\
echo 2. Run: python scripts\preprocess.py
echo 3. Run: python scripts\autolabel.py
echo 4. Run: python scripts\train.py
echo 5. Run: python scripts\generate.py --prompt "your style"
echo.
pause
