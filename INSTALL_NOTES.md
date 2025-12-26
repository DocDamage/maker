# Installation Notes for Windows

This guide helps you set up the MusicGen LoRA training pipeline on Windows with an NVIDIA GPU.

## Prerequisites

1. **Python 3.10 or 3.11** (3.12 may have compatibility issues)
   - Download from: <https://www.python.org/downloads/>
   - ✅ Check "Add Python to PATH" during installation

2. **NVIDIA GPU Drivers** (latest version)
   - Download from: <https://www.nvidia.com/drivers>

3. **CUDA Toolkit 12.1**
   - Download from: <https://developer.nvidia.com/cuda-12-1-0-download-archive>
   - Select: Windows > x86_64 > 11 > exe (local)

4. **FFmpeg** (for some audio formats)
   - Option A: `winget install FFmpeg`
   - Option B: Download from <https://ffmpeg.org/download.html>

## Step-by-Step Installation

### 1. Create Virtual Environment

Open PowerShell in the `musicmaker` directory:

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# If you get an execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA

```powershell
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is working:

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 3. Install Main Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Essentia Installation (Optional)

Essentia provides better audio analysis but can be tricky on Windows:

**Option A: Install from wheel (recommended)**

```powershell
# Check for pre-built wheels at:
# https://github.com/MTG/essentia/releases
pip install essentia-tensorflow
```

**Option B: Skip essentia**
The pipeline will work without essentia, using basic librosa analysis instead.

**Option C: Use WSL2**
For full essentia support, consider using WSL2 (Windows Subsystem for Linux):

```powershell
wsl --install
# Then install essentia inside Ubuntu: pip install essentia
```

### 5. Verify Installation

```powershell
python -c "
import torch
import audiocraft
from peft import LoraConfig
print('All core dependencies installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'AudioCraft: {audiocraft.__version__}')
"
```

## Troubleshooting

### "CUDA out of memory" Error

1. Close other GPU applications (games, browsers with hardware acceleration)
2. Reduce batch size in `config.py` to 1
3. Lower LoRA rank from 8 to 4
4. Try `musicgen-small` instead of `musicgen-medium`

### "No module named 'audiocraft'"

```powershell
pip install audiocraft --upgrade
```

### "FFmpeg not found"

```powershell
# Install via winget
winget install FFmpeg

# Or via chocolatey
choco install ffmpeg

# Restart your terminal after installation
```

### Slow Training

1. Make sure you're using GPU (check `torch.cuda.is_available()`)
2. Enable fp16 in config (should be enabled by default)
3. Close background applications
4. Consider using a cloud GPU for faster training

### "DLL load failed" errors

This usually means missing Visual C++ redistributables:

1. Download from: <https://aka.ms/vs/17/release/vc_redist.x64.exe>
2. Install and restart

## Quick Test

After installation, run a quick test:

```powershell
# Test preprocessing (with a sample audio file)
python scripts/preprocess.py --help

# Test auto-labeling
python scripts/autolabel.py --help

# Test generation (downloads model on first run, ~1-2GB)
python scripts/generate.py --prompt "test" --duration 5
```

## Next Steps

1. Add your music files to `data/raw/`
2. Run `python scripts/preprocess.py`
3. Run `python scripts/autolabel.py`
4. Run `python scripts/train.py`
5. Generate with `python scripts/generate.py --prompt "your style"`
