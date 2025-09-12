# SeismicFlow

SeismicFlow is a standalone GUI application for geophysical and well data analysis, designed for scientists and researchers who want flexibility in algorithm development.

## System Requirements

- **Operating System**: Windows, Linux, or macOS (cross-platform compatible)
- **Python Version**: Python 3.10 (required)
- **GPU (Optional)**: NVIDIA GPU with CUDA 11.7 for faster neural network processing
  - **Note**: GPU is NOT required - the application runs perfectly on CPU only
  - GPU acceleration only speeds up neural network algorithms
  - All features work without GPU, just slower for ML tasks

## Required Files

Download ALL of these files from the repository:
- `SeismicFlow.py` (main application)
- `requirements.txt` (dependencies list)
- `install.py` (automated installer)
- `busy.png` (GUI asset)
- `icon.png` (GUI asset)

## Installation Options

Choose **ONE** of the following installation methods:

### Option A: Using Conda (Recommended)

#### Step 1: Install Conda
If you don't have Conda installed:
- Download Anaconda or Miniconda from: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
- Follow the installation instructions for your operating system

#### Step 2: Create Environment
Open terminal/command prompt and run:
```bash
conda create -n seismic_env python=3.10 -y
```

#### Step 3: Activate Environment
```bash
conda activate seismic_env
```

#### Step 4: Navigate to Project Folder
```bash
cd /path/to/your/downloaded/files
```
Replace `/path/to/your/downloaded/files` with the actual path where you saved the SeismicFlow files.

#### Step 5: Run Automated Installer
```bash
python install.py
```

#### Step 6: Launch Application
```bash
python SeismicFlow.py
```

### Option B: Using Python Only (No Conda)

#### Step 1: Install Python 3.10
If you don't have Python 3.10:
- Download from: https://www.python.org/downloads/
- During installation, **CHECK** "Add Python to PATH"

#### Step 2: Create Virtual Environment
Open terminal/command prompt and navigate to your project folder:
```bash
cd /path/to/your/downloaded/files
python -m venv seismic_env
```

#### Step 3: Activate Virtual Environment

**On Windows:**
```bash
seismic_env\Scripts\activate
```

**On Linux/macOS:**
```bash
source seismic_env/bin/activate
```

#### Step 4: Run Automated Installer
```bash
python install.py
```

#### Step 5: Launch Application
```bash
python SeismicFlow.py
```

## GPU Acceleration Setup (Optional)

**For NVIDIA GPU users only:**

1. Check if you have an NVIDIA GPU:
   - Windows: Open Device Manager → Display adapters
   - Linux: Run `lspci | grep -i nvidia`
   - macOS: Apple menu → About This Mac → Graphics

2. If you have NVIDIA GPU, install CUDA Toolkit 11.7:
   - Download from: https://developer.nvidia.com/cuda-11-7-0-download-archive
   - Choose your operating system and follow installation instructions

3. Restart your computer after CUDA installation

**Important**: GPU setup is completely optional. The application works perfectly without GPU acceleration.

## Troubleshooting

### Common Issues and Solutions

**Issue**: "Python not found" error
**Solution**: Make sure Python 3.10 is installed and added to PATH

**Issue**: "pip not found" error  
**Solution**: Reinstall Python 3.10 and ensure "Add to PATH" is checked

**Issue**: Permission errors during installation
**Solution**: 
- Windows: Run terminal as Administrator
- Linux/macOS: Use `sudo` if needed, or install in user directory

**Issue**: Import errors when running SeismicFlow.py
**Solution**: Make sure you activated your environment and ran `python install.py` successfully

**Issue**: Slow performance on neural network operations
**Solution**: This is normal for CPU-only systems. Consider GPU setup for faster processing.

## Verification

After installation, the GUI should launch immediately when you run `python SeismicFlow.py`. You should see the SeismicFlow interface with all menus and tools available.

## Support

If you encounter any issues:
1. Ensure you followed ALL steps in order
2. Verify all required files are in the same directory
3. Check that your Python version is exactly 3.10
4. Make sure your virtual environment is activated before running commands

## Technical Details

- **Framework**: Qt-based GUI (cross-platform compatible)
- **ML Libraries**: PyTorch 2.0.0, TensorFlow 2.10.1
- **CUDA Support**: Version 11.7 (optional)
- **Dependencies**: Automatically installed via `install.py`

The application is designed to work on any system with Python 3.10, regardless of operating system or hardware configuration.
