# SeismicFlow

**The First Production-Grade Python-Native GUI Platform for Integrated Geoscience Workflows**

SeismicFlow is a standalone GUI application for geophysical and well data analysis, designed for scientists and researchers who want flexibility in algorithm development.

📺 **YouTube Channel**: [https://www.youtube.com/@Seismicflow-inc](https://www.youtube.com/@Seismicflow-inc)

## System Requirements

- **Operating System**: Windows 10/11 (64-bit)
- **Python**: Any version of Python to run the installer (installer will set up Python 3.10 automatically)
- **Disk Space**: Approximately 3-4 GB for complete installation
- **GPU (Optional)**: NVIDIA GPU with CUDA 11.7 for faster neural network processing
  - **Note**: GPU is NOT required - the application runs perfectly on CPU only
  - GPU acceleration only speeds up neural network algorithms
  - All features work without GPU, just slower for ML tasks

## Required Files

Download ALL of these files from the repository and place them in the same folder:

### Application Files:
- `SeismicFlow.py` (main application)
- `install.py` (automated installer)
- `requirements.txt` (dependencies list)

### GUI Assets (Required):
- `splash.png`
- `white-terminal.png`
- `black-terminal.png`
- `busy.gif`
- `busy.png`
- `icon.png`

**Important**: All files must be in the same directory for the application to work properly.

## Installation

### Simple One-Step Installation

This installation method is **completely self-contained** and will not affect any existing Python installations on your system.

#### Step 1: Download All Files
Download all the files listed above from this repository and place them in a folder where you want SeismicFlow to be installed (e.g., `C:\SeismicFlow` or `D:\MyApps\SeismicFlow`).

#### Step 2: Run the Installer
Open Command Prompt or Terminal, navigate to your folder, and run:

```bash
python install.py
```

**Note**: You can use ANY version of Python you already have installed to run `install.py`. The installer will automatically download and set up Python 3.10 in a separate, isolated environment within your chosen folder.

#### What the Installer Does:
1. Downloads a portable Python 3.10 installation (no administrator rights needed)
2. Creates an isolated virtual environment
3. Installs PyTorch 2.0.0 with CUDA 11.7 support
4. Installs TensorFlow 2.10.1
5. Installs all other required dependencies from requirements.txt
6. Creates `SeismicFlow.bat` launcher

**Everything is self-contained**: The installer creates a complete Python environment inside your chosen directory. Your system's Python installation and other Python projects remain completely unaffected.

#### Step 3: Launch SeismicFlow
After installation completes, simply double-click:

```
SeismicFlow.bat
```

The application will launch with its own dedicated Python environment.

## How It Works

The installation creates the following structure in your directory:

```
YourFolder/
├── SeismicFlow.py          (main application)
├── install.py              (installer script)
├── requirements.txt        (dependencies)
├── splash.png              (GUI assets)
├── white-terminal.png      (GUI assets)
├── black-terminal.png      (GUI assets)
├── busy.gif                (GUI assets)
├── busy.png                (GUI assets)
├── icon.png                (GUI assets)
├── python310/              (portable Python - created by installer)
├── venv/                   (virtual environment - created by installer)
└── SeismicFlow.bat         (launcher - created by installer)
```

**Key Benefits**:
- No system-wide Python installation required
- No conflicts with other Python projects
- Complete isolation from other environments
- Easy to uninstall: just delete the entire folder
- No registry modifications or system changes
- Works without administrator privileges

## GPU Acceleration Setup (Optional)

**For NVIDIA GPU users only:**

1. Check if you have an NVIDIA GPU:
   - Windows: Open Device Manager → Display adapters
   - Look for "NVIDIA" in the graphics card name

2. If you have an NVIDIA GPU, install CUDA Toolkit 11.7:
   - Download from: https://developer.nvidia.com/cuda-11-7-0-download-archive
   - Choose Windows x86_64 and follow installation instructions
   - Restart your computer after installation

**Important**: GPU setup is completely optional. The application works perfectly without GPU acceleration.

## Troubleshooting

### Common Issues and Solutions

**Issue**: "Python not found" error when running install.py  
**Solution**: You need at least one Python installation (any version) to run the installer. Download Python from python.org if needed.

**Issue**: Installation fails or gets stuck  
**Solution**: 
- Check your internet connection (installer downloads ~2GB of packages)
- Ensure you have sufficient disk space (3-4 GB required)
- Run Command Prompt as Administrator if you encounter permission errors

**Issue**: SeismicFlow.bat doesn't launch the application  
**Solution**: 
- Ensure install.py completed successfully without errors
- Check that all GUI asset files (.png, .gif) are in the same directory
- Try running from Command Prompt to see any error messages

**Issue**: Missing image errors when running SeismicFlow  
**Solution**: Make sure all required .png and .gif files are in the same folder as SeismicFlow.py

**Issue**: Slow performance on neural network operations  
**Solution**: This is normal for CPU-only systems. Consider GPU setup for faster processing.

## Uninstallation

To completely remove SeismicFlow:

1. Simply delete the entire installation folder
2. No registry cleaning or system changes needed
3. No traces left on your system

## Verification

After installation, when you run `SeismicFlow.bat`, you should see:
1. The SeismicFlow splash screen
2. The main GUI interface with all menus and tools available
3. No error messages in the terminal window

## Technical Details

- **Framework**: Qt-based GUI (cross-platform compatible)
- **ML Libraries**: PyTorch 2.0.0, TensorFlow 2.10.1
- **CUDA Support**: Version 11.7 (optional)
- **Python Version**: 3.10.11 (automatically installed)
- **Installation Type**: Fully portable and self-contained
- **Dependencies**: Automatically installed via `install.py`

## Support

For tutorials, updates, and support:
- **YouTube**: [https://www.youtube.com/@Seismicflow-inc](https://www.youtube.com/@Seismicflow-inc)

If you encounter any issues:
1. Ensure you downloaded ALL required files
2. Verify all files are in the same directory
3. Check that install.py completed without errors
4. Ensure you have sufficient disk space and internet connection
5. Try running as Administrator if permission errors occur

## Why This Installation Method?

Traditional Python installations can be complex and create conflicts between different projects. SeismicFlow uses a portable installation approach that:

- **Just Works**: No dependency conflicts or version mismatches
- **Clean**: Completely isolated from your system
- **Simple**: One command to install, one click to run
- **Safe**: Delete the folder to completely remove everything
- **Professional**: Production-grade environment setup

---

*SeismicFlow: Professional geophysical analysis made accessible*
