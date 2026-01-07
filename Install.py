#!/usr/bin/env python3
"""
Portable installation script for SeismicFlow
Creates a completely self-contained environment with Python 3.10

Run: python install.py
"""

import subprocess
import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_file(url, destination):
    """Download a file with progress indication"""
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def run_command(command, cwd=None):
    """Run a command and check for errors"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True


def main():
    print("=" * 60)
    print("SeismicFlow Installation")
    print("=" * 60)
    print()

    # Setup directories
    base_dir = Path.cwd()
    python_dir = base_dir / "python310"
    python_zip = base_dir / "python310.zip"
    venv_dir = base_dir / "venv"

    print(f"Installation directory: {base_dir}\n")

    # Clean up any corrupted previous installation
    if python_dir.exists():
        print("Checking existing Python installation...")
        pip_exe = python_dir / "Scripts" / "pip.exe"
        if not pip_exe.exists():
            print("Previous installation corrupted, cleaning up...")
            shutil.rmtree(python_dir)
            print("✓ Cleanup complete")

    # Step 1: Download Python 3.10 embeddable
    print("\n" + "=" * 60)
    print("STEP 1: Installing Python 3.10")
    print("=" * 60)

    python_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"

    if not python_dir.exists():
        if not python_zip.exists():
            if not download_file(python_url, python_zip):
                print("\n✗ Failed to download Python")
                return False

        if not extract_zip(python_zip, python_dir):
            print("\n✗ Failed to extract Python")
            return False

        python_zip.unlink()
        print("✓ Cleanup complete")
    else:
        print("✓ Python 3.10 already installed")

    # Step 2: Setup pip
    print("\n" + "=" * 60)
    print("STEP 2: Configuring package manager")
    print("=" * 60)

    python_exe = python_dir / "python.exe"

    # Configure python310._pth to enable site-packages
    pth_file = python_dir / "python310._pth"
    print(f"Configuring {pth_file.name}...")
    if pth_file.exists():
        content = pth_file.read_text()
        modified = False

        if "#import site" in content:
            content = content.replace("#import site", "import site")
            modified = True
        elif "import site" not in content:
            content += "\nimport site\n"
            modified = True

        if "Lib\\site-packages" not in content:
            lines = content.strip().split('\n')
            new_lines = []
            for line in lines:
                if line.strip() == "import site":
                    new_lines.append("Lib\\site-packages")
                new_lines.append(line)
            content = '\n'.join(new_lines)
            modified = True

        if modified:
            pth_file.write_text(content)
            print("✓ Configuration complete")
        else:
            print("✓ Already configured")
    else:
        print("⚠ Warning: python310._pth not found")

    # Install pip
    get_pip = base_dir / "get-pip.py"

    print("\nInstalling pip...")
    if not get_pip.exists():
        if not download_file("https://bootstrap.pypa.io/get-pip.py", get_pip):
            print("✗ Failed to download get-pip.py")
            return False

    if not run_command(f'"{python_exe}" "{get_pip}"'):
        print("✗ Failed to install pip")
        return False

    if get_pip.exists():
        get_pip.unlink()

    print("✓ Package manager installed")

    # Verify pip
    print("\nVerifying installation...")
    result = subprocess.run(f'"{python_exe}" -m pip --version',
                            shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ Verification failed: {result.stderr}")
        return False
    print(f"✓ {result.stdout.strip()}")

    # Step 3: Create virtual environment
    print("\n" + "=" * 60)
    print("STEP 3: Creating virtual environment")
    print("=" * 60)

    if venv_dir.exists():
        print("✓ Virtual environment already exists")
    else:
        print("Installing virtualenv...")
        if not run_command(f'"{python_exe}" -m pip install virtualenv'):
            print("✗ Failed to install virtualenv")
            return False

        print("Creating virtual environment...")
        if not run_command(f'"{python_exe}" -m virtualenv venv'):
            print("✗ Failed to create virtual environment")
            return False
        print("✓ Virtual environment created")

    # Setup paths for venv
    venv_python = venv_dir / "Scripts" / "python.exe"
    venv_pip = venv_dir / "Scripts" / "pip.exe"

    # Step 4: Upgrade pip
    print("\n" + "=" * 60)
    print("STEP 4: Updating package manager")
    print("=" * 60)

    if not run_command(f'"{venv_python}" -m pip install --upgrade pip'):
        print("Warning: Could not upgrade pip, continuing...")

    # Step 5: Install PyTorch
    print("\n" + "=" * 60)
    print("STEP 5: Installing PyTorch with CUDA support")
    print("=" * 60)
    print("This may take several minutes...\n")

    pytorch_cmd = f'"{venv_pip}" install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117'
    if not run_command(pytorch_cmd):
        print("✗ Failed to install PyTorch")
        return False
    print("✓ PyTorch installed")

    # Step 6: Install TensorFlow
    print("\n" + "=" * 60)
    print("STEP 6: Installing TensorFlow")
    print("=" * 60)

    if not run_command(f'"{venv_pip}" install tensorflow==2.10.1'):
        print("✗ Failed to install TensorFlow")
        return False
    print("✓ TensorFlow installed")

    # Step 7: Install other requirements
    print("\n" + "=" * 60)
    print("STEP 7: Installing additional dependencies")
    print("=" * 60)

    requirements_file = base_dir / "requirements.txt"
    if requirements_file.exists():
        if not run_command(f'"{venv_pip}" install -r requirements.txt'):
            print("✗ Failed to install requirements")
            return False
        print("✓ Dependencies installed")
    else:
        print(f"⚠ Warning: requirements.txt not found, skipping")

    # Step 8: Create launcher script
    print("\n" + "=" * 60)
    print("STEP 8: Creating launcher")
    print("=" * 60)

    run_bat = base_dir / "SeismicFlow.bat"
    run_bat.write_text(f'''@echo off
"{venv_python}" SeismicFlow.py
pause
''')
    print("✓ Launcher created")

    # Final summary
    print("\n" + "=" * 60)
    print("Installation completed successfully")
    print("=" * 60)
    print(f"\nInstallation directory: {base_dir}")
    print("\nTo run SeismicFlow: double-click SeismicFlow.bat")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\nInstallation failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)