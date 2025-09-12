#!/usr/bin/env python3
"""
Automatic installation script for SeismicFlow
Run: python install.py
"""

import subprocess
import os


def run_command(command):
    """Run a command and check for errors"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def main():
    print("=== Seismic Project Installation ===")

    # Install PyTorch with CUDA first (special command)
    print("\n1. Installing PyTorch with CUDA support...")
    if not run_command(
            "pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117"):
        print("Failed to install PyTorch")
        return

    # Install TensorFlow
    print("\n2. Installing TensorFlow...")
    if not run_command("pip install tensorflow==2.10.1"):
        print("Failed to install TensorFlow")
        return

    # Now install all other packages from requirements
    print("\n3. Installing other packages...")
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        if not run_command(f"pip install -r {requirements_file}"):
            print("Failed to install other packages")
            return
    else:
        print(f"Warning: {requirements_file} not found, skipping other packages")

    print("\n✅ Installation completed successfully!")
    print("Run: python your_main_script.py")


if __name__ == "__main__":
    main()
