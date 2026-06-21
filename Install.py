#!/usr/bin/env python3
"""
Portable installation script for SeismicFlow
Creates a completely self-contained environment

Windows: downloads Python 3.10 embeddable, creates venv
Linux:   uses system Python 3, creates venv

Run: python install.py  (or  python3 install.py  on Linux)
"""

import subprocess
import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

PLATFORM = sys.platform  # "win32" or "linux"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def download_file(url, destination):
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
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
    """Windows: captures output (original behavior). Linux: streams output live."""
    print(f"Running: {command}")
    if PLATFORM == "win32":
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        if result.stdout:
            print(result.stdout)
        return True
    else:
        # Stream output live so user can see progress and it doesn't appear stuck
        result = subprocess.run(command, shell=True, cwd=cwd)
        return result.returncode == 0


def run_command_with_timeout(command, timeout_seconds, cwd=None):
    """
    Linux-only: run a command with a timeout.
    Streams output live. Returns (success, timed_out).
    """
    import threading
    print(f"Running: {command}")
    proc = subprocess.Popen(command, shell=True, cwd=cwd)

    timed_out = [False]

    def kill_after():
        timed_out[0] = True
        proc.kill()

    timer = threading.Timer(timeout_seconds, kill_after)
    timer.start()
    try:
        proc.wait()
    finally:
        timer.cancel()

    if timed_out[0]:
        return False, True
    return proc.returncode == 0, False


def find_system_python3():
    """
    Find the best available Python 3 on the system (Linux).

    SeismicFlow requires Python 3.8-3.10.
      - PyTorch 2.0.0 and TensorFlow 2.10.1 have prebuilt wheels for
        cp38, cp39, cp310 on Linux x86_64.
      - Python 3.11+ is NOT supported by these pinned versions and will
        cause confusing dependency failures later, so we warn about it.
      - Python 3.7 and below are rejected outright.

    Search order: prefer 3.10 (best-tested), then 3.9, 3.8, then whatever
    the system 'python3' alias resolves to.
    """
    REQUIRED_MIN = (3, 8)
    REQUIRED_MAX = (3, 10)   # inclusive -- 3.11+ not tested with pinned deps

    # Check a specific candidate; returns (path, version_tuple) or None
    def probe(candidate):
        path = shutil.which(candidate)
        if not path:
            return None
        result = subprocess.run(
            [path, "-c", "import sys; v=sys.version_info; print(v.major,v.minor)"],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        try:
            major, minor = map(int, result.stdout.strip().split())
            return path, (major, minor)
        except ValueError:
            return None

    # Preferred candidates in priority order
    candidates = ["python3.10", "python3.9", "python3.8", "python3.11",
                  "python3.12", "python3", "python"]

    found_too_new = None  # track if we saw a 3.11+ but nothing better

    for candidate in candidates:
        hit = probe(candidate)
        if not hit:
            continue
        path, ver = hit

        if ver < REQUIRED_MIN:
            print(f"  x {path} is Python {ver[0]}.{ver[1]} -- too old (need 3.8-3.10), skipping")
            continue

        if ver > REQUIRED_MAX:
            if found_too_new is None:
                found_too_new = (path, ver)
            continue  # keep looking for a better fit

        # Exactly in the supported range
        print(f"\u2713 Found system Python {ver[0]}.{ver[1]}: {path}")
        return path

    # Only found Python 3.11+
    if found_too_new:
        path, ver = found_too_new
        print(
            f"\u26a0 Only found Python {ver[0]}.{ver[1]} ({path}).\n"
            f"  SeismicFlow's pinned packages (PyTorch 2.0.0, TensorFlow 2.10.1)\n"
            f"  do not have prebuilt wheels for Python 3.11+. Installation will\n"
            f"  likely fail or produce a broken environment.\n"
            f"  Strongly recommended: install Python 3.10 first:\n"
            f"    Debian/Ubuntu : sudo apt install python3.10\n"
            f"    Fedora/RHEL   : sudo dnf install python3.10\n"
            f"  Attempting with Python {ver[0]}.{ver[1]} anyway..."
        )
        return path

    return None


def check_url_reachable(url, timeout=10):
    """Quick check whether a URL is reachable at all."""
    import urllib.request
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────
# Windows-only: embedded Python setup (untouched)
# ──────────────────────────────────────────────────────────────

def setup_windows_python(base_dir):
    """Download and configure embedded Python 3.10 on Windows."""
    python_dir = base_dir / "python310"
    python_zip = base_dir / "python310.zip"

    if python_dir.exists():
        print("Checking existing Python installation...")
        pip_exe = python_dir / "Scripts" / "pip.exe"
        if not pip_exe.exists():
            print("Previous installation corrupted, cleaning up...")
            shutil.rmtree(python_dir)

    if not python_dir.exists():
        python_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
        if not python_zip.exists():
            if not download_file(python_url, python_zip):
                return None
        if not extract_zip(python_zip, python_dir):
            return None
        python_zip.unlink()
        print("✓ Python 3.10 extracted")
    else:
        print("✓ Python 3.10 already installed")

    python_exe = python_dir / "python.exe"

    pth_file = python_dir / "python310._pth"
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
            print("✓ Embedded Python configured")

    get_pip = base_dir / "get-pip.py"
    print("\nInstalling pip...")
    if not get_pip.exists():
        if not download_file("https://bootstrap.pypa.io/get-pip.py", get_pip):
            return None
    if not run_command(f'"{python_exe}" "{get_pip}"'):
        return None
    if get_pip.exists():
        get_pip.unlink()

    result = subprocess.run(f'"{python_exe}" -m pip --version',
                            shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ pip verification failed: {result.stderr}")
        return None
    print(f"✓ {result.stdout.strip()}")

    return python_exe


# ──────────────────────────────────────────────────────────────
# Shared: venv creation
# ──────────────────────────────────────────────────────────────

def create_venv(base_python, base_dir):
    venv_dir = base_dir / "venv"

    if venv_dir.exists():
        print("✓ Virtual environment already exists")
    else:
        print("Installing virtualenv...")
        if not run_command(f'"{base_python}" -m pip install virtualenv'):
            print("✗ Failed to install virtualenv")
            return None
        print("Creating virtual environment...")
        if not run_command(f'"{base_python}" -m virtualenv "{venv_dir}"'):
            print("✗ Failed to create virtual environment")
            return None
        print("✓ Virtual environment created")

    if PLATFORM == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip    = venv_dir / "Scripts" / "pip.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        venv_pip    = venv_dir / "bin" / "pip"

    return venv_python, venv_pip


# ──────────────────────────────────────────────────────────────
# Linux-only: robust PyTorch install with fallback
# ──────────────────────────────────────────────────────────────

def install_pytorch_linux(venv_pip):
    """
    Try GPU (cu117) first with a timeout and URL reachability check.
    Fall back to CPU-only if:
      - the PyTorch index URL is unreachable
      - download times out
      - install fails for any reason
    Versions stay the same (2.0.0 / 0.15.1) regardless of GPU/CPU.
    """
    TORCH_VERSION      = "torch==2.0.0"
    TORCHVISION_VERSION = "torchvision==0.15.1"
    GPU_INDEX_URL      = "https://download.pytorch.org/whl/cu117"
    TIMEOUT_SECONDS    = 600  # 10 minutes — big download, be generous

    # ── Attempt 1: GPU wheel ──────────────────────────────────
    print("Checking PyTorch GPU index URL reachability...")
    if check_url_reachable(GPU_INDEX_URL, timeout=15):
        print("✓ GPU index URL reachable, attempting GPU install...")
        print("  (downloading ~2 GB — this will take a while, progress shown below)\n")
        cmd = (
            f'"{venv_pip}" install {TORCH_VERSION} {TORCHVISION_VERSION} '
            f'--index-url {GPU_INDEX_URL}'
        )
        success, timed_out = run_command_with_timeout(cmd, TIMEOUT_SECONDS)
        if success:
            print("✓ PyTorch (GPU/CUDA) installed successfully")
            return True
        if timed_out:
            print(f"⚠ GPU install timed out after {TIMEOUT_SECONDS//60} minutes")
        else:
            print("⚠ GPU install failed (no matching wheel or network error)")
    else:
        print("⚠ GPU index URL not reachable (geo-block or network issue)")

    # ── Attempt 2: CPU wheel ──────────────────────────────────
    print("\nFalling back to CPU-only PyTorch (same versions, no CUDA)...")
    print("  GPU features will not be available, but the app will run.\n")
    cmd = (
        f'"{venv_pip}" install {TORCH_VERSION} {TORCHVISION_VERSION} '
        f'--index-url https://download.pytorch.org/whl/cpu'
    )
    success, timed_out = run_command_with_timeout(cmd, TIMEOUT_SECONDS)
    if success:
        print("✓ PyTorch (CPU) installed successfully")
        return True
    if timed_out:
        print(f"⚠ CPU install also timed out after {TIMEOUT_SECONDS//60} minutes")
    else:
        print("⚠ CPU install also failed")

    # ── Attempt 3: plain PyPI (no index URL) ─────────────────
    print("\nFalling back to plain PyPI install...")
    cmd = f'"{venv_pip}" install {TORCH_VERSION} {TORCHVISION_VERSION}'
    success, timed_out = run_command_with_timeout(cmd, TIMEOUT_SECONDS)
    if success:
        print("✓ PyTorch installed from PyPI")
        return True

    print("✗ All PyTorch install attempts failed")
    return False


# ──────────────────────────────────────────────────────────────
# Linux-only: robust TensorFlow install with fallback
# ──────────────────────────────────────────────────────────────

def install_tensorflow_linux(venv_pip):
    """
    Try tensorflow==2.10.1 first.
    If it fails (no wheel for this Python/OS combo), try tensorflow-cpu==2.10.1.
    """
    TIMEOUT_SECONDS = 600

    print("Installing TensorFlow 2.10.1...")
    cmd = f'"{venv_pip}" install tensorflow==2.10.1'
    success, timed_out = run_command_with_timeout(cmd, TIMEOUT_SECONDS)
    if success:
        print("✓ TensorFlow installed")
        return True
    if timed_out:
        print("⚠ TensorFlow install timed out")
    else:
        print("⚠ tensorflow==2.10.1 failed (may have no wheel for this platform)")

    print("\nFalling back to tensorflow-cpu==2.10.1...")
    cmd = f'"{venv_pip}" install tensorflow-cpu==2.10.1'
    success, timed_out = run_command_with_timeout(cmd, TIMEOUT_SECONDS)
    if success:
        print("✓ TensorFlow (CPU) installed")
        return True

    print("✗ All TensorFlow install attempts failed")
    return False


# ──────────────────────────────────────────────────────────────
# Linux-only: robust PyQt6 install with version fallback ladder
# ──────────────────────────────────────────────────────────────

def install_pyqt6_linux(venv_pip):
    """
    PyQt6 has no pure-Python wheel — if pip can't find a prebuilt wheel for
    the exact pinned version on this platform, it silently falls back to
    building from source via SIP, which requires 'qmake' (Qt dev tools) to
    be installed on the system. If qmake is missing, that build crashes
    with PyProjectOptionException, killing the whole install.

    Strategy:
      1. Try the preferred pinned version first, *wheel-only*
         (--only-binary=:all:). This is the normal/expected path and
         matches what would happen with a plain `pip install` when a
         wheel exists -- it just refuses to silently slip into a source
         build if it doesn't.
      2. If that version has no wheel for this platform, walk backward
         through a ladder of known-good recent PyQt6 versions, still
         wheel-only, stopping at the first one that installs cleanly.
      3. Only if EVERY version in the ladder has no wheel at all (very
         unlikely on standard x86_64/aarch64 Linux, but possible on an
         unusual architecture) fall back to allowing a source build --
         and only then, with an explicit warning about needing qmake,
         since that is the one path that can hit the original failure.

    Returns the installed version string (e.g. "6.9.1") on success, or
    None if every attempt failed. The caller needs the actual version --
    not just success/failure -- so install_pyqt6_webengine_linux() can
    pick a WebEngine release from a compatible minor-version family
    rather than blindly using whatever was pinned in requirements.txt.
    """
    TIMEOUT_SECONDS = 900  # PyQt6 wheels are large; be generous

    PREFERRED_VERSION = "6.10.1"
    FALLBACK_LADDER = ["6.9.1", "6.9.0", "6.8.1", "6.8.0", "6.7.1", "6.6.1"]

    versions_to_try = [PREFERRED_VERSION] + [
        v for v in FALLBACK_LADDER if v != PREFERRED_VERSION
    ]

    print(f"Attempting PyQt6 install (wheel-only, preferred version {PREFERRED_VERSION})...")

    for i, version in enumerate(versions_to_try):
        label = "preferred" if i == 0 else f"fallback #{i}"
        print(f"\n→ Trying PyQt6=={version} ({label}), wheel-only...")
        cmd = f'"{venv_pip}" install "PyQt6=={version}" --only-binary=:all:'
        success, timed_out = run_command_with_timeout(cmd, TIMEOUT_SECONDS)
        if success:
            print(f"✓ PyQt6=={version} installed successfully (prebuilt wheel)")
            if version != PREFERRED_VERSION:
                print(
                    f"  Note: preferred version {PREFERRED_VERSION} had no wheel for this "
                    f"platform, used {version} instead. If your code depends on a feature "
                    f"only in {PREFERRED_VERSION}+, double-check compatibility."
                )
            return version
        if timed_out:
            print(f"⚠ PyQt6=={version} timed out after {TIMEOUT_SECONDS//60} minutes")
        else:
            print(f"⚠ PyQt6=={version} has no prebuilt wheel for this platform, or install failed")

    # ── Last resort: allow a source build, with a loud warning ────────
    print(
        "\n⚠ No PyQt6 version in the ladder has a prebuilt wheel for this platform.\n"
        "  Falling back to a SOURCE BUILD for the preferred version. This requires\n"
        "  Qt6 development tools (qmake) to be installed on the system, or it will\n"
        "  fail with a PyProjectOptionException, the same crash this fallback exists\n"
        "  to avoid wherever possible.\n"
        "  If this next step fails, install Qt6 dev tools first, then re-run:\n"
        "    Debian/Ubuntu : sudo apt-get install qt6-base-dev\n"
        "    Fedora/RHEL   : sudo dnf install qt6-qtbase-devel\n"
        "    Arch          : sudo pacman -S qt6-base\n"
    )
    cmd = f'"{venv_pip}" install "PyQt6=={PREFERRED_VERSION}"'
    success, timed_out = run_command_with_timeout(cmd, TIMEOUT_SECONDS)
    if success:
        print(f"✓ PyQt6=={PREFERRED_VERSION} installed successfully (built from source)")
        return PREFERRED_VERSION
    if timed_out:
        print(f"⚠ Source build timed out after {TIMEOUT_SECONDS//60} minutes")
    else:
        print("✗ Source build failed -- qmake/Qt6 dev tools are likely missing (see commands above)")

    print("✗ All PyQt6 install attempts failed")
    return None


# ──────────────────────────────────────────────────────────────
# Linux-only: robust PyQt6-WebEngine install, version-matched to
# whichever PyQt6 actually got installed above
# ──────────────────────────────────────────────────────────────

def install_pyqt6_webengine_linux(venv_pip, pyqt6_version):
    """
    PyQt6-WebEngine is a SEPARATE PyPI package from PyQt6 itself, and
    requirements.txt pins it independently (PyQt6-WebEngine==6.10.0).
    It needs its own install step -- it does NOT get installed as a
    side effect of installing PyQt6.

    It also needs to be from a compatible release family with whatever
    PyQt6 version actually landed above. If the PyQt6 ladder fell back
    from 6.10.1 to, say, 6.7.1, blindly installing WebEngine 6.10.0
    anyway risks a version mismatch at import time. Real-world Linux
    distros commonly pair PyQt6 and PyQt6-WebEngine from neighboring
    minor versions (e.g. PyQt6 6.8.1 with WebEngine 6.8.0 is a known
    working combination), so this matches WebEngine's minor version to
    PyQt6's minor version where possible, instead of using a fixed pin
    regardless of what PyQt6 ended up being.

    pyqt6_version: the actual installed PyQt6 version string returned
    by install_pyqt6_linux(), e.g. "6.9.1". Required -- this function
    should only be called after PyQt6 itself installed successfully.

    Returns True/False.
    """
    if not pyqt6_version:
        print("⚠ Skipping PyQt6-WebEngine: PyQt6 itself did not install successfully")
        return False

    TIMEOUT_SECONDS = 900

    # requirements.txt pin, tried first regardless of PyQt6's landed
    # version -- this is the normal/expected case when PyQt6 itself
    # also installed at or near its preferred version.
    PINNED_VERSION = "6.10.0"

    # Minor-version family matched to PyQt6's actual minor version,
    # e.g. PyQt6==6.8.1 -> try WebEngine 6.8.x. Mirrors the same
    # major.minor as PyQt6 first, then steps back one minor at a time.
    pyqt6_major_minor = ".".join(pyqt6_version.split(".")[:2])  # "6.9.1" -> "6.9"
    KNOWN_WEBENGINE_VERSIONS = [
        "6.10.0", "6.9.1", "6.9.0", "6.8.1", "6.8.0", "6.7.1", "6.7.0", "6.6.0",
    ]
    matched_family = [
        v for v in KNOWN_WEBENGINE_VERSIONS if v.startswith(pyqt6_major_minor)
    ]

    versions_to_try = [PINNED_VERSION] + [
        v for v in matched_family + KNOWN_WEBENGINE_VERSIONS
        if v != PINNED_VERSION
    ]
    # de-duplicate while preserving order
    seen = set()
    versions_to_try = [v for v in versions_to_try if not (v in seen or seen.add(v))]

    print(
        f"Attempting PyQt6-WebEngine install (wheel-only, matching installed "
        f"PyQt6=={pyqt6_version})..."
    )

    for i, version in enumerate(versions_to_try):
        label = "pinned" if i == 0 else f"fallback #{i}"
        print(f"\n→ Trying PyQt6-WebEngine=={version} ({label}), wheel-only...")
        cmd = f'"{venv_pip}" install "PyQt6-WebEngine=={version}" --only-binary=:all:'
        success, timed_out = run_command_with_timeout(cmd, TIMEOUT_SECONDS)
        if success:
            print(f"✓ PyQt6-WebEngine=={version} installed successfully (prebuilt wheel)")
            if version != PINNED_VERSION:
                print(
                    f"  Note: pinned version {PINNED_VERSION} had no wheel for this platform "
                    f"or was incompatible, used {version} instead to match installed "
                    f"PyQt6=={pyqt6_version}."
                )
            return True
        if timed_out:
            print(f"⚠ PyQt6-WebEngine=={version} timed out after {TIMEOUT_SECONDS//60} minutes")
        else:
            print(f"⚠ PyQt6-WebEngine=={version} has no prebuilt wheel for this platform, or install failed")

    print(
        "\n⚠ No PyQt6-WebEngine version tried has a prebuilt wheel for this platform.\n"
        "  Skipping WebEngine -- the app will run, but any feature using "
        "QtWebEngineWidgets / QtWebEngineCore will be unavailable.\n"
        "  PyQt6-WebEngine also requires qmake/Qt6 dev tools for a source build, "
        "same as PyQt6 itself (see install_pyqt6_linux warning above)."
    )
    return False


def install_requirements_excluding_pyqt6_linux(venv_pip, requirements_file):
    """
    Linux-only: install requirements.txt normally, but skip:
      1. PyQt6 and PyQt6-WebEngine  -- already handled safely above with
         wheel-only fallback ladder.
      2. Windows-only packages      -- they have no Linux distribution at
         all; letting pip try them causes a hard ERROR that aborts the
         entire install.  We strip them here instead of relying on pip to
         ignore them gracefully, because pip does not ignore them gracefully.

    All exclusions are explicit by exact package name (lowercased) so the
    intent is clear and unrelated packages whose names happen to share a
    prefix are never accidentally removed.

    Writes a temp copy with excluded line(s) stripped, installs from that,
    then cleans up.
    """
    # Handled by their own dedicated install functions above
    PYQT_PACKAGES = ("pyqt6", "pyqt6-webengine")

    # Packages that exist only on Windows -- no Linux wheel, no source dist,
    # pip errors out hard if you try to install them on Linux.
    WINDOWS_ONLY_PACKAGES = (
        "pywin32",
        "pywin32-ctypes",
        "pywin32-setuptools",
        "pywinpty",
        "comtypes",
        "winshell",
        "wmi",
        "pywinauto",
        "win32api",
        "win32con",
        "win32com",
    )

    EXCLUDED_PACKAGES = PYQT_PACKAGES + WINDOWS_ONLY_PACKAGES

    def is_excluded(line):
        stripped = line.strip().lower()
        # skip blank lines and comments early
        if not stripped or stripped.startswith("#"):
            return False
        # match "pkgname==..." / "pkgname>=..." / "pkgname" but not
        # unrelated packages that happen to start with the same letters
        for pkg in EXCLUDED_PACKAGES:
            if (stripped == pkg
                    or stripped.startswith(pkg + "=")
                    or stripped.startswith(pkg + ">")
                    or stripped.startswith(pkg + "<")
                    or stripped.startswith(pkg + "!")
                    or stripped.startswith(pkg + " ")
                    or stripped.startswith(pkg + "[")):
                return True
        return False

    lines = requirements_file.read_text().splitlines()
    excluded_lines = [line for line in lines if is_excluded(line)]
    filtered = [line for line in lines if not is_excluded(line)]

    if excluded_lines:
        pyqt_excluded = [
            l for l in excluded_lines
            if any(l.strip().lower().startswith(p) for p in PYQT_PACKAGES)
        ]
        win_excluded = [
            l for l in excluded_lines
            if any(l.strip().lower().startswith(p) for p in WINDOWS_ONLY_PACKAGES)
        ]
        if pyqt_excluded:
            print(
                f"(Excluding {len(pyqt_excluded)} PyQt6/PyQt6-WebEngine line(s) from "
                f"requirements.txt -- already installed above)"
            )
        if win_excluded:
            print(
                f"(Excluding {len(win_excluded)} Windows-only package(s) from "
                f"requirements.txt -- not available on Linux: "
                + ", ".join(l.strip() for l in win_excluded) + ")"
            )

    temp_req = requirements_file.parent / "_requirements_no_pyqt6.tmp.txt"
    temp_req.write_text("\n".join(filtered) + "\n")

    try:
        success = run_command(f'"{venv_pip}" install -r "{temp_req}"')
    finally:
        if temp_req.exists():
            temp_req.unlink()

    return success


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SeismicFlow Installation")
    print("=" * 60)
    print()

    base_dir = Path.cwd()
    print(f"Installation directory: {base_dir}")
    print(f"Platform: {'Windows' if PLATFORM == 'win32' else 'Linux'}\n")

    # ── Step 1: Get a base Python ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Setting up Python")
    print("=" * 60)

    if PLATFORM == "win32":
        base_python = setup_windows_python(base_dir)
        if base_python is None:
            print("✗ Failed to set up Python")
            return False
    else:
        base_python = find_system_python3()
        if base_python is None:
            print("✗ Could not find Python 3.8+ on this system.")
            print("  Install it with:  sudo apt install python3  (Debian/Ubuntu)")
            print("                or: sudo dnf install python3  (CentOS/RHEL)")
            return False

    # ── Step 2: Virtual environment ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Creating virtual environment")
    print("=" * 60)

    result = create_venv(base_python, base_dir)
    if result is None:
        return False
    venv_python, venv_pip = result

    # ── Step 3: Upgrade pip ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Updating package manager")
    print("=" * 60)

    if not run_command(f'"{venv_python}" -m pip install --upgrade pip'):
        print("Warning: Could not upgrade pip, continuing...")

    # ── Step 4: PyTorch ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Installing PyTorch")
    print("=" * 60)

    if PLATFORM == "win32":
        print("This may take several minutes...\n")
        pytorch_cmd = (
            f'"{venv_pip}" install torch==2.0.0 torchvision==0.15.1 '
            f'--index-url https://download.pytorch.org/whl/cu117'
        )
        if not run_command(pytorch_cmd):
            print("✗ Failed to install PyTorch")
            return False
        print("✓ PyTorch installed")
    else:
        if not install_pytorch_linux(venv_pip):
            print("✗ Failed to install PyTorch")
            return False

    # ── Step 5: TensorFlow ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Installing TensorFlow")
    print("=" * 60)

    if PLATFORM == "win32":
        if not run_command(f'"{venv_pip}" install tensorflow==2.10.1'):
            print("✗ Failed to install TensorFlow")
            return False
        print("✓ TensorFlow installed")
    else:
        if not install_tensorflow_linux(venv_pip):
            print("✗ Failed to install TensorFlow")
            return False

    # ── Step 6: requirements.txt ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Installing additional dependencies")
    print("=" * 60)

    requirements_file = base_dir / "requirements.txt"

    if PLATFORM == "win32":
        if requirements_file.exists():
            if not run_command(f'"{venv_pip}" install -r "{requirements_file}"'):
                print("✗ Failed to install requirements")
                return False
            print("✓ Dependencies installed")
        else:
            print("⚠ requirements.txt not found, skipping")
    else:
        # Linux: install PyQt6 separately first, with its own wheel-only
        # fallback ladder (see install_pyqt6_linux). This is what prevents
        # the qmake/source-build crash. The function returns the actual
        # installed version string (e.g. "6.9.1") so that WebEngine can
        # be matched to the same minor-version family.
        pyqt6_version = install_pyqt6_linux(venv_pip)
        if not pyqt6_version:
            print("✗ Failed to install PyQt6")
            return False

        # Install PyQt6-WebEngine, version-matched to whichever PyQt6
        # actually landed above. This is a separate package from PyQt6
        # itself and is NOT installed as a side-effect of PyQt6.
        # SeismicFlow imports QtWebEngineWidgets at startup so this is
        # required -- skipping it causes an immediate ModuleNotFoundError.
        if not install_pyqt6_webengine_linux(venv_pip, pyqt6_version):
            print("⚠ PyQt6-WebEngine could not be installed.")
            print("  The app will crash at startup (QtWebEngineWidgets missing).")
            print("  Try manually: pip install PyQt6-WebEngine --only-binary=:all:")
            # Don't hard-fail the whole install — everything else is fine.

        if requirements_file.exists():
            if not install_requirements_excluding_pyqt6_linux(venv_pip, requirements_file):
                print("✗ Failed to install requirements")
                return False
            print("✓ Dependencies installed")
        else:
            print("⚠ requirements.txt not found, skipping")

    # ── Step 7: Launcher ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Creating launcher")
    print("=" * 60)

    if PLATFORM == "win32":
        launcher = base_dir / "SeismicFlow.bat"
        launcher.write_text(f'@echo off\n"{venv_python}" SeismicFlow.py\npause\n')
        print("✓ Launcher created: SeismicFlow.bat")
        run_instruction = "double-click SeismicFlow.bat"
    else:
        launcher = base_dir / "SeismicFlow.sh"
        launcher.write_text(
            f'#!/bin/bash\n'
            f'cd "$(dirname "$0")"\n'
            f'"{venv_python}" SeismicFlow.py\n'
        )
        launcher.chmod(0o755)
        print("✓ Launcher created: SeismicFlow.sh")
        run_instruction = "run ./SeismicFlow.sh"

    # ── Done ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Installation completed successfully")
    print("=" * 60)
    print(f"\nInstallation directory: {base_dir}")
    print(f"\nTo run SeismicFlow: {run_instruction}")
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
