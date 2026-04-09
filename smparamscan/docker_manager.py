"""Manage HDECAY execution via Docker or local fallback."""

import hashlib
import os
import shutil
import subprocess
import tempfile

from .config import DOCKER_IMAGE_NAME

HDECAY_SOURCE_URL = "https://www.fuw.edu.pl/~kalino/fortran/hdecay.f"

# Patching old-style Fortran declarations for modern gfortran
_FORTRAN_PATCHES = [
    ("COMPLEX FUNCTION F0*16", "COMPLEX*16 FUNCTION F0"),
    ("COMPLEX FUNCTION LI2*16", "COMPLEX*16 FUNCTION LI2"),
    ("COMPLEX FUNCTION CLI2*16", "COMPLEX*16 FUNCTION CLI2"),
]


def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hash_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Docker mode
# ---------------------------------------------------------------------------

def _get_build_hash() -> str:
    root = _get_project_root()
    return _hash_file(os.path.join(root, "Dockerfile"))


def _get_stored_hash() -> str | None:
    root = _get_project_root()
    hashfile = os.path.join(root, ".docker_build_hash")
    if os.path.exists(hashfile):
        return open(hashfile).read().strip()
    return None


def _store_hash(h: str):
    root = _get_project_root()
    with open(os.path.join(root, ".docker_build_hash"), "w") as f:
        f.write(h)


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _image_exists() -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE_NAME],
        capture_output=True,
    )
    return result.returncode == 0


def _build_docker_image(force: bool = False):
    current_hash = _get_build_hash()
    stored_hash = _get_stored_hash()

    if not force and stored_hash == current_hash and _image_exists():
        return

    print("Building HDECAY Docker image...")
    root = _get_project_root()
    result = subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE_NAME, root],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed:\n{result.stderr}")

    _store_hash(current_hash)
    print("Docker image built successfully.")


def _run_hdecay_docker(input_content: str) -> str:
    work_dir = tempfile.mkdtemp(prefix="hdecay_")
    with open(os.path.join(work_dir, "hdecay.in"), "w") as f:
        f.write(input_content)

    result = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{work_dir}:/app/work",
         DOCKER_IMAGE_NAME],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"HDECAY run failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return work_dir


# ---------------------------------------------------------------------------
# Local fallback mode
# ---------------------------------------------------------------------------

def _local_hdecay_binary() -> str:
    root = _get_project_root()
    return os.path.join(root, ".hdecay_local", "hdecay")


def _ensure_local_hdecay():
    """Download and compile HDECAY locally if needed."""
    binary = _local_hdecay_binary()
    if os.path.exists(binary):
        return

    local_dir = os.path.dirname(binary)
    os.makedirs(local_dir, exist_ok=True)
    source = os.path.join(local_dir, "hdecay.f")

    print("Downloading HDECAY source...")
    result = subprocess.run(
        ["wget", "-q", "-O", source, HDECAY_SOURCE_URL],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download HDECAY: {result.stderr}")

    # Patch old-style Fortran declarations
    with open(source) as f:
        code = f.read()
    for old, new in _FORTRAN_PATCHES:
        code = code.replace(old, new)
    with open(source, "w") as f:
        f.write(code)

    print("Compiling HDECAY...")
    result = subprocess.run(
        ["gfortran", "-O2", "-std=legacy", "-o", binary, source],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"HDECAY compilation failed:\n{result.stderr}")
    print("HDECAY compiled successfully (local mode).")


def _run_hdecay_local(input_content: str) -> str:
    binary = _local_hdecay_binary()
    work_dir = tempfile.mkdtemp(prefix="hdecay_")
    with open(os.path.join(work_dir, "hdecay.in"), "w") as f:
        f.write(input_content)

    result = subprocess.run(
        [binary], capture_output=True, text=True, cwd=work_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"HDECAY run failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return work_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_use_docker: bool | None = None


def build_image(force: bool = False):
    """Prepare HDECAY — Docker image or local compilation."""
    global _use_docker
    if _use_docker is None:
        _use_docker = _docker_available()

    if _use_docker:
        try:
            _build_docker_image(force=force)
            return
        except RuntimeError:
            print("Docker build failed (registry unreachable?). "
                  "Falling back to local compilation.")
            _use_docker = False

    _ensure_local_hdecay()


def run_hdecay(input_content: str) -> str:
    """Run HDECAY and return path to output directory."""
    if _use_docker is None:
        build_image()

    if _use_docker:
        return _run_hdecay_docker(input_content)
    else:
        return _run_hdecay_local(input_content)
