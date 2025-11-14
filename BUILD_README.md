# nvdiffrast Binary Wheel Build

This fork adds support for building **true binary wheels** for Windows with pre-compiled CUDA extensions.

## What's Different?

| Aspect | Original nvdiffrast | This Fork |
|--------|-------------------|-----------|
| Wheel Type | Source wheel (`py3-none-any`) | Binary wheel (`cp311-cp311-win_amd64`) |
| Installation Requires MSVC? | ✅ YES (~7GB) | ❌ NO |
| Compilation | JIT at runtime (1-2 min first use) | Pre-compiled (instant load) |
| Setup | Complex | Simple `pip install` |

## Changes Made

### 1. Modified `setup.py`
- Added environment variable `BUILD_BINARY_WHEEL` to trigger binary build mode
- Uses `torch.utils.cpp_extension.CUDAExtension` to compile CUDA code during wheel build
- Falls back to original source wheel behavior if `BUILD_BINARY_WHEEL != 1`

### 2. Modified `nvdiffrast/torch/ops.py`
- Added pre-compiled extension loader at the beginning of `_get_plugin()`
- Tries to import `nvdiffrast_plugin` (the pre-compiled extension) first
- Falls back to JIT compilation if pre-compiled version not found
- Backwards compatible with source wheels

### 3. Added GitHub Actions Workflow
- `.github/workflows/build_binary_wheel.yml`
- Automatically builds binary wheels on push
- Installs CUDA 11.8 toolkit in CI
- Tests the built wheel
- Uploads as artifact

## Building Binary Wheels

### Using GitHub Actions (Recommended)

1. Push changes to your fork
2. GitHub Actions automatically builds the wheel
3. Download from Actions > Artifacts

### Manual Build (Requires MSVC + CUDA 11.8)

```powershell
# Install dependencies
pip install wheel setuptools ninja
pip install torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4

# Build binary wheel
$env:BUILD_BINARY_WHEEL = "1"
$env:TORCH_CUDA_ARCH_LIST = "6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0"
python setup.py bdist_wheel

# Wheel will be in dist/ folder
```

## Installing Binary Wheel

```bash
# Install PyTorch first (required dependency)
pip install torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu118

# Install the binary wheel
pip install nvdiffrast-0.3.4-cp311-cp311-win_amd64.whl

# Test it
python -c "import nvdiffrast.torch as dr; print('Success!')"
```

## Requirements

### Build Requirements (CI only)
- Python 3.11
- PyTorch 2.2.2 (CUDA 11.8)
- CUDA Toolkit 11.8
- MSVC compiler
- NumPy 1.26.4

### Runtime Requirements (End Users)
- Python 3.11
- PyTorch 2.2.2 (CUDA 11.8)
- NVIDIA GPU with CUDA support
- CUDA 11.8 runtime (included in GPU drivers)
- **NO MSVC Build Tools required** ✅

## Supported GPU Architectures

The wheel includes pre-compiled code for these compute capabilities:
- 6.0, 6.1 (GTX 10-series)
- 7.0, 7.5 (RTX 20-series, Tesla V100/T4)
- 8.0 (A100)
- 8.6 (RTX 30-series)
- 8.9 (RTX 40-series)
- 9.0 (Future GPUs)

## Backwards Compatibility

The modified `setup.py` is **fully backwards compatible**:
- Without `BUILD_BINARY_WHEEL=1`: Builds original source wheel
- With `BUILD_BINARY_WHEEL=1`: Builds binary wheel with pre-compiled extensions

The modified `ops.py` is also backwards compatible:
- If pre-compiled extension exists: Uses it
- If not: Falls back to JIT compilation (original behavior)

## Troubleshooting

### "Could not locate MSVC installation" during build
- Use GitHub Actions (has MSVC pre-installed)
- Or install Visual Studio Build Tools locally

### "CUDA error: no kernel image is available"
- Your GPU architecture wasn't included in `TORCH_CUDA_ARCH_LIST`
- Rebuild with your GPU's compute capability

### Import error at runtime
- Ensure PyTorch 2.2.2 (CUDA 11.8) is installed
- Verify CUDA runtime is available (check GPU drivers)

## Credits

- Original nvdiffrast: https://github.com/NVlabs/nvdiffrast
- Binary wheel modifications: Aero-Ex

## License

Same as original nvdiffrast (NVIDIA Source Code License)
