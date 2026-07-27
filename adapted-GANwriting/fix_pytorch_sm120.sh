#!/usr/bin/env bash
# ============================================================
# fix_pytorch_sm120.sh  (v2 — handles Python 3.8 constraint)
#
# Problem: PyTorch ≥2.7.0 supports sm_120 (RTX 5080/Blackwell)
#          but requires Python ≥3.10.
#          The existing 'ganwriting' env is Python 3.8.
#
# Solution:
#   1. Create a new conda env 'ganwriting310' with Python 3.10
#   2. Install PyTorch 2.7+ cu128 (sm_120 support)
#   3. Reinstall the project's other dependencies
#
# Usage:
#   bash fix_pytorch_sm120.sh
#   conda activate ganwriting310
#   python main_run.py 0
# ============================================================

set -e
NEW_ENV="ganwriting310"

echo "=== Step 1: Create conda env with Python 3.10 ==="
conda create -y -n "$NEW_ENV" python=3.10

echo ""
echo "=== Step 2: Install PyTorch 2.7+ with CUDA 12.8 (sm_120 support) ==="
conda run -n "$NEW_ENV" pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "=== Step 3: Install project dependencies ==="
conda run -n "$NEW_ENV" pip install \
    opencv-python \
    numpy \
    Pillow \
    scipy \
    scikit-image

echo ""
echo "=== Step 4: Verify GPU access ==="
conda run -n "$NEW_ENV" python - <<'PY'
import torch
print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("Capability      : sm_%d%d" % cap)
    if cap[0] >= 12:
        print("Blackwell check : PASS (sm_120+ supported)")
    else:
        print("Blackwell check : WARNING — unexpected capability")
    x = torch.randn(4, 4, device='cuda')
    print("Smoke test      : PASSED")
else:
    print("ERROR: CUDA not available — check CUDA 12.8 driver install")
    exit(1)
PY

echo ""
echo "================================================================"
echo "  Done. To run training:"
echo "    conda activate $NEW_ENV"
echo "    cd /path/to/research-GANwriting"
echo "    python main_run.py 0"
echo "================================================================"