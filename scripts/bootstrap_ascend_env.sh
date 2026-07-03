#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

ASCEND_VENV="${ASCEND_VENV:-/data/venvs/kf-triton-ascend}"
ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
TRITON_ASCEND_VERSION="${TRITON_ASCEND_VERSION:-3.2.0}"
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$ASCEND_SET_ENV" ]]; then
  FOUND_SET_ENV="$(find /usr/local/Ascend -maxdepth 4 -name set_env.sh 2>/dev/null | head -n 1 || true)"
  if [[ -n "$FOUND_SET_ENV" ]]; then
    ASCEND_SET_ENV="$FOUND_SET_ENV"
  fi
fi

if [[ ! -f "$ASCEND_SET_ENV" ]]; then
  echo "Cannot find Ascend set_env.sh. Set ASCEND_SET_ENV=/path/to/set_env.sh and rerun." >&2
  exit 1
fi

echo "Using Ascend environment: $ASCEND_SET_ENV"
echo "Using Python venv:       $ASCEND_VENV"

echo
echo "Initializing benchmark submodule..."
bash scripts/setup_benchmark_submodule.sh

echo
echo "Creating or reusing Python venv..."
mkdir -p "$(dirname "$ASCEND_VENV")"
"$PYTHON_BIN" -m venv --system-site-packages "$ASCEND_VENV"

# shellcheck source=/dev/null
source "$ASCEND_VENV/bin/activate"
# shellcheck source=/dev/null
source "$ASCEND_SET_ENV"

if [[ "$SKIP_PIP_INSTALL" != "1" ]]; then
  echo
  echo "Installing Triton-Ascend backend dependencies..."
  python -m pip install -U pip
  python -m pip install --no-cache-dir --force-reinstall "triton-ascend==$TRITON_ASCEND_VERSION"
  python -m pip install --no-cache-dir \
    wheel \
    pybind11 \
    ninja \
    cmake \
    attrs==24.2.0 \
    numpy==1.26.4 \
    scipy==1.13.1 \
    decorator==5.1.1 \
    psutil==6.0.0 \
    pyyaml
else
  echo
  echo "Skipping pip install because SKIP_PIP_INSTALL=1."
fi

echo
echo "NPU device summary:"
npu-smi info || true

echo
echo "Python/NPU smoke test:"
python - <<'PY'
import platform
import torch
import torch_npu

print("python:", platform.python_version())
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("device count:", torch.npu.device_count())

x = torch.randn(1024, 1024).npu()
y = torch.nn.functional.gelu(x)
torch.npu.synchronize()

print("x:", x.device, x.dtype, tuple(x.shape))
print("y:", y.device, y.dtype, tuple(y.shape))
print("torch_npu smoke: ok")
PY

echo
echo "Triton-Ascend diagnostics:"
python scripts/diagnose_triton_ascend.py

cat <<EOF

Bootstrap complete.

Before running experiments in a new shell:
  source "$ASCEND_VENV/bin/activate"
  source "$ASCEND_SET_ENV"

Next project smoke run:
  bash scripts/create_gelu_triton_v9_submission.sh
  python scripts/probe_gelu_triton_backend.py \\
    --candidate outputs/submissions/gelu_triton_v9/gelu_triton_v9/t1/gelu.py \\
    --shape 32 512 1024
EOF
