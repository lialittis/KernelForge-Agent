#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

AKG_SUBMODULE_PATH="${AKG_SUBMODULE_PATH:-third_party/akg}"
ASCEND_VENV="${ASCEND_VENV:-/data/venvs/kf-triton-ascend}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_SPARSE_ADD="${SKIP_SPARSE_ADD:-0}"

if [[ ! -d "$AKG_SUBMODULE_PATH/.git" && ! -f "$AKG_SUBMODULE_PATH/.git" ]]; then
  echo "AKG submodule is not initialized at $AKG_SUBMODULE_PATH." >&2
  echo "Run: bash scripts/setup_benchmark_submodule.sh" >&2
  exit 1
fi

if [[ "$SKIP_SPARSE_ADD" != "1" ]]; then
  echo "Ensuring AKG Agents runner paths are present in sparse checkout..."
  git -C "$AKG_SUBMODULE_PATH" sparse-checkout add akg_agents/
fi

if [[ ! -d "$ASCEND_VENV" ]]; then
  echo "Creating Python venv with system site packages: $ASCEND_VENV"
  mkdir -p "$(dirname "$ASCEND_VENV")"
  "$PYTHON_BIN" -m venv --system-site-packages "$ASCEND_VENV"
fi

# shellcheck source=/dev/null
source "$ASCEND_VENV/bin/activate"

python -m pip install \
  "langchain>=1.0.0" \
  "langchain-community>=0.4.0" \
  "langchain-core>=0.3.0" \
  "langgraph>=1.0.0" \
  "langchain-deepseek>=1.0.0" \
  "pandas>=2.0.0" \
  "tree-sitter>=0.21.0" \
  "tree-sitter-cpp>=0.22.0"

cat <<EOF

AKG Agents runner dependencies are installed.

Probe the runner imports with:
  source "$ASCEND_VENV/bin/activate"
  export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/$AKG_SUBMODULE_PATH/akg_agents/python:\${PYTHONPATH:-}"
  python $AKG_SUBMODULE_PATH/akg_agents/examples/kernel_related/run_torch_bench_lite.py --help

The runner still needs a configured AKG Agents model level before it can
generate candidates, for example AKG_AGENTS_STANDARD_* environment variables
or ~/.akg/settings.json as described by the AKG Agents error messages.
EOF
