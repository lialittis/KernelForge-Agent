#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

AKG_SUBMODULE_PATH="${AKG_SUBMODULE_PATH:-third_party/akg}"
RUNNER_PATH="${RUNNER_PATH:-$AKG_SUBMODULE_PATH/akg_agents/examples/kernel_related/run_torch_bench_lite.py}"
ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ASCEND_VENV="${ASCEND_VENV:-/data/venvs/kf-triton-ascend}"
CASE="${CASE:-sigmoid_scale_sum}"
BACKEND="${BACKEND:-npu}"
MODE="${MODE:-full}"
PASS_N="${PASS_N:-4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
OUTPUT="${OUTPUT:-outputs/results/akg_agents_full_${CASE}_$(date +%Y_%m_%d).json}"
AKG_AGENTS_LOG_DIR="${AKG_AGENTS_LOG_DIR:-$ROOT_DIR/outputs/akg_agents_logs}"
ALLOW_MISSING_MODEL_CONFIG="${ALLOW_MISSING_MODEL_CONFIG:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

usage() {
  cat <<'EOF'
Run the AKG Agents Bench Lite full-mode comparison entry point.

Environment overrides:
  CASE=sigmoid_scale_sum
  BACKEND=npu
  MODE=full
  PASS_N=4
  MAX_CONCURRENT=1
  OUTPUT=outputs/results/akg_agents_full_sigmoid_scale_sum_YYYY_MM_DD.json
  ASCEND_SET_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
  ASCEND_VENV=/data/venvs/kf-triton-ascend
  ALLOW_MISSING_MODEL_CONFIG=1   Run even if no standard model config is visible.
  CHECK_ONLY=1                   Print the resolved command without running it.

Any positional arguments after -- are passed through to run_torch_bench_lite.py.
EOF
}

extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --allow-missing-model-config)
      ALLOW_MISSING_MODEL_CONFIG=1
      shift
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -f "$ASCEND_SET_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$ASCEND_SET_ENV"
fi

if [[ -d "$ASCEND_VENV" ]]; then
  # shellcheck source=/dev/null
  source "$ASCEND_VENV/bin/activate"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/$AKG_SUBMODULE_PATH/akg_agents/python:${PYTHONPATH:-}"
export AKG_AGENTS_LOG_DIR
mkdir -p "$(dirname "$OUTPUT")" "$AKG_AGENTS_LOG_DIR"

if [[ ! -f "$RUNNER_PATH" ]]; then
  echo "AKG Agents runner not found: $RUNNER_PATH" >&2
  echo "Run: bash scripts/setup_akg_agents_runner_deps.sh" >&2
  exit 1
fi

has_standard_env=0
while IFS='=' read -r name _; do
  if [[ "$name" == AKG_AGENTS_STANDARD_* ]]; then
    has_standard_env=1
    break
  fi
done < <(env)

has_settings_file=0
for settings in "$HOME/.akg/settings.json" "$ROOT_DIR/.akg/settings.json" "$ROOT_DIR"/.akg/settings*.json; do
  if [[ -f "$settings" ]]; then
    has_settings_file=1
    break
  fi
done

if [[ "$has_standard_env" != "1" && "$has_settings_file" != "1" && "$CHECK_ONLY" == "1" ]]; then
  cat >&2 <<'EOF'
Warning: AKG Agents standard model configuration was not found.
CHECK_ONLY=1 is set, so the resolved command will be printed without running.
EOF
elif [[ "$has_standard_env" != "1" && "$has_settings_file" != "1" && "$ALLOW_MISSING_MODEL_CONFIG" != "1" ]]; then
  cat >&2 <<EOF
AKG Agents standard model configuration was not found.

Configure one of:
  - AKG_AGENTS_STANDARD_* environment variables
  - ~/.akg/settings.json
  - .akg/settings*.json

Then rerun this script. Use --allow-missing-model-config only for reproducing
the known pre-key failure path.
EOF
  exit 2
fi

cmd=(
  python "$RUNNER_PATH"
  --mode "$MODE"
  --backend "$BACKEND"
  --cases "$CASE"
  --pass-n "$PASS_N"
  --max-concurrent "$MAX_CONCURRENT"
  --output "$OUTPUT"
)

if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

printf 'Resolved AKG Agents command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

if [[ "$CHECK_ONLY" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"
