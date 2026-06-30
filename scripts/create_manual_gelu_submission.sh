#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
SUBMISSION_ROOT="$ROOT_DIR/outputs/submissions/manual_baseline/manual_baseline"

mkdir -p "$SUBMISSION_ROOT/t1"

cat > "$SUBMISSION_ROOT/meta.json" <<'JSON'
{
  "team_name": "manual_baseline"
}
JSON

cat > "$SUBMISSION_ROOT/t1/gelu.py" <<'PY'
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return torch.nn.functional.gelu(input_tensor)
PY

cat <<EOF
Manual GELU submission created:

$SUBMISSION_ROOT
EOF

