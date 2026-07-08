#!/usr/bin/env bash
set -euo pipefail

AKG_REPO_URL="https://atomgit.com/mindspore/akg.git"
AKG_BRANCH="br_agents"
AKG_COMMIT="47aa428fcdc8c68f78d331dc578bc6c74fb9d91d"
AKG_SUBMODULE_PATH="third_party/akg"
BENCHMARK_PATH="akg_agents/benchmark/akg_kernels_bench_lite"

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

echo "Initializing AKG submodule..."
git submodule sync -- "$AKG_SUBMODULE_PATH"
git submodule update --init --depth 1 --filter=blob:none -- "$AKG_SUBMODULE_PATH"

echo "Pinning AKG submodule to $AKG_COMMIT..."
git -C "$AKG_SUBMODULE_PATH" fetch --depth 1 origin "$AKG_BRANCH"
git -C "$AKG_SUBMODULE_PATH" checkout --detach "$AKG_COMMIT"

echo "Enabling sparse checkout for benchmark path..."
git -C "$AKG_SUBMODULE_PATH" sparse-checkout init --no-cone
git -C "$AKG_SUBMODULE_PATH" sparse-checkout set --no-cone "$BENCHMARK_PATH/"

cat <<EOF
AKG benchmark is ready.

Repository: $AKG_REPO_URL
Branch:     $AKG_BRANCH
Commit:     $AKG_COMMIT
Path:       $AKG_SUBMODULE_PATH/$BENCHMARK_PATH
EOF
