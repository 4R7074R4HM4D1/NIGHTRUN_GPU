#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
mkdir -p "$HERE/build"
g++ -O3 -std=c++17 -Wall -Wextra -pthread \
  -I"$CUDA_HOME/include" "$HERE/bridge_chain_service.cpp" \
  -L"$CUDA_HOME/lib64/stubs" -lcuda \
  -o "$HERE/build/bridge_chain_service"
echo "Built Stage 3 QEMU command executor (CUDA Driver API)."
