#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
mkdir -p "$HERE/build"
g++ -O3 -std=c++17 -Wall -Wextra -pthread \
  -I"$CUDA_HOME/include" "$HERE/bridge_model_service.cpp" \
  -L"$CUDA_HOME/lib64/stubs" -lcuda \
  -o "$HERE/build/bridge_model_service"
echo 'Built Stage 5A CUDA Driver API persistent model allocator.'
