#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
mkdir -p "$HERE/build"
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -arch=sm_89 --cubin \
  -Xptxas=-v "$HERE/nr_layer_mlp.cu" -o "$HERE/build/nr_layer_mlp_sm89.cubin"
g++ -O3 -std=c++17 -Wall -Wextra -pthread \
  -I"$CUDA_HOME/include" "$HERE/bridge_mlp_service.cpp" \
  -L"$CUDA_HOME/lib64/stubs" -lcuda \
  -o "$HERE/build/bridge_mlp_service"
echo 'Built Stage 4 NightRun layer-0 MLP GPU kernel and PCI executor.'
