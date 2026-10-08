#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
mkdir -p "$HERE/build"
# Recompile the NightRun-owned kernel, retaining our offline sm_89 image.
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -arch=sm_89 --cubin \
  "$HERE/nr_layer_mlp.cu" -o "$HERE/build/nr_layer_mlp_sm89.cubin"
g++ -O3 -std=c++17 -Wall -Wextra -pthread \
  -I"$CUDA_HOME/include" "$HERE/bridge_resident_mlp_service.cpp" \
  -L"$CUDA_HOME/lib64/stubs" -lcuda \
  -o "$HERE/build/bridge_resident_mlp_service"
echo 'Built unified NightRun GPU-resident model + layer-0 MLP executor.'
