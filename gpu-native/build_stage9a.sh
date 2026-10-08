#!/usr/bin/env bash
# Stage 9A: offline compile NVIDIA kernels, then link exact CUBINs into UEFI.
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC="${NVCC:-/usr/local/cuda-12.8/bin/nvcc}"
mkdir -p gpu-native/build
for name in nr_layer_mlp nr_attention nr_rope_kv; do
  "$NVCC" -O3 -std=c++17 -arch=sm_89 --cubin \
    "gpu-native/${name}.cu" -o "gpu-native/build/${name}_sm89.cubin"
  printf '[stage9a-build] %s: ' "$name"
  stat -c '%s bytes' "gpu-native/build/${name}_sm89.cubin"
done
# Existing build script rebuilds the CUDA Driver API executor.
bash gpu-native/build_stage7.sh
printf '[stage9a-build] All 3 CUBIN modules ready; now run cargo xtask build.\n'
