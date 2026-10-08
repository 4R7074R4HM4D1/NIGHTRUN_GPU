#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p gpu-native/build
NVCC="${NVCC:-/usr/local/cuda-12.8/bin/nvcc}"
"$NVCC" -O3 -std=c++17 -arch=sm_89 --cubin \
  gpu-native/nr_attention.cu -o gpu-native/build/nr_attention_sm89.cubin
CXX="${CXX:-g++}"
"$CXX" -O3 -std=c++17 -Wall -Wextra -pthread \
  -I/usr/local/cuda-12.8/include \
  gpu-native/bridge_resident_attention_mlp_service.cpp \
  -L/usr/local/cuda-12.8/lib64/stubs -lcuda -o gpu-native/build/bridge_resident_attention_mlp_service
printf 'Built NightRun Stage 6A GPU-resident attention executor.\n'
