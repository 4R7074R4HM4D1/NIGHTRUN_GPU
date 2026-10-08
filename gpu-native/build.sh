#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
mkdir -p "$SCRIPT_DIR/build"
"$CUDA_HOME/bin/nvcc" -O3 --std=c++17 --cubin -arch=sm_89 -Xptxas=-v \
  "$SCRIPT_DIR/nr_q8_matvec.cu" \
  -o "$SCRIPT_DIR/build/nr_q8_matvec_sm89.cubin"
g++ -O3 -std=c++17 -Wall -Wextra \
  -I"$CUDA_HOME/include" "$SCRIPT_DIR/q8_driver_test.cpp" \
  -L"$CUDA_HOME/lib64/stubs" -lcuda \
  -o "$SCRIPT_DIR/build/q8_driver_test"
printf '\nBuilt device kernel and CUDA Driver API validation harness.\n'
