#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
mkdir -p "$SCRIPT_DIR/build"
"$CUDA_HOME/bin/nvcc" -O3 --std=c++17 --cubin -arch=sm_89 -Xptxas=-v \
  -I "$SCRIPT_DIR" "$SCRIPT_DIR/nr_transformer_chain.cu" \
  -o "$SCRIPT_DIR/build/nr_transformer_chain_sm89.cubin"
g++ -O3 -std=c++17 -Wall -Wextra \
  -I"$CUDA_HOME/include" "$SCRIPT_DIR/transformer_chain_test.cpp" \
  -L"$CUDA_HOME/lib64/stubs" -lcuda \
  -o "$SCRIPT_DIR/build/transformer_chain_test"
printf '\nBuilt Stage 2 CUBIN and CUDA Graph validation harness.\n'
