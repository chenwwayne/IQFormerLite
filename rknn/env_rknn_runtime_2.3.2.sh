#!/usr/bin/env bash

RUNTIME_LIB_DIR="/home/orangepi/IQFormerLite/rknn/runtime/lib"

if [ ! -f "${RUNTIME_LIB_DIR}/librknnrt.so" ]; then
  echo "Missing ${RUNTIME_LIB_DIR}/librknnrt.so" >&2
  return 1 2>/dev/null || exit 1
fi

export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "Using RKNN runtime: ${RUNTIME_LIB_DIR}/librknnrt.so"
strings "${RUNTIME_LIB_DIR}/librknnrt.so" | grep 'librknnrt version' | head -n 1
