#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

MAX_JOBS=${MAX_JOBS:-10}
KERNEL_SIZES=(15 17 31)
GRID_SIZES=(2 4 8)
GRID_RANGES=("-2 2" "-1 1")

running=0
for kernel_size in "${KERNEL_SIZES[@]}"; do
  for grid_size in "${GRID_SIZES[@]}"; do
    for grid_range in "${GRID_RANGES[@]}"; do
      tag="kan_k${kernel_size}_g${grid_size}_r${grid_range// /_}"
      python main.py --aux_mode kan --kernel_size "$kernel_size" --grid_size "$grid_size" --grid_range $grid_range --comment "$tag" &
      running=$((running + 1))
      if [ "$running" -ge "$MAX_JOBS" ]; then
        wait -n
        running=$((running - 1))
      fi
    done
  done
done
wait
