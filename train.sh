#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

MAX_JOBS=${MAX_JOBS:-10}
# KERNEL_SIZES=(15 17 31)
# GRID_SIZES=(2 4 8)
# GRID_RANGES=("-2 2" "-1 1")
KERNEL_SIZES=(15)
GRID_SIZES=(2)
GRID_RANGES=("-2 2")

# Define the list of models to run
# MODELS=('IQFormerLite' 'IQFormer' 'MCFormer' 'AMCNET' 'MCLDNN' 'PETCGDNN' 'FEA_T128')
# MODELS=('MCFormer' 'AMCNET' 'MCLDNN' 'PETCGDNN' 'FEA_T128')
MODELS=('IQFormer')
running=0

for model in "${MODELS[@]}"; do
    # Check if the model supports KAN parameters (IQFormerLite and IQFormer)
    # if [[ "$model" == "IQFormerLite" || "$model" == "IQFormer" ]]; then
    if [["$model" == "IQFormerLite"]]; then
        for kernel_size in "${KERNEL_SIZES[@]}"; do
          for grid_size in "${GRID_SIZES[@]}"; do
            for grid_range in "${GRID_RANGES[@]}"; do
              # Add model name to the tag
              tag="${model}_kan_k${kernel_size}_g${grid_size}_r${grid_range// /_}temp"
              python main.py --model "$model" --aux_mode kan --kernel_size "$kernel_size" --grid_size "$grid_size" --grid_range $grid_range --comment "$tag" &
              
              running=$((running + 1))
              if [ "$running" -ge "$MAX_JOBS" ]; then
                wait -n
                running=$((running - 1))
              fi
            done
          done
        done
    else
        # For other models, run with default settings (no KAN parameters)
        tag="${model}_base"
        python main.py --model "$model" --aux_mode none --comment "$tag" &
        
        running=$((running + 1))
        if [ "$running" -ge "$MAX_JOBS" ]; then
            wait -n
            running=$((running - 1))
        fi
    fi
done
wait
