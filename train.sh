#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

MAX_JOBS=${MAX_JOBS:-10}
KERNEL_SIZE=31
GRID_SIZE=4
GRID_RANGE="-2 2"
SEEDS=(0123 1234 2345 3456 4567 5678 6789 7890)

# Define the list of models to run
# MODELS=('IQFormerLite' 'IQFormer' 'MCFormer' 'AMCNET' 'MCLDNN' 'PETCGDNN' 'FEA_T128')
# MODELS=('MCFormer' 'AMCNET' 'MCLDNN' 'PETCGDNN' 'FEA_T128')
MODELS=('IQFormerLite')
running=0

for model in "${MODELS[@]}"; do
    # Check if the model supports KAN parameters (IQFormerLite and IQFormer)
    # if [[ "$model" == "IQFormerLite" || "$model" == "IQFormer" ]]; then
    if [ "$model" == "IQFormerLite" ]; then
        for seed in "${SEEDS[@]}"; do
          tag="${model}_kan_k${KERNEL_SIZE}_g${GRID_SIZE}_r${GRID_RANGE// /_}_seed${seed}"
          python main.py --model "$model" --aux_mode kan --kernel_size "$KERNEL_SIZE" --grid_size "$GRID_SIZE" --grid_range $GRID_RANGE --seed "$seed" --comment "$tag" &
          
          running=$((running + 1))
          if [ "$running" -ge "$MAX_JOBS" ]; then
            wait -n
            running=$((running - 1))
          fi
        done
    else
        # For other models, run with default settings (no KAN parameters)
        for seed in "${SEEDS[@]}"; do
          tag="${model}_base_seed${seed}"
          python main.py --model "$model" --aux_mode none --seed "$seed" --comment "$tag" &
          
          running=$((running + 1))
          if [ "$running" -ge "$MAX_JOBS" ]; then
              wait -n
              running=$((running - 1))
          fi
        done
    fi
done
wait
