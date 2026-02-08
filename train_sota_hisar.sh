#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export CUDA_VISIBLE_DEVICES

MAX_JOBS=${MAX_JOBS:-6}
KERNEL_SIZE=31
GRID_SIZE=4
GRID_RANGE="-2 2"
# SEEDS=(4567) # 0123 FOR RML201610A BEST 63.35
# SEEDS=(0123 2345 3456 4567 5678 6789 7890 8901 9012)
SEEDS=(2345)
# Define the list of models to run
# MODELS=('IQFormer' 'MCFormer' 'AMCNET' 'MCLDNN' 'PETCGDNN' 'FEA_T128')
MODELS=('IQFormer' 'MCFormer' 'AMCNET' 'MCLDNN' 'PETCGDNN' 'FEA_T128')
# MODELS=('IQFormerLite')
running=0

for model in "${MODELS[@]}"; do
    if [ "$model" == "IQFormerLite" ]; then
        for seed in "${SEEDS[@]}"; do
          tag="${model}_kan_k${KERNEL_SIZE}_g${GRID_SIZE}_r${GRID_RANGE}_seed${seed}"
          python main.py --model "$model" --aux_mode kan --kernel_size "$KERNEL_SIZE" --grid_size "$GRID_SIZE" --grid_range $GRID_RANGE --seed "$seed" --comment "$tag" --report --database_path /home/cww/dataset/HisarMod2019 --database_choose 2019  &
          
          running=$((running + 1))
          if [ "$running" -ge "$MAX_JOBS" ]; then
            wait -n
            running=$((running - 1))
          fi
        done
    else
        for seed in "${SEEDS[@]}"; do
          tag="${model}_base_seed${seed}"
          if [ "$model" == "IQFormer" ]; then
            python main.py --model "$model" --aux_mode stft --seed "$seed" --comment "$tag" --database_path /home/cww/dataset/HisarMod2019 --database_choose 2019 --batch_size 256 &
          else
            python main.py --model "$model" --aux_mode none --seed "$seed" --comment "$tag" --database_path /home/cww/dataset/HisarMod2019 --database_choose 2019 --batch_size 256 &
          fi
          
          running=$((running + 1))
          if [ "$running" -ge "$MAX_JOBS" ]; then
              wait -n
              running=$((running - 1))
          fi
        done
    fi
done
wait
