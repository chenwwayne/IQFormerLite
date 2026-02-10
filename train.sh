#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES
MAX_JOBS=${MAX_JOBS:-10}

DATABASE_PATH="/home/cww/IQFormer_lite/dataset"
DATABASE_CHOOSE="2016.10a"

# MODELS=('IQFormerLite' 'IQFormer' 'MCFormer' 'AMCNET' 'MCLDNN' 'PETCGDNN' 'FEA_T128')
MODELS=('MCLDNN')

SEED=1234
BATCH_SIZE=1024
NUM_EPOCHS=100

running=0

for model in "${MODELS[@]}"; do
    echo "Starting training for $model..."
    
    ARGS=""
    if [ "$model" == "IQFormerLite" ]; then
        ARGS="--aux_mode kan --kernel_size 31 --grid_size 4 --grid_range -2 2"
    else
        ARGS="--aux_mode none"
    fi

    python main.py \
        --model "$model" \
        --database_path "$DATABASE_PATH" \
        --database_choose "$DATABASE_CHOOSE" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --seed "$SEED" \
        --comment "${model}_seed{$SEED}" \
        $ARGS &

    running=$((running + 1))
    if [ "$running" -ge "$MAX_JOBS" ]; then
        wait -n
        running=$((running - 1))
    fi
done

wait
echo "All training jobs finished."
