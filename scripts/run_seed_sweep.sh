#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

DATABASE_PATH="${DATABASE_PATH:-./dataset}"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
LR="${LR:-0.001}"
MIN_SNR="${MIN_SNR:--20}"
MAX_SNR="${MAX_SNR:-18}"
TEST_SIZE="${TEST_SIZE:-0.2}"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
DATASET_LIST="${DATASET_LIST:-2016.10a 2016.10b}"
MODEL_LIST="${MODEL_LIST:-IQFormerLite IQFormer}"
MAX_JOBS="${MAX_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a DATASETS <<< "${DATASET_LIST}"
read -r -a MODELS <<< "${MODEL_LIST}"

declare -A AUX_MODE
AUX_MODE["IQFormerLite"]="kan"
AUX_MODE["IQFormer"]="stft"

declare -A EXTRA_ARGS
EXTRA_ARGS["IQFormerLite"]="--kernel_size 31 --grid_size 4 --grid_range -2 2"
EXTRA_ARGS["IQFormer"]=""

run_one() {
    local model="$1"
    local dataset="$2"
    local seed="$3"
    local aux_mode="$4"
    local extra_args="$5"
    local comment="${model}_aux-${aux_mode}_seed${seed}"
    local -a cmd=(
        python main.py
        --model "${model}"
        --database_path "${DATABASE_PATH}"
        --database_choose "${dataset}"
        --batch_size "${BATCH_SIZE}"
        --eval_batch_size "${EVAL_BATCH_SIZE}"
        --num_epochs "${NUM_EPOCHS}"
        --minSNR "${MIN_SNR}"
        --maxSNR "${MAX_SNR}"
        --test_size "${TEST_SIZE}"
        --lr "${LR}"
        --seed "${seed}"
        --comment "${comment}"
        --aux_mode "${aux_mode}"
    )

    echo "Running ${model} on ${dataset} with seed=${seed} aux_mode=${aux_mode}"
    if [[ -n "${extra_args}" ]]; then
        # shellcheck disable=SC2206
        cmd+=(${extra_args})
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        cmd+=(--dry_run)
    fi
    "${cmd[@]}"
}

pids=()
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        aux_mode="${AUX_MODE[${model}]:-none}"
        extra_args="${EXTRA_ARGS[${model}]:-}"
        for seed in "${SEEDS[@]}"; do
            if [[ "${MAX_JOBS}" -le 1 ]]; then
                run_one "${model}" "${dataset}" "${seed}" "${aux_mode}" "${extra_args}"
            else
                run_one "${model}" "${dataset}" "${seed}" "${aux_mode}" "${extra_args}" &
                pids+=("$!")
                if [[ "${#pids[@]}" -ge "${MAX_JOBS}" ]]; then
                    wait "${pids[0]}"
                    pids=("${pids[@]:1}")
                fi
            fi
        done
    done
done

if [[ "${MAX_JOBS}" -gt 1 ]]; then
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
fi

echo "All seed-sweep jobs finished."
