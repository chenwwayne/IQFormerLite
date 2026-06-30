#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

DATABASE_PATH="${DATABASE_PATH:-./dataset}"
DATASET_LIST="${DATASET_LIST:-2016.10a 2016.10b}"
KERNEL_LIST="${KERNEL_LIST:-7 15 51}"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
LR="${LR:-0.001}"
MAX_JOBS="${MAX_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
CONSOLE_LOG_DIR="${CONSOLE_LOG_DIR:-logs/conv_kernel_sweep_console}"

read -r -a DATASETS <<< "${DATASET_LIST}"
read -r -a KERNELS <<< "${KERNEL_LIST}"
read -r -a SEEDS <<< "${SEED_LIST}"
mkdir -p "${CONSOLE_LOG_DIR}"

run_one() {
    local dataset="$1"
    local kernel="$2"
    local seed="$3"
    local label="conv_k${kernel}"
    local comment="IQFormerLite_lkf-${label}_seed${seed}"
    local log_file="${CONSOLE_LOG_DIR}/${dataset}_${label}_seed${seed}.log"
    local -a cmd=(
        python main.py --model IQFormerLite
        --database_path "${DATABASE_PATH}" --database_choose "${dataset}"
        --batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}"
        --num_epochs "${NUM_EPOCHS}" --minSNR -20 --maxSNR 18 --test_size 0.2
        --lr "${LR}" --seed "${seed}" --comment "${comment}"
        --aux_mode kan --lkf_variant conv --band_k 32 --kernel_size "${kernel}"
        --grid_size 4 --grid_range -2 2 --skip_post_test_artifacts
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
        cmd+=(--dry_run)
    fi
    printf 'Running dataset=%s variant=%s seed=%s\n' "${dataset}" "${label}" "${seed}"
    "${cmd[@]}" 2>&1 | tee "${log_file}"
}

pids=()
for dataset in "${DATASETS[@]}"; do
    for kernel in "${KERNELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if [[ "${MAX_JOBS}" -le 1 ]]; then
                run_one "${dataset}" "${kernel}" "${seed}"
            else
                run_one "${dataset}" "${kernel}" "${seed}" &
                pids+=("$!")
                if [[ "${#pids[@]}" -ge "${MAX_JOBS}" ]]; then
                    wait "${pids[0]}"
                    pids=("${pids[@]:1}")
                fi
            fi
        done
    done
done
for pid in "${pids[@]}"; do wait "${pid}"; done
echo "All requested convolution kernel sweep jobs finished."
