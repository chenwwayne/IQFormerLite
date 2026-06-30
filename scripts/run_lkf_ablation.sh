#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

DATABASE_PATH="${DATABASE_PATH:-./dataset}"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
LR="${LR:-0.001}"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
DATASET_LIST="${DATASET_LIST:-2016.10a 2016.10b}"
VARIANTS_10A="${VARIANTS_10A:-none conv base_only rbf_only bspline}"
VARIANTS_10B="${VARIANTS_10B:-conv rbf_only}"
MAX_JOBS="${MAX_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
CONSOLE_LOG_DIR="${CONSOLE_LOG_DIR:-logs/lkf_ablation_console}"

read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a DATASETS <<< "${DATASET_LIST}"
mkdir -p "${CONSOLE_LOG_DIR}"

run_one() {
    local dataset="$1"
    local variant="$2"
    local seed="$3"
    local aux_mode="kan"
    local comment="IQFormerLite_lkf-${variant}_seed${seed}"
    local -a cmd=(
        python main.py
        --model IQFormerLite
        --database_path "${DATABASE_PATH}"
        --database_choose "${dataset}"
        --batch_size "${BATCH_SIZE}"
        --eval_batch_size "${EVAL_BATCH_SIZE}"
        --num_epochs "${NUM_EPOCHS}"
        --minSNR -20
        --maxSNR 18
        --test_size 0.2
        --lr "${LR}"
        --seed "${seed}"
        --comment "${comment}"
        --skip_post_test_artifacts
    )

    if [[ "${variant}" == "none" ]]; then
        aux_mode="none"
        cmd+=(--aux_mode none)
    else
        cmd+=(--aux_mode kan --lkf_variant "${variant}" --band_k 32 \
              --kernel_size 31 --grid_size 4 --grid_range -2 2)
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        cmd+=(--dry_run)
    fi

    local log_file="${CONSOLE_LOG_DIR}/${dataset}_${variant}_seed${seed}.log"
    printf 'Running dataset=%s variant=%s seed=%s aux_mode=%s\n' \
        "${dataset}" "${variant}" "${seed}" "${aux_mode}"
    "${cmd[@]}" 2>&1 | tee "${log_file}"
}

pids=()
for dataset in "${DATASETS[@]}"; do
    if [[ "${dataset}" == "2016.10a" ]]; then
        read -r -a variants <<< "${VARIANTS_10A}"
    elif [[ "${dataset}" == "2016.10b" ]]; then
        read -r -a variants <<< "${VARIANTS_10B}"
    else
        echo "Unsupported ablation dataset: ${dataset}" >&2
        exit 2
    fi
    for variant in "${variants[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if [[ "${MAX_JOBS}" -le 1 ]]; then
                run_one "${dataset}" "${variant}" "${seed}"
            else
                run_one "${dataset}" "${variant}" "${seed}" &
                pids+=("$!")
                if [[ "${#pids[@]}" -ge "${MAX_JOBS}" ]]; then
                    wait "${pids[0]}"
                    pids=("${pids[@]:1}")
                fi
            fi
        done
    done
done

for pid in "${pids[@]}"; do
    wait "${pid}"
done

echo "All requested LKF ablation jobs finished."
