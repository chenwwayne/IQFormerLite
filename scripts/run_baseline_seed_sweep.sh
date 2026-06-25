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
MODEL_LIST="${MODEL_LIST:-MCLDNN MCFormer PETCGDNN AMCNET FEA_T128}"
GPU_LIST="${GPU_LIST:-0 1 2}"
MAX_JOBS="${MAX_JOBS:-3}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
SKIP_POST_TEST_ARTIFACTS="${SKIP_POST_TEST_ARTIFACTS:-0}"

read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a DATASETS <<< "${DATASET_LIST}"
read -r -a MODELS <<< "${MODEL_LIST}"
read -r -a GPUS <<< "${GPU_LIST}"

mkdir -p logs/seed_sweep_baselines

run_one() {
    local model="$1"
    local dataset="$2"
    local seed="$3"
    local gpu="$4"
    local comment="${model}_aux-none_seed${seed}"
    local run_tag="model_${dataset}_${NUM_EPOCHS}_${BATCH_SIZE}_${LR}_${comment}"
    local run_dir="logs/${run_tag}"
    local stdout_log="logs/seed_sweep_baselines/${run_tag}.log"

    if [[ "${SKIP_DONE}" == "1" && -f "${run_dir}/Test_ACC.csv" ]]; then
        echo "Skipping completed ${run_tag}"
        return 0
    fi

    echo "Running ${run_tag} on GPU ${gpu}"
    local -a cmd=(
        conda run --no-capture-output -n pytorch-3.9 python main.py
        --model "${model}" \
        --database_path "${DATABASE_PATH}" \
        --database_choose "${dataset}" \
        --batch_size "${BATCH_SIZE}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --num_epochs "${NUM_EPOCHS}" \
        --minSNR "${MIN_SNR}" \
        --maxSNR "${MAX_SNR}" \
        --test_size "${TEST_SIZE}" \
        --lr "${LR}" \
        --seed "${seed}" \
        --comment "${comment}" \
        --aux_mode none
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
        cmd+=(--dry_run)
    fi
    if [[ "${SKIP_POST_TEST_ARTIFACTS}" == "1" ]]; then
        cmd+=(--skip_post_test_artifacts)
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" 2>&1 | tee "${stdout_log}"
}

pids=()
job_idx=0
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
            if [[ "${MAX_JOBS}" -le 1 ]]; then
                run_one "${model}" "${dataset}" "${seed}" "${gpu}"
            else
                run_one "${model}" "${dataset}" "${seed}" "${gpu}" &
                pids+=("$!")
                if [[ "${#pids[@]}" -ge "${MAX_JOBS}" ]]; then
                    wait "${pids[0]}"
                    pids=("${pids[@]:1}")
                fi
            fi
            job_idx=$((job_idx + 1))
        done
    done
done

if [[ "${MAX_JOBS}" -gt 1 ]]; then
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
fi

echo "All baseline seed-sweep jobs finished."
