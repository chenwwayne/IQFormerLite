#!/bin/bash

MODEL_PATH=""
PLATFORM="rk3588"
DATABASE="2016.10a"
DATASET_PATH=""

RKNN_PYTHON="/home/orangepi/miniconda3/envs/rknn/bin/python"
BASE_PYTHON="$RKNN_PYTHON"
export PYTHONNOUSERSITE=1

usage() {
    echo "Usage: $0 -m <model_path> [options]"
    echo "  -m, --model       Path to the .pt model file (Required)"
    echo "  -p, --platform    Target RKNN platform (default: rk3588)"
    echo "  --database        Database (default: 2016.10a)"
    echo "  --dataset         Dataset path for INT8 quantization"
    exit 1
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        --database)
            DATABASE="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required."
    usage
fi

MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_BASENAME="IQFormer"
MODEL_ONNX="${MODEL_DIR}/${MODEL_BASENAME}.onnx"
MODEL_ONNX_FIXED="${MODEL_DIR}/${MODEL_BASENAME}_fixed.onnx"
MODEL_RKNN_FP32="${MODEL_DIR}/${MODEL_BASENAME}_fp32.rknn"
MODEL_RKNN_FP16="${MODEL_DIR}/${MODEL_BASENAME}_fp16.rknn"
MODEL_RKNN_INT8="${MODEL_DIR}/${MODEL_BASENAME}_int8.rknn"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

if [ -z "$DATASET_PATH" ]; then
    DATASET_PATH="${PROJECT_ROOT}/calib/dataset.txt"
    if [ ! -f "$DATASET_PATH" ]; then
        find "${PROJECT_ROOT}/calib" -maxdepth 1 -name "iq_*.npy" | sort > "$DATASET_PATH"
    fi
fi

echo "=========================================="
echo "Configuration:"
echo "  Model Path:    $MODEL_PATH"
echo "  ONNX Path:     $MODEL_ONNX"
echo "  Fixed ONNX:    $MODEL_ONNX_FIXED"
echo "  RKNN FP32:     $MODEL_RKNN_FP32"
echo "  RKNN FP16:     $MODEL_RKNN_FP16"
echo "  RKNN INT8:     $MODEL_RKNN_INT8"
echo "  Platform:      $PLATFORM"
echo "  Dataset Path:  $DATASET_PATH"
echo "=========================================="

echo "=========================================="
echo "Step 1: Converting PT to ONNX"
echo "=========================================="

$BASE_PYTHON "$SCRIPT_DIR/pt2onnx.py" \
  --model_path "$MODEL_PATH" \
  --onnx_path "$MODEL_ONNX" \
  --database_choose "$DATABASE"

if [ $? -ne 0 ]; then
    echo "Error: PT to ONNX conversion failed."
    exit 1
fi

echo "=========================================="
echo "Step 2: Fixing ONNX Transpose Nodes"
echo "=========================================="

$RKNN_PYTHON "$SCRIPT_DIR/fix_onnx.py" \
  "$MODEL_ONNX" \
  "$MODEL_ONNX_FIXED"

if [ $? -ne 0 ]; then
    echo "Error: ONNX fix failed."
    exit 1
fi

echo "=========================================="
echo "Step 3: Converting ONNX to RKNN (FP32)"
echo "=========================================="

$RKNN_PYTHON "$SCRIPT_DIR/onnx2rknn.py" \
  "$MODEL_ONNX_FIXED" \
  "$PLATFORM" \
  fp32 \
  "$MODEL_RKNN_FP32"

if [ $? -ne 0 ]; then
    echo "Error: ONNX to RKNN conversion failed for FP32."
    exit 1
fi

echo "=========================================="
echo "Step 4: Converting ONNX to RKNN (FP16)"
echo "=========================================="

$RKNN_PYTHON "$SCRIPT_DIR/onnx2rknn.py" \
  "$MODEL_ONNX_FIXED" \
  "$PLATFORM" \
  fp16 \
  "$MODEL_RKNN_FP16"

if [ $? -ne 0 ]; then
    echo "Error: ONNX to RKNN conversion failed for FP16."
    exit 1
fi

echo "=========================================="
echo "Step 5: Converting ONNX to RKNN (INT8)"
echo "=========================================="

if [ -z "$DATASET_PATH" ]; then
    echo "Error: Dataset path is required for INT8 quantization."
    exit 1
fi

$RKNN_PYTHON "$SCRIPT_DIR/onnx2rknn.py" \
  "$MODEL_ONNX_FIXED" \
  "$PLATFORM" \
  int8 \
  "$MODEL_RKNN_INT8" \
  "$DATASET_PATH"

if [ $? -ne 0 ]; then
    echo "Error: ONNX to RKNN conversion failed for INT8."
    exit 1
fi

echo "=========================================="
echo "Success! RKNN models saved to:"
echo "  $MODEL_RKNN_FP32"
echo "  $MODEL_RKNN_FP16"
echo "  $MODEL_RKNN_INT8"
echo "=========================================="
