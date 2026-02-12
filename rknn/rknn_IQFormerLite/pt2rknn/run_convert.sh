#!/bin/bash

# Default values
MODEL_PATH=""
KERNEL_SIZE=31
GRID_SIZE=4
GRID_RANGE="-2 2"
PLATFORM="rk3588"
AUX_MODE="kan"
DATABASE="2016.10a"
DATASET_PATH=""

# Python environments
RKNN_PYTHON="/home/cww/install/miniconda3/envs/rknn/bin/python"
BASE_PYTHON="$RKNN_PYTHON" # Use RKNN python for all operations

# Function to print usage
usage() {
    echo "Usage: $0 -m <model_path> [options]"
    echo ""
    echo "Arguments:"
    echo "  -m, --model       Path to the .pt model file (Required)"
    echo "  -k, --kernel      Kernel size (default: 31)"
    echo "  -g, --grid        Grid size (default: 4)"
    echo "  -p, --platform    Target RKNN platform (default: rk3588)"
    echo "  --aux_mode        Aux mode (default: kan)"
    echo "  --database        Database (default: 2016.10a)"
    echo "  --dataset         Dataset path for INT8 quantization"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -m ./weights.pt -k 17 -p rk3588"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -k|--kernel)
            KERNEL_SIZE="$2"
            shift 2
            ;;
        -g|--grid)
            GRID_SIZE="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        --aux_mode)
            AUX_MODE="$2"
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

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required."
    usage
fi

# Derive output paths
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_BASENAME=$(basename "$MODEL_PATH" .pt)
MODEL_ONNX="${MODEL_DIR}/${MODEL_BASENAME}.onnx"
MODEL_ONNX_FIXED="${MODEL_DIR}/${MODEL_BASENAME}_fixed.onnx"
MODEL_RKNN_FP32="${MODEL_DIR}/${MODEL_BASENAME}_fp32.rknn"
MODEL_RKNN_FP16="${MODEL_DIR}/${MODEL_BASENAME}_fp16.rknn"
MODEL_RKNN_INT8="${MODEL_DIR}/${MODEL_BASENAME}_int8.rknn"

echo "=========================================="
echo "Configuration:"
echo "  Model Path:    $MODEL_PATH"
echo "  ONNX Path:     $MODEL_ONNX"
echo "  Fixed ONNX:    $MODEL_ONNX_FIXED"
echo "  RKNN FP32:     $MODEL_RKNN_FP32"
echo "  RKNN FP16:     $MODEL_RKNN_FP16"
echo "  RKNN INT8:     $MODEL_RKNN_INT8"
echo "  Kernel Size:   $KERNEL_SIZE"
echo "  Grid Size:     $GRID_SIZE"
echo "  Platform:      $PLATFORM"
echo "  Dataset Path:  $DATASET_PATH"
echo "=========================================="

# Get directory of this script to find python scripts
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

echo "=========================================="
echo "Step 1: Converting PT to ONNX"
echo "=========================================="

$BASE_PYTHON "$SCRIPT_DIR/pt2onnx.py" \
  --model_path "$MODEL_PATH" \
  --onnx_path "$MODEL_ONNX" \
  --database_choose "$DATABASE" \
  --aux_mode "$AUX_MODE" \
  --grid_size "$GRID_SIZE" \
  --kernel_size "$KERNEL_SIZE" \
  --grid_range $GRID_RANGE

if [ $? -ne 0 ]; then
    echo "Error: PT to ONNX conversion failed."
    exit 1
fi

echo "=========================================="
echo "Step 2: Fixing ONNX Transpose Nodes"
echo "=========================================="

# Use RKNN python for fixing ONNX to ensure compatibility, or base python if onnx is installed there
# Using RKNN python because it definitely has onnx installed (as seen in previous steps)
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
