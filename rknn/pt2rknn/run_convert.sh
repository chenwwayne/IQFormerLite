#!/bin/bash

# Default values
MODEL_PATH=""
KERNEL_SIZE=31
GRID_SIZE=4
GRID_RANGE="-2 2"
PLATFORM="rk3588"
AUX_MODE="kan"
DATABASE="2016.10a"

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
MODEL_RKNN="${MODEL_DIR}/${MODEL_BASENAME}.rknn"

echo "=========================================="
echo "Configuration:"
echo "  Model Path:    $MODEL_PATH"
echo "  ONNX Path:     $MODEL_ONNX"
echo "  Fixed ONNX:    $MODEL_ONNX_FIXED"
echo "  RKNN Path:     $MODEL_RKNN"
echo "  Kernel Size:   $KERNEL_SIZE"
echo "  Grid Size:     $GRID_SIZE"
echo "  Platform:      $PLATFORM"
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
echo "Step 3: Converting ONNX to RKNN"
echo "=========================================="

$RKNN_PYTHON "$SCRIPT_DIR/onnx2rknn.py" \
  "$MODEL_ONNX_FIXED" \
  "$PLATFORM" \
  fp \
  "$MODEL_RKNN"

if [ $? -ne 0 ]; then
    echo "Error: ONNX to RKNN conversion failed."
    exit 1
fi

echo "=========================================="
echo "Success! RKNN model saved to:"
echo "  $MODEL_RKNN"
echo "=========================================="
