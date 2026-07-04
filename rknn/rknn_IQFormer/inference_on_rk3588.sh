# conda activate rknn
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../env_rknn_runtime_2.3.2.sh"
sudo chmod a+rx /sys/kernel/debug
sudo chmod a+rx /sys/kernel/debug/rknpu
sudo chmod a+r /sys/kernel/debug/rknpu/load
# there is an ouput if excute the following command
# ls -ld /sys/kernel/debug /sys/kernel/debug/rknpu
# cat /sys/kernel/debug/rknpu/load
PYTHONNOUSERSITE=1 /home/orangepi/miniconda3/envs/rknn/bin/python "${SCRIPT_DIR}/inference_on_rk3588.py" --models_dir /home/orangepi/IQFormerLite/rknn/rknn_IQFormer/weights/IQFormer --output_csv /home/orangepi/IQFormerLite/rknn/rknn_IQFormer/benchmark.csv
