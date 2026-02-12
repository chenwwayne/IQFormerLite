# conda activate rknn
sudo chmod a+rx /sys/kernel/debug
sudo chmod a+rx /sys/kernel/debug/rknpu
sudo chmod a+r /sys/kernel/debug/rknpu/load
# there is an ouput if excute the following command
# ls -ld /sys/kernel/debug /sys/kernel/debug/rknpu
# cat /sys/kernel/debug/rknpu/load
PYTHONNOUSERSITE=1 /home/orangepi/miniconda3/envs/rknn/bin/python inference_on_rk3588.py --models_dir /home/orangepi/IQFormerLite/rknn_iqformer/weights/IQFormer --output_csv /home/orangepi/IQFormerLite/rknn_iqformer/benchmark.csv
