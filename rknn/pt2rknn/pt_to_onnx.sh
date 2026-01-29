# 定义路径变量
MODEL_PATH="/home/cww/IQFormer/save_models/model_2016.10a_60_1024_0.001_kan_k31_g4_r-2_2"

python /home/cww/IQFormer/utils/pt_to_onnx.py \
  --model_path  $MODEL_PATH/weight.pt \
  --onnx_path   $MODEL_PATH/weight_kan.onnx \
  --database_choose 2016.10a \
  --aux_mode kan \
  --kernel_size 31 \
  --grid_size 4 \
  --grid_range -2 2 \
