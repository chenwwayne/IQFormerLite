import sys
import os
import onnx
from rknn.api import RKNN

# Matches IQFormer input shape (Batch=1, Channels=2, Length=128)
DEFAULT_INPUT_SIZE_LIST = [[1, 2, 128]]

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path platform [dtype(optional)] [output_rknn_path(optional)] [dataset(optional)]".format(sys.argv[0]))
        print("       dtype choose from [fp32, fp16, int8]")
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    model_type = sys.argv[3] if len(sys.argv) > 3 else "fp32"
    if model_type in ["fp", "fp32"]:
        dtype = "fp32"
    elif model_type in ["fp16"]:
        dtype = "fp16"
    elif model_type in ["int8", "quant"]:
        dtype = "int8"
    else:
        print("Invalid dtype: {}".format(model_type))
        exit(1)

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = model_path.replace('.onnx', '.rknn')

    dataset_path = sys.argv[5] if len(sys.argv) > 5 else None

    do_quant = dtype == "int8"
    return model_path, platform, dtype, do_quant, output_path, dataset_path

if __name__ == '__main__':
    model_path, platform, dtype, do_quant, output_path, dataset_path = parse_arg()

    # Step 1: No internal fix here, assume input model is already fixed by external script
    
    # Step 2: Convert to RKNN
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    if dtype == "fp16":
        rknn.config(target_platform=platform, float_dtype="float16")
    else:
        rknn.config(target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    # Explicitly set inputs=['iq'] to match IQFormer model
    ret = rknn.load_onnx(model=model_path, inputs=['iq'], input_size_list=DEFAULT_INPUT_SIZE_LIST)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    if do_quant:
        if not dataset_path:
            print('Dataset path is required for INT8 quantization!')
            exit(1)
        ret = rknn.build(do_quantization=True, dataset=dataset_path)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
