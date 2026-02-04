"""
FLOPs Calculation Script
========================

Function:
    Calculates and reports the FLOPs (Floating Point Operations) and Parameters 
    for 8 specific deep learning models: IQFormer, IQFormerLite, MCLDNN, AMCNET, 
    MCFormer, PETCGDNN, FEA_T128, and FEA_T1024.

Input:
    - Models are initialized with specific parameters (mostly for 11 classes, 128 input length).
    - Dummy input tensors are generated with shapes (1, 2, 128) or (1, 2, 1024).

Output:
    - Prints a formatted table showing Model Name, Input Shape, FLOPs (M), and Params (M).
"""

import torch
import torch.nn as nn
from thop import profile
import sys
import os
from prettytable import PrettyTable

# Add project root directory to path to allow importing models
# Assuming this script is in /utils/, so ".." takes us to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import models
try:
    from model.IQFormer import IQFormer
    from model.IQFormerLite import IQFormerLite
    from model.MCLDNN import MCLDNN
    from model.AMCNET import AMC_Net
    from model.MCFormer import MCformer
    from model.PETCGDNN import PETCGDNN
    import model.FEA_T128 as FEA_T128_module
    import model.FEA_T1024 as FEA_T1024_module
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Ensure you are running this script from the project root or utils directory and dependencies are met.")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_flops(model_name, model, input_shape=None, inputs=None):
    """
    Calculates FLOPs and Params for a given model.
    
    Args:
        model_name (str): Name of the model.
        model (nn.Module): The model instance.
        input_shape (tuple): Shape of the single dummy input tensor.
        inputs (tuple): Tuple of input tensors (overrides input_shape).
        
    Returns:
        tuple: (flops_in_million, params_in_million) or (None, None) if failed.
    """
    try:
        model = model.to(device)
        if inputs is None:
            if input_shape is None:
                raise ValueError("Either input_shape or inputs must be provided")
            inputs = (torch.randn(input_shape).to(device),)
        else:
            inputs = tuple(t.to(device) for t in inputs)
            
        # Use thop to calculate FLOPs
        flops, params = profile(model, inputs=inputs, verbose=False)
        return flops / 1e6, params / 1e6
    except Exception as e:
        print(f"Error calculating FLOPs for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    print(f"Calculating FLOPs on device: {device}...\n")
    
    # Initialize PrettyTable
    table = PrettyTable()
    table.field_names = ["Model", "Input Shape", "FLOPs (M)", "Params (M)"]
    table.float_format = ".2"
    table.align = "l"

    # 1. IQFormer
    try:
        # IQFormer requires 2 inputs: x (B, 2, 128) and stft (B, 1, 32, 128)
        iqformer = IQFormer([2,3,2], embed_dims=[64,64,64],
                mlp_ratios=1,
                act_layer=nn.GELU,
                num_classes=11,
                down_patch_size=3, down_stride=2, down_pad=1,
                drop_rate=0.2, drop_path_rate=0.,
                use_layer_scale=False, layer_scale_init_value=1e-5,
                fork_feat=False,
                vit_num=1)
        
        x_input = torch.randn(1, 2, 128)
        stft_input = torch.randn(1, 1, 32, 128)
        
        flops, params = calculate_flops("IQFormer", iqformer, inputs=(x_input, stft_input))
        if flops is not None:
            table.add_row(["IQFormer", "(1, 2, 128) + STFT", flops, params])
    except Exception as e:
        print(f"IQFormer Init Error: {e}")

    # 2. IQFormerLite
    try:
        # Using same configuration as IQFormer for comparison
        iqformer_lite = IQFormerLite([2,3,2], embed_dims=[64,64,64],
                mlp_ratios=1,
                act_layer=nn.GELU,
                num_classes=11,
                down_patch_size=3, down_stride=2, down_pad=1,
                drop_rate=0.2, drop_path_rate=0.,
                use_layer_scale=False, layer_scale_init_value=1e-5,
                fork_feat=False,
                vit_num=1,
                aux_mode='none',
                band_k=32)
        flops, params = calculate_flops("IQFormerLite", iqformer_lite, input_shape=(1, 2, 128))
        if flops is not None:
            table.add_row(["IQFormerLite", "(1, 2, 128)", flops, params])
    except Exception as e:
        print(f"IQFormerLite Init Error: {e}")

    # 3. MCLDNN
    try:
        mcldnn = MCLDNN(num_classes=11)
        flops, params = calculate_flops("MCLDNN", mcldnn, input_shape=(1, 2, 128))
        if flops is not None:
            table.add_row(["MCLDNN", "(1, 2, 128)", flops, params])
    except Exception as e:
        print(f"MCLDNN Init Error: {e}")

    # 4. AMCNET
    try:
        # Defaults to sig_len=1024, adjust to 128 to match others/input
        amcnet = AMC_Net(num_classes=11, sig_len=128)
        flops, params = calculate_flops("AMCNET", amcnet, input_shape=(1, 2, 128))
        if flops is not None:
            table.add_row(["AMCNET", "(1, 2, 128)", flops, params])
    except Exception as e:
        print(f"AMCNET Init Error: {e}")

    # 5. MCFormer
    try:
        mcformer = MCformer(num_classes=11)
        flops, params = calculate_flops("MCFormer", mcformer, input_shape=(1, 2, 128))
        if flops is not None:
            table.add_row(["MCFormer", "(1, 2, 128)", flops, params])
    except Exception as e:
        print(f"MCFormer Init Error: {e}")

    # 6. PETCGDNN
    try:
        # Defaults to frame_length=1024, adjust to 128
        petcgdnn = PETCGDNN(num_classes=11, frame_length=128)
        flops, params = calculate_flops("PETCGDNN", petcgdnn, input_shape=(1, 2, 128))
        if flops is not None:
            table.add_row(["PETCGDNN", "(1, 2, 128)", flops, params])
    except Exception as e:
        print(f"PETCGDNN Init Error: {e}")

    # 7. FEA_T128
    try:
        fea_t128 = FEA_T128_module.FEA_T(seq_length=128, num_class=11)
        flops, params = calculate_flops("FEA_T128", fea_t128, input_shape=(1, 2, 128))
        if flops is not None:
            table.add_row(["FEA_T128", "(1, 2, 128)", flops, params])
    except Exception as e:
        print(f"FEA_T128 Init Error: {e}")

    # 8. FEA_T1024
    try:
        # Input size should be 1024 for this model
        fea_t1024 = FEA_T1024_module.FEA_T(seq_length=1024, num_class=11)
        flops, params = calculate_flops("FEA_T1024", fea_t1024, input_shape=(1, 2, 1024))
        if flops is not None:
            table.add_row(["FEA_T1024", "(1, 2, 1024)", flops, params])
    except Exception as e:
        print(f"FEA_T1024 Init Error: {e}")

    print(table)

if __name__ == "__main__":
    main()
