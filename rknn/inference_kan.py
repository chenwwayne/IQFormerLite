import numpy as np
import os
import argparse
import time
import sys

# Try to import RKNN or RKNNLite
try:
    from rknn.api import RKNN
    RKNN_TYPE = 'toolkit'
except ImportError:
    try:
        from rknnlite.api import RKNNLite as RKNN
        RKNN_TYPE = 'lite'
    except ImportError:
        print("Error: neither rknn.api nor rknnlite.api is available.")
        sys.exit(1)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def generate_simulated_data(sim_type, shape):
    """
    Generate simulated data based on type.
    shape: (batch, channels, length) e.g., (1, 2, 1024)
    """
    if sim_type == 'sine':
        print("Generating sine wave data...")
        length = shape[-1]
        # Create a time vector
        t = np.linspace(0, 10 * np.pi, length)
        
        # Generate data: Channel 0 = Cosine, Channel 1 = Sine
        data = np.zeros(shape, dtype=np.float32)
        
        # Handle different number of dimensions
        if len(shape) == 3: # (N, C, L)
            for i in range(shape[0]):
                data[i, 0, :] = np.cos(t)
                data[i, 1, :] = np.sin(t)
        elif len(shape) == 2: # (C, L)
            data[0, :] = np.cos(t)
            data[1, :] = np.sin(t)
            
        return data
        
    elif sim_type == 'constant':
        print("Generating constant data (ones)...")
        return np.ones(shape, dtype=np.float32)
        
    elif sim_type == 'zeros':
        print("Generating constant data (zeros)...")
        return np.zeros(shape, dtype=np.float32)
        
    else: # random
        print(f"Generating random noise data (type: {sim_type})...")
        return np.random.randn(*shape).astype(np.float32)

def load_data(data_path=None, shape=(1, 2, 1024), sim_type='random'):
    """
    Load data from file or generate random data.
    Supports .h5 (RML2018 format) and .pkl (RML2016 format).
    """
    if data_path is None:
        return generate_simulated_data(sim_type, shape), None
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    
    if data_path.endswith('.h5'):
        import h5py
        f = h5py.File(data_path, 'r')
        # Based on RML2018_random in dataset.py
        samples = f['samples'][:] 
        # RML2018 usually is (N, 1024, 2). 
        # But IQFormer stemIQ conv1d expects (Batch, Channels, Length) -> (N, 2, 1024).
        if samples.shape[-1] == 2: # (N, 1024, 2)
            samples = samples.transpose(0, 2, 1) # -> (N, 2, 1024)
        
        labels = f['label'][:]
        f.close()
        return samples.astype(np.float32), labels

    elif data_path.endswith('.pkl') or data_path.endswith('.dat'):
        import pickle
        with open(data_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])
        samples = []
        labels = []
        
        # Extract all samples
        for mod in mods:
            for snr in snrs:
                if (mod, snr) in data:
                    s = data[(mod, snr)]
                    samples.append(s)
                    labels.extend([mod]*len(s))
        
        samples = np.vstack(samples) 
        return samples.astype(np.float32), labels

    else:
        print("Unsupported file format.")
        sys.exit(1)

def show_outputs(output, top_k=5):
    # output is usually (1, num_classes)
    output = output.flatten()
    probabilities = softmax(output)
    indices = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
    
    print("----- Top {} Predictions -----".format(top_k))
    for i in range(min(top_k, len(probabilities))):
        idx = indices[i]
        print(f"Class {idx}: {probabilities[idx]:.6f}")
    print("----------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RKNN Inference Script for IQFormer (KAN)')
    parser.add_argument('--model', type=str, default='/home/cww/IQFormer_lite/save_models/model_2016.10a_60_1024_0.001_kan_k31_g4_r-2_2/weight_kan.rknn', help='Path to .rknn model file')
    parser.add_argument('--data', type=str, default=None, help='Path to data file (.h5 or .pkl)')
    parser.add_argument('--simulate', type=str, default='random', choices=['random', 'sine', 'constant', 'zeros'], help='Type of simulated input data (default: random)')
    parser.add_argument('--target', type=str, default='rk3588', help='Target platform (e.g. rk3588)')
    parser.add_argument('--device_id', type=str, default=None, help='Device ID for adb')
    args = parser.parse_args()

    # 1. Create RKNN object
    print(f"Using {RKNN_TYPE.upper()} API")
    rknn = RKNN(verbose=True)

    # 2. Load RKNN model
    print('--> Loading model')
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        print('Load RKNN model failed!')
        sys.exit(ret)
    print('done')

    # 3. Init runtime environment
    print('--> Init runtime environment')
    if RKNN_TYPE == 'toolkit':
        # On PC, target specifies the simulation target or connected device
        ret = rknn.init_runtime(target=args.target, device_id=args.device_id)
    else:
        # On board (Lite), usually no arguments needed for default NPU
        ret = rknn.init_runtime()
        
    if ret != 0:
        print('Init runtime environment failed!')
        sys.exit(ret)
    print('done')

    # 4. Prepare Input
    # Model expects (1, 2, 1024) based on description
    # We take one sample for inference
    samples, _ = load_data(args.data, shape=(1, 2, 1024), sim_type=args.simulate)
    
    # Take the first sample and add batch dimension if needed
    input_data = samples[0] # (2, 1024)
    input_data = np.expand_dims(input_data, 0) # (1, 2, 1024)
    
    print(f"Input shape: {input_data.shape}")

    # 5. Inference
    print('--> Running model')
    start_time = time.time()
    outputs = rknn.inference(inputs=[input_data], data_format=['nchw']) 
    end_time = time.time()
    print('done')
    print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")

    # 6. Post-process
    # outputs is a list of numpy arrays
    print(f"Output shape: {outputs[0].shape}")
    show_outputs(outputs[0])

    # 7. Release
    rknn.release()
