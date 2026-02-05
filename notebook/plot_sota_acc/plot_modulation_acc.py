
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def plot_modulation_accuracy():
    # Set paths
    base_path = '/home/cww/IQFormer_lite/notebook/plot_sota_acc'
    
    # Find all model directories ending with 'base'
    model_dirs = glob.glob(os.path.join(base_path, 'model*base'))
    
    if not model_dirs:
        print("No model directories found!")
        return

    # Style settings matching the notebook
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h', 'H', 'x', 'd', '+', '1']
    # Use tab20 colors
    colors = plt.cm.tab20.colors
    
    for model_dir in model_dirs:
        dir_name = os.path.basename(model_dir)
        
        # Extract model name
        # Assuming format: model_2016.10a_60_1024_0.001_AMC-Net_base
        try:
            parts = dir_name.split('_')
            # The model name is typically the 6th element (index 5)
            # e.g., parts[5] is 'AMC-Net'
            model_name = parts[5]
        except IndexError:
            model_name = dir_name
            print(f"Could not parse model name from {dir_name}, using full directory name.")

        csv_path = os.path.join(model_dir, 'Test_mod_SNR.csv')
        
        if not os.path.exists(csv_path):
            print(f"Skipping {model_name}: {csv_path} not found.")
            continue
            
        print(f"Processing {model_name}...")
        
        # Read Data
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        # Create Plot
        plt.figure(figsize=(10, 8)) # Slightly larger to accommodate legend if needed
        
        # Identify modulation columns (all columns except SNR)
        # Ensure 'SNR' is not treated as a modulation
        mod_columns = [col for col in df.columns if col.lower() != 'snr']
        
        # Plot each modulation curve
        for idx, column in enumerate(mod_columns):
            plt.plot(
                df['SNR'],
                df[column],
                label=column,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2,
                markersize=8
            )

        # Axis and Grid setup
        plt.xlim(-22, 20)
        plt.ylim(0, 1.05) # Little bit of headroom for legend if needed
        
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        
        plt.xticks(range(-20, 21, 5), fontsize=12)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
        
        # Labels and Title
        plt.title(f'{model_name} Modulation Accuracy', fontsize=18)
        plt.xlabel('SNR (dB)', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        
        # Legend
        # Using 'best' location to avoid covering data lines, or outside
        plt.legend(fontsize=10, loc='best', ncol=2) 
        
        plt.tight_layout()
        
        # Save figure
        save_filename = f'plot_{model_name}_mod_acc.png'
        save_path = os.path.join(base_path, "mod_acc_result", save_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=450)
        plt.close()
        
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_modulation_accuracy()
