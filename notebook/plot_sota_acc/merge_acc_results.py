import pandas as pd
import os
import glob

# Define the working directory
base_dir = '/home/cww/IQFormer_lite/logs/RML201610a'
output_file = os.path.join(base_dir, 'RML201610a.csv')

# Find all directories matching the pattern model_*_base
search_pattern = os.path.join(base_dir, 'model_*_base')
directories = glob.glob(search_pattern)

# Initialize a list to store DataFrames
dfs = []

print(f"Found {len(directories)} directories.")

for d in directories:
    dir_name = os.path.basename(d)
    
    # Extract model name
    # Assumption: format is model_{database}_{epochs}_{batch}_{lr}_{model_name}_base
    # We split by '_' 5 times, the last part contains {model_name}_base
    try:
        parts = dir_name.split('_', 5)
        if len(parts) < 6:
            print(f"Skipping {dir_name}: unexpected format")
            continue
            
        model_part = parts[-1]
        if model_part.endswith('_base'):
            model_name = model_part[:-5] # Remove '_base'
        else:
            print(f"Skipping {dir_name}: does not end with _base")
            continue
        
        if model_name == 'PETCGDNN':
            model_name = 'PET-CGDNN'
            
        csv_path = os.path.join(d, 'Test_ACC.csv')
        if not os.path.exists(csv_path):
            print(f"Skipping {dir_name}: Test_ACC.csv not found")
            continue
            
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        # Ensure SNR is integer and remove non-numeric rows (like 'Avg')
        if 'SNR' in df.columns:
            df = df[df['SNR'] != 'Avg']
            df['SNR'] = df['SNR'].astype(int)
        
        # Rename the accuracy column ('0') to the model name
        if '0' in df.columns:
            df = df.rename(columns={'0': model_name})
        else:
            print(f"Warning: Column '0' not found in {csv_path}, columns are {df.columns}")
            # Assume the second column is accuracy if '0' is missing
            if len(df.columns) > 1:
                df = df.rename(columns={df.columns[1]: model_name})
            else:
                continue
                
        # Set SNR as index for easy merging
        df = df.set_index('SNR')
        
        # Add to list
        dfs.append(df)
        print(f"Loaded {model_name} from {dir_name}")
        
    except Exception as e:
        print(f"Error processing {dir_name}: {e}")

# Merge all DataFrames
if dfs:
    merged_df = pd.concat(dfs, axis=1)
    # Sort index (SNR)
    merged_df = merged_df.sort_index()
    
    # Reset index to make SNR a column again
    merged_df = merged_df.reset_index()
    
    desired_order = [
        'MCLDNN',
        'MCFormer',
        'PET-CGDNN',
        'AMC-Net',
        'FEA-T',
        'IQFormer',
        'IQFormerLite'
    ]
    existing_order = [col for col in desired_order if col in merged_df.columns]
    other_columns = [col for col in merged_df.columns if col not in existing_order and col != 'SNR']
    merged_df = merged_df[['SNR'] + existing_order + other_columns]
    
    # Save to CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully merged {len(dfs)} files into {output_file}")
    print(merged_df.head())
else:
    print("No data found to merge.")
