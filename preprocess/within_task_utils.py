import os
import pandas as pd
import pickle
import re

def extract_user_id(filename):
    """
    Extracts digits from a filename regardless of extension.
    Example: "msnv1.seg" or "msnv1.tsv" -> 1
    """
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def cyclic_split_df(df, n=4):
    """
    Splits a DataFrame into n groups in a cyclic (round-robin) manner.
    For example, with n=4:
      - Group 0 gets rows 0, 4, 8, ...
      - Group 1 gets rows 1, 5, 9, ...
      - etc.
    """
    return [df.iloc[i::n].reset_index(drop=True) for i in range(n)]

def crop_and_cyclic_split_from_seg(seg_directory, raw_directory, output_directory):
    """
    For each segmentation file (.seg) in seg_directory, this function:
      1. Reads the seg file (assumed to be whitespace-separated with no header).
         It assigns columns: 'mmd_id', 'mmd_id_dup', 'rec_start', 'rec_end'.
      2. Extracts the user id from the seg file name.
      3. Locates the corresponding raw TSV file for that user in raw_directory.
      4. Uses the rec_start and rec_end values (recording timestamps) to crop the raw data 
         based on its 'RecordingTimestamp' column.
      5. Restricts the cropped data to only the valid columns.
      6. Cyclically splits the segment into 4 groups.
      7. Saves each cyclic group as a pickle file named: 
             {prefix}_{user_id}_{mmd_id}_{i}.pkl
         where prefix is determined by the seg_directory name (e.g., "ctrl" for control).
    
    Parameters:
      seg_directory (str): Directory containing segmentation files (e.g., "control_segs").
      raw_directory (str): Directory containing raw TSV files for each user.
      output_directory (str): Directory where output pickle files will be saved.
    """
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Define the valid columns you want in the final pickle files.
    valid_columns = [
        'GazePointLeftX (ADCSpx)', 'GazePointLeftY (ADCSpx)',
        'GazePointRightX (ADCSpx)', 'GazePointRightY (ADCSpx)',
        'GazePointX (ADCSpx)', 'GazePointY (ADCSpx)', 
        'GazePointX (MCSpx)', 'GazePointY (MCSpx)',
        'GazePointLeftX (ADCSmm)', 'GazePointLeftY (ADCSmm)', 
        'GazePointRightX (ADCSmm)', 'GazePointRightY (ADCSmm)',
        'DistanceLeft', 'DistanceRight',
        'PupilLeft', 'PupilRight', 
        'FixationPointX (MCSpx)', 'FixationPointY (MCSpx)'
    ]
    
    # Determine prefix from seg_directory name.
    seg_directory_lower = seg_directory.lower()
    if 'adaptive-bar' in seg_directory_lower:
        prefix = "bar"
    elif 'adaptive-link' in seg_directory_lower:
        prefix = "link"
    elif 'control' in seg_directory_lower:
        prefix = "ctrl"
    else:
        prefix = "unknown"
    
    # List seg files
    seg_files = [f for f in os.listdir(seg_directory) if f.endswith('.seg')]
    
    for seg_file in seg_files:
        seg_path = os.path.join(seg_directory, seg_file)
        try:
            # Read the seg file (assuming whitespace-separated and no header)
            df_seg = pd.read_csv(seg_path, sep='\s+', header=None,
                                 names=['mmd_id', 'mmd_id_dup', 'rec_start', 'rec_end'])
        except Exception as e:
            print(f"Error reading seg file {seg_path}: {e}")
            continue
        
        # Extract user id from seg file name.
        user_id = extract_user_id(seg_file)
        if user_id is None:
            print(f"Could not extract user id from seg file {seg_file}")
            continue
        
        # Find the corresponding raw TSV file for this user.
        raw_files = [f for f in os.listdir(raw_directory) if f.endswith('.tsv') and str(user_id) in f]
        if not raw_files:
            print(f"No raw TSV file found for user {user_id} in {raw_directory}")
            continue
        raw_file = raw_files[0]  # assuming one raw file per user.
        raw_path = os.path.join(raw_directory, raw_file)
        
        try:
            df_raw = pd.read_csv(raw_path, delimiter='\t', low_memory=False)
        except Exception as e:
            print(f"Error reading raw file {raw_path}: {e}")
            continue
        
        # Ensure the raw data has a 'RecordingTimestamp' column.
        if 'RecordingTimestamp' not in df_raw.columns:
            print(f"'RecordingTimestamp' column not found in {raw_path}. Skipping user {user_id}.")
            continue
        
        # Process each task (row) in the seg file.
        for idx, row in df_seg.iterrows():
            mmd_id = row['mmd_id']
            rec_start = row['rec_start']
            rec_end = row['rec_end']
            
            # Crop the raw data based on RecordingTimestamp.
            # Here we assume the values are numeric.
            segment_df = df_raw[(df_raw['RecordingTimestamp'] >= rec_start) &
                                (df_raw['RecordingTimestamp'] <= rec_end)].copy()
            
            if segment_df.empty:
                print(f"No data found for user {user_id}, mmd_id {mmd_id} between {rec_start} and {rec_end}")
                continue
            
            # Restrict to the valid columns.
            if not set(valid_columns).issubset(segment_df.columns):
                print(f"Warning: Not all valid columns found in raw data for user {user_id}, mmd_id {mmd_id}. Skipping.")
                continue
            segment_df = segment_df[valid_columns]
            
            # Perform cyclic split into 4 groups.
            cyclic_groups = cyclic_split_df(segment_df, n=4)
            
            # Save each cyclic group as a pickle file.
            for i, cyclic_df in enumerate(cyclic_groups):
                output_filename = os.path.join(output_directory, f"{prefix}_{user_id}_{mmd_id}_{i}.pkl")
                with open(output_filename, 'wb') as f:
                    pickle.dump(cyclic_df, f)
                print(f"Saved cyclic segment {i} for user {user_id}, mmd_id {mmd_id} as {output_filename}")

# Example usage:
crop_and_cyclic_split_from_seg('control_segs', 'control/raw_data', 'control_output_segments')
crop_and_cyclic_split_from_seg('bar_segs', 'adaptive-bar/raw_data', 'bar_output_segments')
crop_and_cyclic_split_from_seg('link_segs', 'adaptive-link/raw_data', 'link_output_segments')
