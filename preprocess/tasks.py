"""
Utility functions for separating the raw user data into tasks 
"""
import os
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import shutil

def extract_user_id(filename:str) -> int:
    """
    Extracts digits from a filename regardless of extension.
    Example: "msnv1.seg" or "msnv1.tsv" -> 1
    """
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_seg_id(filename:str) -> int:
    """
    Extracts a numeric ID from the filename by capturing digits that occur before the file extension.
    Works for names like "msnv1.seg", "msnv1.se", or "1.seg".
    """
    match = re.search(r'(\d+)(?=\.[^.]+$)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def split_from_seg(seg_directory:str, raw_directory:str, output_directory:str) -> None:
    """
    For each segmentation file (.seg) in seg_directory, this function:
      1. Reads the seg file (assumed to be whitespace-separated with no header).
         It assigns columns: 'mmd_id', 'mmd_id_dup', 'rec_start', 'rec_end'.
      2. Extracts the user id from the seg file name.
      3. Locates the corresponding raw TSV file for that user in raw_directory.
      4. Uses the rec_start and rec_end values (recording timestamps) to crop the raw data 
         based on its 'RecordingTimestamp' column.
      5. Saves the segmented file
    
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
    raw_directory_lower = raw_directory.lower()
    if 'adaptive-bar' in raw_directory_lower:
        prefix = "bar"
    elif 'adaptive-link' in raw_directory_lower:
        prefix = "link"
    elif 'control' in raw_directory_lower:
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
            segment_df = df_raw[(df_raw['RecordingTimestamp'] >= rec_start) &
                                (df_raw['RecordingTimestamp'] <= rec_end)].copy()
            

            segment_df = segment_df[valid_columns]

            output_filename = os.path.join(output_directory, f"{prefix}_{user_id}_{mmd_id}.pkl")
            with open(output_filename, 'wb') as f:
                pickle.dump(segment_df, f)
        
