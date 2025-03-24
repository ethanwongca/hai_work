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

def crop_merge_and_cyclic_split_from_seg(seg_directory, raw_directory, output_directory, mode='full'):
    """
    Processes segmentation files (.seg) and corresponding raw TSV data to:
      1. For each user:
          - Read all tasks from the seg file.
          - For each task, crop the raw data based on rec_start and rec_end.
          - If mode is '29s', restrict each task to the first 29 seconds 
            (i.e. first 3480 rows at 120Hz). If the task is too short, keep what is available.
          - Filter out invalid rows (rows with NaN or -1 in valid columns) and keep only selected columns.
          - Tag each task with its mmd_id.
      2. Merge all tasks (sorted by task_id) for that user into a single DataFrame.
      3. Compute per-user statistics (e.g., total length, mean segment length, minimum and maximum task lengths in '29s' mode).
      4. Cyclically split the merged DataFrame into 4 groups.
      5. Save each cyclic split as a pickle file.
      6. Save group-level statistics across all users.
    
    Parameters:
      seg_directory (str): Directory containing segmentation (.seg) files.
      raw_directory (str): Directory containing raw TSV files for each user.
      output_directory (str): Directory where output pickle files and stats will be saved.
      mode (str): Either 'full' (use entire task duration) or '29s' (limit each task to the first 29 seconds at 120Hz).
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Define the valid columns.
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
    
    # Determine prefix from raw_directory name.
    raw_directory_lower = raw_directory.lower()
    if 'adaptive-bar' in raw_directory_lower:
        prefix = "bar"
    elif 'adaptive-link' in raw_directory_lower:
        prefix = "link"
    elif 'control' in raw_directory_lower:
        prefix = "ctrl"
    else:
        prefix = "unknown"
    
    # Container to hold per-user statistics.
    user_stats = []
    
    # List all segmentation files.
    seg_files = [f for f in os.listdir(seg_directory) if f.endswith('.seg')]
    
    for seg_file in seg_files:
        seg_path = os.path.join(seg_directory, seg_file)
        try:
            # Read segmentation file (whitespace-separated, no header).
            df_seg = pd.read_csv(seg_path, sep='\s+', header=None,
                                 names=['mmd_id', 'mmd_id_dup', 'rec_start', 'rec_end'])
        except Exception as e:
            print(f"Error reading seg file {seg_path}: {e}")
            continue
        
        # Extract user id from the seg filename.
        user_id = extract_user_id(seg_file)
        if user_id is None:
            print(f"Could not extract user id from seg file {seg_file}")
            continue
        
        # Find the corresponding raw TSV file.
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
        
        if 'RecordingTimestamp' not in df_raw.columns:
            print(f"'RecordingTimestamp' column not found in {raw_path}. Skipping user {user_id}.")
            continue
        
        # List to store valid segments (DataFrames) for this user.
        user_segments = []
        segment_lengths = []  # store each segment's length (number of rows)
        
        # Process each task in the segmentation file.
        for idx, row in df_seg.iterrows():
            mmd_id = row['mmd_id']
            rec_start = row['rec_start']
            rec_end = row['rec_end']
            
            # Process based on mode.
            if mode == '29s':
                # Grab first 29 seconds based on sampling rate (120Hz): 120*29 = 3480 rows.
                segment_df = df_raw[df_raw['RecordingTimestamp'] >= rec_start].copy()
                segment_df = segment_df.head(120 * 29)
            else:
                # Use the full duration.
                segment_df = df_raw[(df_raw['RecordingTimestamp'] >= rec_start) &
                                    (df_raw['RecordingTimestamp'] <= rec_end)].copy()
            
            if segment_df.empty:
                print(f"No data for user {user_id}, mmd_id {mmd_id} in the specified window.")
                continue
            
            # Remove rows with NaN in valid columns.
            segment_df = segment_df.dropna(subset=valid_columns)
            # Remove rows where any required column equals -1.
            for col in valid_columns:
                segment_df = segment_df[segment_df[col] != -1]
            
            if segment_df.empty:
                print(f"All rows invalid for user {user_id}, mmd_id {mmd_id}.")
                continue
            
            # Restrict to only the valid columns.
            if not set(valid_columns).issubset(segment_df.columns):
                print(f"Warning: Not all valid columns found for user {user_id}, mmd_id {mmd_id}. Skipping.")
                continue
            segment_df = segment_df[valid_columns].copy()
            
            # Add a column for the task id (for sorting later).
            segment_df['task_id'] = mmd_id
            
            # Record the length of this segment.
            seg_len = len(segment_df)
            segment_lengths.append(seg_len)
            
            user_segments.append(segment_df)
        
        if not user_segments:
            print(f"No valid segments found for user {user_id}")
            continue
        
        # Sort the segments by task_id (assuming mmd_id is numeric).
        user_segments_sorted = sorted(user_segments, key=lambda df: df['task_id'].iloc[0])
        
        # Merge (concatenate) all segments into a single DataFrame.
        merged_df = pd.concat(user_segments_sorted, ignore_index=True)
        total_length = len(merged_df)
        mean_length = sum(segment_lengths) / len(segment_lengths)
        min_length = min(segment_lengths)
        max_length = max(segment_lengths)
        
        # Record per-user stats.
        user_stats.append({
            'user_id': user_id,
            'num_segments': len(user_segments),
            'total_length': total_length,
            'mean_segment_length': mean_length,
            'min_segment_length': min_length if mode == '29s' else None,
            'max_segment_length': max_length if mode == '29s' else None
        })
        
        # Cyclically split the merged data into 4 groups.
        cyclic_groups = cyclic_split_df(merged_df, n=4)
        for i, cyclic_df in enumerate(cyclic_groups):
            output_filename = os.path.join(output_directory, f"{prefix}_{user_id}_{i}.pkl")
            with open(output_filename, 'wb') as f:
                pickle.dump(cyclic_df, f)
            print(f"Saved cyclic group {i} for user {user_id} as {output_filename}")
    
    # Compute group-level statistics.
    stats_df = pd.DataFrame(user_stats)
    print("Group-level statistics:")
    print(stats_df.describe())
    
    # Optionally, save the group stats.
    stats_filename = os.path.join(output_directory, f"{prefix}_group_stats.pkl")
    with open(stats_filename, 'wb') as f:
        pickle.dump(stats_df, f)
    print(f"Saved group-level stats as {stats_filename}")

if __name__ == "__main__":
    # For full tasks:
    crop_merge_and_cyclic_split_from_seg('control_segs', 'control/raw_data', 'data_combined', mode='29s')
    crop_merge_and_cyclic_split_from_seg('bar_segs', 'adaptive-bar/raw_data', 'data_combined', mode='29s')
    crop_merge_and_cyclic_split_from_seg('link_segs', 'adaptive-link/raw_data', 'data_combined', mode='29s')

