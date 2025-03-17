import os
import pandas as pd
import pickle
import re
import matplotlib.pyploy as plt
import shutil

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

def extract_seg_id(filename):
    """
    Extracts a numeric ID from the filename by capturing digits that occur before the file extension.
    Works for names like "msnv1.seg", "msnv1.se", or "1.seg".
    """
    match = re.search(r'(\d+)(?=\.[^.]+$)', filename)
    if match:
        return int(match.group(1))
    else:
        return None


def crop_and_cyclic_split_from_seg(seg_directory, raw_directory, output_directory):
    """
    For each segmentation file (.seg) in seg_directory, this function:
      1. Reads the seg file (assumed to be whitespace-separated with no header).
         It assigns columns: 'mmd_id', 'mmd_id_dup', 'rec_start', 'rec_end'.
      2. Extracts the user id from the seg file name.
      3. Locates the corresponding raw TSV file for that user in raw_directory.
      4. Uses the rec_start and rec_end values (recording timestamps) to crop the raw data 
         based on its 'RecordingTimestamp' column.
      5. Removes invalid rows from the cropped data. In this example, rows are removed if:
            - Any of the required valid columns contain NaN.
            - Any of the required valid columns equal -1.
      6. Restricts the cropped data to only the valid columns.
      7. Cyclically splits the segment into 4 groups.
      8. Saves each cyclic group as a pickle file named: 
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
            
            if segment_df.empty:
                print(f"No data found for user {user_id}, mmd_id {mmd_id} between {rec_start} and {rec_end}")
                continue
            
            # Remove rows with NaN in valid columns.
            segment_df = segment_df.dropna(subset=valid_columns)
            # Remove rows where any required gaze value equals -1.
            for col in valid_columns:
                segment_df = segment_df[segment_df[col] != -1]
            
            if segment_df.empty:
                print(f"All rows invalid for user {user_id}, mmd_id {mmd_id} between {rec_start} and {rec_end}")
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


def crop_and_scanpath_from_seg(seg_directory, raw_directory, output_directory):
    """
    For each segmentation file (.seg or .se) in seg_directory, this function:
      1. Reads the seg file (assumed to be whitespace-separated with no header) with columns: 
         mmd_id, mmd_id_dup, rec_start, rec_end.
      2. Extracts a segmentation id from the file name.
      3. Finds the corresponding raw TSV file in raw_directory (assumes the raw file name contains the user id).
      4. For each row (task) in the seg file, crops the raw data based on its numeric RecordingTimestamp range.
      5. Removes rows with invalid gaze points (where any required gaze value equals -1).
      6. Computes the average gaze coordinates from left/right gaze columns.
      7. Plots a single scanpath image (scatter for fixations and line for saccades) for the entire task.
      8. Saves the image as a PNG file with a naming convention:
             {prefix}_{user_id}_{seg_id}_{mmd_id}.png
         where prefix is determined from seg_directory (e.g. "ctrl" for control).
    
    Parameters:
      seg_directory (str): Directory containing segmentation files (e.g., "control_segs").
      raw_directory (str): Directory containing raw TSV files (one per user).
      output_directory (str): Directory where output scanpath images will be saved.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Required gaze columns for scanpath generation.
    valid_gaze_columns = [
        'GazePointLeftX (ADCSpx)',
        'GazePointLeftY (ADCSpx)',
        'GazePointRightX (ADCSpx)',
        'GazePointRightY (ADCSpx)'
    ]
    
    # Determine prefix from seg_directory name.
    seg_directory_lower = seg_directory.lower()
    raw_directory_lower = raw_directory.lower()
    if 'adaptive-bar' in raw_directory_lower:
        prefix = "bar"
    elif 'adaptive-link' in raw_directory_lower:
        prefix = "link"
    elif 'control' in raw_directory_lower:
        prefix = "ctrl"
    else:
        prefix = "unknown"
    
    # List segmentation files (accepting .seg or .se files)
    seg_files = [f for f in os.listdir(seg_directory) if f.lower().endswith(('.seg', '.se'))]
    
    for seg_file in seg_files:
        seg_path = os.path.join(seg_directory, seg_file)
        try:
            # Read the seg file: assume whitespace-separated, no header.
            df_seg = pd.read_csv(seg_path, sep='\s+', header=None,
                                 names=['mmd_id', 'mmd_id_dup', 'rec_start', 'rec_end'])
        except Exception as e:
            print(f"Error reading seg file {seg_path}: {e}")
            continue
        
        # Extract segmentation id from the filename.
        seg_id = extract_seg_id(seg_file)
        if seg_id is None:
            print(f"Could not extract seg id from file {seg_file}. Skipping.")
            continue
        
        # For this example, assume each seg file represents one user; use seg id as the user id.
        user_id = seg_id
        
        # Find the corresponding raw TSV file for this user.
        raw_files = [f for f in os.listdir(raw_directory) if f.lower().endswith('.tsv') and str(user_id) in f]
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
            
            # Crop the raw data based on RecordingTimestamp range.
            segment_df = df_raw[(df_raw['RecordingTimestamp'] >= rec_start) &
                                (df_raw['RecordingTimestamp'] <= rec_end)].copy()
            if segment_df.empty:
                print(f"No data for user {user_id}, mmd_id {mmd_id} between {rec_start} and {rec_end}")
                continue
            
            # Remove rows where any required gaze column equals -1.
            for col in valid_gaze_columns:
                segment_df = segment_df[segment_df[col] != -1]
            if segment_df.empty:
                print(f"All rows invalid for user {user_id}, mmd_id {mmd_id} between {rec_start} and {rec_end}")
                continue
            
            # Compute average gaze coordinates.
            segment_df['r_GazePointX (ADCSpx)'] = (segment_df['GazePointLeftX (ADCSpx)'] +
                                                    segment_df['GazePointRightX (ADCSpx)']) / 2
            segment_df['r_GazePointY (ADCSpx)'] = (segment_df['GazePointLeftY (ADCSpx)'] +
                                                    segment_df['GazePointRightY (ADCSpx)']) / 2
            
            # Drop rows with NaN in the computed gaze points
            segment_df = segment_df.dropna(subset=['r_GazePointX (ADCSpx)', 'r_GazePointY (ADCSpx)'])
            
            # Extract the computed gaze points.
            x = segment_df['r_GazePointX (ADCSpx)'].values
            y = segment_df['r_GazePointY (ADCSpx)'].values
            if len(x) == 0 or len(y) == 0:
                print(f"No valid gaze data for user {user_id}, mmd_id {mmd_id}. Skipping task.")
                continue

            # Wrap the plotting/saving in a try/except block to catch any errors (e.g., from np.interp)
            try:
                plt.figure()
                plt.scatter(x, y, s=5)  # scatter for fixations
                plt.plot(x, y, linewidth=1)  # line connecting gaze points
                plt.axis('off')
                
                # Save the same image four times with different suffixes.
                for i in range(4):
                    output_filename = os.path.join(output_directory, f"{prefix}_{user_id}_{mmd_id}_{i}.png")
                    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
                    print(f"Saved scanpath image for user {user_id}, mmd_id {mmd_id}, cycle {i}, as {output_filename}")
                plt.close()
            except Exception as e:
                print(f"Error generating scanpath for user {user_id}, mmd_id {mmd_id}: {e}. Skipping this task.")
                plt.close()
                continue

