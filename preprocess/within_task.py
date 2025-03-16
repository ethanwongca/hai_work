import os
import pandas as pd
import pickle

def extract_user_id(filename):
    """
    Returns the user id from the filename by capturing any digit(s) immediately before ".tsv"
    Example:
        "AdaptiveMSNV_Bar_Study_New test_IVT60_Rec msnv10.tsv" -> 10
    """
    import re
    match = re.search(r'(\d+)\.tsv$', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def cyclic_split_df(df, n=4):
    """
    Splits a DataFrame into n groups in a cyclic (round-robin) manner.
    For example, with n=4:
      - Group 0 gets rows 0,4,8,...
      - Group 1 gets rows 1,5,9,...
      - Group 2 gets rows 2,6,10,...
      - Group 3 gets rows 3,7,11,...
    """
    return [df.iloc[i::n].reset_index(drop=True) for i in range(n)]

def crop_combine_and_cyclic_split_tasks(task_csv, raw_directory, output_directory):
    """
    For each raw TSV file corresponding to a user, this function:
      1. Uses a tasks CSV (with columns: user_id, mmd_id, screen_id, task_start, task_end)
         to crop a combined segment per task (i.e. combining 'mmd' and 'questions' segments).
      2. Converts the task start/end times (which are in HH:MM:SS.sss format) to full datetime objects,
         using the date from the raw data's 'LocalTimeStamp'.
      3. Crops the raw data (which has a 'LocalTimeStamp' column) for the interval between the earliest
         start and the latest end time for that task.
      4. Restricts the DataFrame to only the valid columns.
      5. Cyclically splits the resulting segment into 4 groups.
      6. Saves each cyclic group as a separate pickle file with naming convention:
         {prefix}_{user_id}_{mmd_id}_cyclic_{i}.pkl, where prefix is determined by the raw_directory.
    
    Parameters:
      task_csv (str): Path to the CSV file with task info.
      raw_directory (str): Directory containing raw TSV files (each file assumed to be per user).
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
    
    # Determine the prefix based on the raw_directory name.
    # If the directory name includes "control", we use "ctrl".
    raw_directory_lower = raw_directory.lower()
    if 'adaptive-bar' in raw_directory_lower:
        prefix = "bar"
    elif 'adaptive-link' in raw_directory_lower:
        prefix = "link"
    elif 'control' in raw_directory_lower:
        prefix = "ctrl"
    else:
        prefix = "unknown"
    
    # Read tasks CSV
    df_tasks = pd.read_csv(task_csv)
    
    # Process each raw TSV file in the directory
    raw_files = [f for f in os.listdir(raw_directory) if f.endswith('.tsv')]
    
    for file in raw_files:
        # Extract user id from the filename
        user_id = extract_user_id(file)
        if user_id is None:
            print(f"Could not extract user id from file {file}")
            continue
        
        # Filter tasks for this user
        user_tasks = df_tasks[df_tasks['user_id'] == user_id].copy()
        if user_tasks.empty:
            print(f"No tasks found for user {user_id}")
            continue
        
        # Read the raw TSV file
        raw_path = os.path.join(raw_directory, file)
        try:
            df_raw = pd.read_csv(raw_path, delimiter='\t')
        except Exception as e:
            print(f"Error reading {raw_path}: {e}")
            continue
        
        # Convert 'LocalTimeStamp' to datetime
        df_raw['LocalTimeStamp'] = pd.to_datetime(df_raw['LocalTimeStamp'])
        # Assume tasks' times are on the same day as the raw data; extract that date
        raw_date = df_raw['LocalTimeStamp'].iloc[0].date()
        
        # Convert task_start and task_end columns to full datetime using raw_date
        user_tasks['task_start_dt'] = pd.to_datetime(user_tasks['task_start'].apply(lambda t: f"{raw_date} {t}"))
        user_tasks['task_end_dt'] = pd.to_datetime(user_tasks['task_end'].apply(lambda t: f"{raw_date} {t}"))
        
        # Group tasks by mmd_id (each task might be split into two rows: one for mmd and one for questions)
        grouped = user_tasks.groupby('mmd_id')
        for mmd_id, group in grouped:
            # Combined segment spans from the earliest start to the latest end in the group
            combined_start = group['task_start_dt'].min()
            combined_end = group['task_end_dt'].max()
            
            # Crop the raw data to only include rows within the combined segment
            segment_df = df_raw[(df_raw['LocalTimeStamp'] >= combined_start) & 
                                (df_raw['LocalTimeStamp'] <= combined_end)].copy()
            
            if segment_df.empty:
                print(f"No data found for user {user_id}, mmd_id {mmd_id} between {combined_start} and {combined_end}")
                continue
            
            # Restrict the DataFrame to only the valid columns.
            if not set(valid_columns).issubset(segment_df.columns):
                print(f"Warning: Not all valid columns found in data for user {user_id}, mmd_id {mmd_id}. Skipping.")
                continue
            
            segment_df = segment_df[valid_columns]
            
            # Perform a cyclic split into 4 groups
            cyclic_groups = cyclic_split_df(segment_df, n=4)
            
            # Save each cyclic group as a separate pickle file
            for i, cyclic_df in enumerate(cyclic_groups):
                output_filename = os.path.join(output_directory, f"{prefix}_{user_id}_{mmd_id}_{i}.pkl")
                with open(output_filename, 'wb') as f:
                    pickle.dump(cyclic_df, f)
                print(f"Saved cyclic segment {i} for user {user_id}, mmd_id {mmd_id} as {output_filename}")
              
if __name__ == "__main__":
  crop_combine_and_cyclic_split_tasks('control_timestamps.csv', 'control/raw_data', 'control_output_segments')
  crop_combine_and_cyclic_split_tasks('MSNVLink_task_time_and_end_time.csv', 'adaptive-link/raw_data', 'link_output_segments')
