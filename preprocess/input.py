"""
Prepares the RNN and CNN input for VTNet

Cyclic Splitting and prunes data under a specific validity threshold and time threshold. Along with building the corresponding scanpaths
"""

import os
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import shutil

def cyclic_split_df(df, n=4):
    """
    Splits a DataFrame into n groups in a cyclic (round-robin) manner.
    For example, with n=4:
      - Group 0 gets rows 0, 4, 8, ...
      - Group 1 gets rows 1, 5, 9, ...
      - etc.

    Returns a list of dataframes.
    """
    return [df.iloc[i::n].reset_index(drop=True) for i in range(n)]

def clean_df(df):
    """
    Clean the DataFrame to remove rows where all features are negatives or NaN.
    """
    ignore = (df < 0) | (df.isna())
    # mask is True for rows where all entries are either negative or NaN.
    mask = ignore.all(axis=1)
    return df[~mask]

def get_proportion_validity_with_time(file_path: str, time_interval: int) -> float:
    """Calculate the percentage of valid gaze data over a given time interval.

    A row is considered invalid if all its features are negative or NaN.

    Args:
        file_path: Path to the pickle file containing the DataFrame.
        time_interval: Duration in seconds for which the data is expected.

    Returns:
        The proportion (between 0 and 1) of valid data sequences.
    """
    SAMPLING_RATE = 120
    expected_rows = SAMPLING_RATE * time_interval

    df = pd.read_pickle(file_path)
    df_used = df.iloc[:expected_rows]

    mask = (df_used < 0) | (df_used.isna())
    invalid_count = mask.all(axis=1).sum()
    valid_count = len(df_used) - invalid_count
    if len(df_used) == 0:
        return 0.0  
    return valid_count / len(df_used)

def cyclic_split_time(time_interval: int, input_directory: str, output_directory: str, validity_cutoff: float = 0.75, time_cutoff: int = 5) -> None:
    """
    Processes pickle files in the input_directory by verifying that they have at least 75% valid data.
    If valid, each DataFrame is cleaned, cyclically split, and saved in the output_directory.
    Files that don't meet the criteria are skipped.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    file_count = 0
    for file in os.listdir(input_directory):
        if file.endswith('.pkl'):
            full_path = os.path.join(input_directory, file)
            if get_proportion_validity_with_time(full_path, time_interval) >= validity_cutoff:
                df = pd.read_pickle(full_path)
                # Limit DataFrame to the specific time interval for cyclic splitting 
                df = df.head(120 * time_interval)
                # Clean invalid rows
                df = clean_df(df)
                # Skips any data points less than time_cutoff seconds (120 is sampling rate )
                if len(df) < (120 * time_cutoff):
                    file_count += 1
                    continue 
                # Cyclic Split 
                dfs = cyclic_split_df(df)
                for i, split in enumerate(dfs):
                    output_file = os.path.join(output_directory, file[:-4] + f"_{i}.pkl")
                    with open(output_file, 'wb') as f:
                        pickle.dump(split, f)
                    print(f"Saved {output_file}")
            else:
                file_count += 1
                
    print(f"Number Invalid Tasks: {file_count}")

def scanpath_from_pickle(time_interval: int, input_directory: str, output_directory: str, validity_cutoff: float = 0.75, time_cutoff: int = 5):
    """
    For each pickle file in pickle_directory, this function:
      1. Loads the pickle file (assumed to be a preprocessed DataFrame containing gaze data).
      2. Removes rows with invalid gaze points (i.e. any required gaze column equals -1).
      3. Computes the average gaze coordinates from left and right gaze columns.
      4. Plots a single scanpath image (scatter for fixations and line for saccades) for the entire DataFrame.
      5. Saves the image as a PNG file using the pickle file’s base name.
    
    Parameters:
      pickle_directory (str): Directory containing pickle files.
      output_directory (str): Directory where output scanpath images will be saved.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    valid_gaze_columns = [
        'GazePointLeftX (ADCSpx)',
        'GazePointLeftY (ADCSpx)',
        'GazePointRightX (ADCSpx)',
        'GazePointRightY (ADCSpx)'
    ]
    file_count = 0
    for file in os.listdir(input_directory):
        if file.endswith('.pkl'):
            full_path = os.path.join(input_directory, file)
            if get_proportion_validity_with_time(full_path, time_interval) >= validity_cutoff:
                df = pd.read_pickle(full_path)
                # Limit DataFrame to the specific time interval for cyclic splitting 
                df = df.head(120 * time_interval)
                # Clean invalid rows
                df = clean_df(df)
                if len(df) < (120 * time_cutoff):
                    file_count += 1
                    continue 

                #Compute 
                df['r_GazePointX (ADCSpx)'] = (df['GazePointLeftX (ADCSpx)'] + df['GazePointRightX (ADCSpx)']) / 2
                df['r_GazePointY (ADCSpx)'] = (df['GazePointLeftY (ADCSpx)'] + df['GazePointRightY (ADCSpx)']) / 2

                df = df.dropna(subset=['r_GazePointX (ADCSpx)', 'r_GazePointY (ADCSpx)'])

                x = df['r_GazePointX (ADCSpx)'].values
                y = df['r_GazePointY (ADCSpx)'].values

                try:
                    plt.figure()
                    plt.scatter(x, y, s=5)  # Adjust marker size as needed.
                    plt.plot(x, y, linewidth=1)
                    plt.axis('off')  # Hide axis for a cleaner image.
                    
                    base_name = os.path.splitext(file)[0]
                    for i in range(4):
                        output_filename = os.path.join(output_directory, f"{base_name}_{i}.png")
                        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
                        print(f"Saved scanpath image for {file} as {output_filename}")
                    
                    plt.close()
                except Exception as e:
                    print(f"Error generating scanpath for {file}: {e}")
                    plt.close()
    

