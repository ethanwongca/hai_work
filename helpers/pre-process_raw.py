# Pre-Process Raw Tobii Exports for VTNet_att for the combined MSNV dataset 

import pandas as pd 
import re
import os 

def cnn_preprocess():
  directories = [
        'adaptive-bar/raw_data',
        'adaptive-link/raw_data',
        'control/raw_data'
    ]

  valid_columns = [
          'GazePointLeftX (ADCSpx)',
          'GazePointLeftY (ADCSpx)',
          'CamLeftX',
          'CamLeftY',
          'PupilLeft',
          'DistanceLeft',
          'ValidityLeft',
          'GazePointRightX (ADCSpx)',
          'GazePointRightY (ADCSpx)',
          'CamRightX',
          'CamRightY',
          'PupilRight',
          'DistanceRight',
          'ValidityRight'
      ]
  # Iterate through each directory
  for directory in directories:
      # Create the corresponding pkl directory if it doesn't exist
      pkl_dir = os.path.join(os.path.dirname(directory), 'pkl_cyclic_final')
      if not os.path.exists(pkl_dir):
          os.makedirs(pkl_dir)
      else:
          shutil.rmtree(pkl_dir)
          os.makedirs(pkl_dir)
  
      # Iterate through files in the directory
      for file_name in os.listdir(directory):
          # Check if the file is a .tsv file
          if file_name.endswith('.tsv'):
              # Construct the full file path
              file_path = os.path.join(directory, file_name)
  
              # Read the .tsv file into a DataFrame
              df = pd.read_csv(file_path, sep='\t')
  
              # Select only the valid columns
              df_selected = df[valid_columns]
  
              # Create the output file name (same name but with .pkl extension)
              output_file_name = file_name.replace('.tsv', '.pkl')
              output_file_path = os.path.join(pkl_dir, output_file_name)
  
              # Save the selected DataFrame as a .pkl file
              df_selected.to_pickle(output_file_path)
  
              print(f"Processed {file_name} and saved to {output_file_path}")


def rnn_preprocess():
  directories = [
        'adaptive-bar/raw_data',
        'adaptive-link/raw_data',
        'control/raw_data'
    ]

  valid_columns = [
          'GazePointLeftX (ADCSpx)',
          'GazePointLeftY (ADCSpx)',
          'CamLeftX',
          'CamLeftY',
          'PupilLeft',
          'DistanceLeft',
          'ValidityLeft',
          'GazePointRightX (ADCSpx)',
          'GazePointRightY (ADCSpx)',
          'CamRightX',
          'CamRightY',
          'PupilRight',
          'DistanceRight',
          'ValidityRight'
      ]
  # Iterate through each directory
  for directory in directories:
      # Create the corresponding pkl directory if it doesn't exist
      pkl_dir = os.path.join(os.path.dirname(directory), 'pkl_cyclic_final')
      if not os.path.exists(pkl_dir):
          os.makedirs(pkl_dir)
      else:
          shutil.rmtree(pkl_dir)
          os.makedirs(pkl_dir)
  
      # Iterate through files in the directory
      for file_name in os.listdir(directory):
          # Check if the file is a .tsv file
          if file_name.endswith('.tsv'):
              # Construct the full file path
              file_path = os.path.join(directory, file_name)
  
              # Read the .tsv file into a DataFrame
              df = pd.read_csv(file_path, sep='\t')
  
              # Select only the valid columns
              df_selected = df[valid_columns]
  
              # Create the output file name (same name but with .pkl extension)
              output_file_name = file_name.replace('.tsv', '.pkl')
              output_file_path = os.path.join(pkl_dir, output_file_name)
  
              # Save the selected DataFrame as a .pkl file
              df_selected.to_pickle(output_file_path)
  
              print(f"Processed {file_name} and saved to {output_file_path}")

def map_name():
  # Map the name of the .pkl files made from the previous functions
  # Mapping from folder name to the desired prefix.
  mapping = {
      "control": "ctrl",
      "adaptive_bar": "bar",
      "adaptive_link": "link"
  }
  
  # List of directories (each should be named "control", "adapt_bar", or "adapt_link")
  csv_files = ["control", "adaptive_bar", "adaptive_link"]  # adjust these paths as needed
  
  
  
  for folder in csv_files:
      in_path = os.path.join(folder, "final_pkl_cyclic")
      prefix = mapping.get(folder, folder)
      for file in os.listdir(in_path):
          # Use regex to match any characters after "Rec" until we find a digit pattern like "2_3".
          match = re.search(r"Rec\s*.*?(\d+_\d+)", file)
          if match:
              digits = match.group(1)
              new_name = f"{prefix}_{digits}.pkl"
              old_path = os.path.join(in_path, file)
              new_path = os.path.join(in_path, new_name)
              os.rename(old_path, new_path)
              print(f"Renamed {file} to {new_name}")
          else:
              print(f"Pattern not found in {file}")
