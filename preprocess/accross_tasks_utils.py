# Preprocess Raw Tobii Exports for VTNet_att for the combined MSNV dataset (for accross tasks)
# Preprocesses without splitting into tasks 
 
import pandas as pd 
import re
import os 
import shutil
 
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
   
   # Columns to keep from each TSV file
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
   
   # Constants
   SAMPLING_RATE = 120  # Tobii T120 tracker captures at 120Hz
   MIN_TIME_SECONDS = 3   # Minimum required duration of data in seconds
   MIN_FRAMES = SAMPLING_RATE * MIN_TIME_SECONDS  # Minimum frames required
   CYCLIC_SPLITS = 4  # Number of cyclic splits
   
   # Process each file in each directory
   for dir in directories:
       output_path = os.path.join(dir, 'final_pkl_cyclic')
       if not os.path.exists(output_path):
           os.makedirs(output_path)
   
       print(f"\nProcessing directory: {dir}")
   
       for f in os.listdir(dir):
           # Process only TSV files; skip others (like existing pickle files)
           if not f.endswith('.tsv'):
               print(f"Skipping {f} (not a TSV file)")
               continue
   
           f_path = os.path.join(dir, f)
           try:
               # Read the TSV file
               df = pd.read_csv(f_path, sep='\t')
   
               # Check if the DataFrame has the minimum required rows
               if len(df) < MIN_FRAMES:
                   print(f"Skipping {f} (Less than {MIN_TIME_SECONDS}s of data)")
                   continue
   
               # Check if all valid columns exist in the DataFrame
               if not set(valid_columns).issubset(df.columns):
                   print(f"Skipping {f} (missing one or more valid columns)")
                   continue
   
               # Subset the DataFrame to only the valid columns
               df = df[valid_columns]
   
               # Cyclic Splitting and Saving as Pickle Files
               for i in range(CYCLIC_SPLITS):
                   split_df = df.iloc[i::CYCLIC_SPLITS].reset_index(drop=True)
                   output_filename = os.path.join(output_path, f.replace('.tsv', f'_{i+1}.pkl'))
                   with open(output_filename, 'wb') as handle:
                       pickle.dump(split_df, handle)
                   print(f"Saved: {output_filename}")
   
           except Exception as e:
               print(f"Error processing {f}: {e}")
   
   print("\n Preprocessing complete!")

def sort_files_into_folders(df: pd.DataFrame, source_folder: str, destination_folder: str) -> None:
    # Define label columns and folder structure
    label_columns = ['Meara_label', 'BarChartLit_label', 'VerbalWM_label']
    categories = ['high', 'low']
    file_types = {'pkl': 'pickle_files', 'png': 'images'}

    # Create base directories for each label, category, and file type
    for label in label_columns:
        for category in categories:
            for file_type_folder in file_types.values():
                folder_path = os.path.join(destination_folder, label, category, file_type_folder)
                os.makedirs(folder_path, exist_ok=True)

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        uid = row['uid']
        for label in label_columns:
            category = 'high' if row[label] == 1 else 'low'
            for file_name in os.listdir(source_folder):
                if file_name.startswith(f'{uid}_'):
                    file_extension = file_name.split('.')[-1]
                    if file_extension in file_types:
                        source_file = os.path.join(source_folder, file_name)
                        file_type_folder = file_types[file_extension]
                        destination_path = os.path.join(destination_folder, label, category, file_type_folder)
                        shutil.copy(source_file, destination_path)

    print(f'Files successfully sorted into {destination_folder}')

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
