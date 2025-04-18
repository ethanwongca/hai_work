"""
Functions to setup the task data for VTNet processing
"""

import os     
import shutil   
import pandas as pd
import pickle
import random
import numpy as np
from sklearn.model_selection import GroupKFold

# Set manual seed for reproducibility
MANUAL_SEED = 1
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)

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


def get_grouped_splits(confused_items, not_confused_items, k):
    """
    Splits data ensuring that no single user’s data appears in both training
    and testing sets using GroupKFold.

    Args:
        confused_items (list): list of file names for the confused class.
        not_confused_items (list): list of file names for the not_confused class.
        k (int): number of folds for cross-validation.

    Returns:
        tuple: (train_confused_splits, test_confused_splits,
                train_not_confused_splits, test_not_confused_splits)
    """
    train_confused_splits = []
    test_confused_splits = []
    train_not_confused_splits = []
    test_not_confused_splits = []

    # Build a groups list based on a user identifier extracted from the filename.
    # This example assumes that the filename contains the user ID at the beginning,
    # e.g. "U123_itemname.pkl". Adjust the splitting logic if your filenames differ.
    groups_confused = [fname.split('_')[0] + "_" + fname.split('_')[1] for fname in confused_items]
    groups_not_confused = [fname.split('_')[0] + "_" + fname.split('_')[1] for fname in not_confused_items]

    # Combine items and groups for both classes.
    items = confused_items + not_confused_items
    groups = groups_confused + groups_not_confused
    # Dummy labels: 0 for confused, 1 for not_confused.
    dummy_y = [0] * len(confused_items) + [1] * len(not_confused_items)

    gkf = GroupKFold(n_splits=k)
    for train_idx, test_idx in gkf.split(X=items, y=dummy_y, groups=groups):
        # Split items based on the dummy labels.
        train_confused = [items[i] for i in train_idx if dummy_y[i] == 0]
        test_confused = [items[i] for i in test_idx if dummy_y[i] == 0]
        train_not_confused = [items[i] for i in train_idx if dummy_y[i] == 1]
        test_not_confused = [items[i] for i in test_idx if dummy_y[i] == 1]

        train_confused_splits.append(train_confused)
        test_confused_splits.append(test_confused)
        train_not_confused_splits.append(train_not_confused)
        test_not_confused_splits.append(test_not_confused)

    return (train_confused_splits, test_confused_splits,
            train_not_confused_splits, test_not_confused_splits)


