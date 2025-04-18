"""
Utility functions for analyzing and visualizing gaze validity data. Does EDA with the data. 
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import pandas as pd

def validity_chart(validity_percentages: List[float], time_interval: int) -> None:
    """Plot a bar chart of task validity percentages across thresholds.

    The chart displays the percentage of tasks with valid gaze data at various
    validity thresholds.

    Args:
        validity_percentages: A list of validity percentages.
        time_interval: Time interval in seconds for the task.
    """
    thresholds = np.arange(0.5, 1.05, 0.05)
    y_thresholds = np.arange(0, 110, 10)
    proportions = [
        sum(v >= thresh for v in validity_percentages) / len(validity_percentages) * 100
        for thresh in thresholds
    ]

    plt.figure()
    plt.bar(thresholds, proportions, width=0.04, edgecolor='black', align='center')
    plt.xlabel("Validity Threshold")
    plt.ylabel("% of Tasks with Valid Gaze Data")
    plt.title(f"Validity of Tasks Across Thresholds at {time_interval} Seconds")
    plt.yticks(y_thresholds)
    plt.xticks(thresholds)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def get_validity_percentages(directory: str, time_interval: int) -> List[float]:
    """Compute validity percentages for pickle files in a directory.

    Iterates over all pickle files in the specified directory and calculates
    the validity percentage for each file over the given time interval.

    Args:
        directory: Path to the directory containing pickle files.
        time_interval: Time interval in seconds.

    Returns:
        A list of validity proportions.
    """
    validity_percentages = []
    for file in os.listdir(directory):
        if file.endswith('.pkl'):
            file_path = os.path.join(directory, file)
            percentage_valid = get_proportion_validity_with_time(file_path, time_interval)
            validity_percentages.append(percentage_valid)
    return validity_percentages


def get_time_chart(directory: str, time_interval: int) -> None:
    """Plot a chart comparing the number of files meeting the required time.

    Iterates over all pickle files in the specified directory and checks whether
    each file meets the required time interval. It then plots a bar chart showing
    the number of valid versus invalid files.

    Args:
        directory: Path to the directory containing pickle files.
        time_interval: Time interval in seconds.
    """
    validity_flags = []
    for file in os.listdir(directory):
        if file.endswith('.pkl'):
            file_path = os.path.join(directory, file)
            meets_time = check_meet_task_time(file_path, time_interval)
            validity_flags.append(meets_time)

    valid_count = sum(validity_flags)
    invalid_count = len(validity_flags) - valid_count

    plt.figure()
    plt.bar(
        [f"At least {time_interval} Seconds", f"Under {time_interval} Seconds"],
        [valid_count, invalid_count]
    )
    plt.xlabel("File Validity")
    plt.ylabel("Number of Tasks")
    plt.title("Number of Valid vs. Invalid Files")
    plt.show()


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


def check_meet_task_time(file_path: str, time_interval: int) -> bool:
    """Determine if the file contains sufficient data for the given time interval.

    Args:
        file_path: Path to the pickle file containing the DataFrame.
        time_interval: Duration in seconds to check.

    Returns:
        True if the data meets or exceeds the expected number of samples,
        False otherwise.
    """
    SAMPLING_RATE = 120
    measure_interval = SAMPLING_RATE * time_interval
    df = pd.read_pickle(file_path)
    return measure_interval <= df.shape[0]

def task_count_chart_validity(directory_path:str, validity_percentage:float, time:int) -> None:
    """
    Builds a histogram that is all the sequence lengths of tasks in a specific validity threshold given a specific time window

    Args:
        directory_path: Path to the pickle file containing the DataFrame.
        validity_percentage: % of tasks, that is valid in that specific time window
        time_: Duration in seconds for which the data is expected.

    Returns:
        None: (Plots histogram)
    """
    counts = []
    for file in os.listdir(directory_path):
        if file.endswith('.pkl'):
            # Ex. if we have like 0.9 validiy over time threshold 29 seconds > 0.75 <determines heights>
            if get_proportion_validity_with_time(os.path.join(directory_path, file), time) >= validity_percentage:
                df = pd.read_pickle(os.path.join(directory_path, file))
                df_subset = df.iloc[:time*120]
                mask = (df_subset < 0) | (df_subset.isna())
                invalid_count = mask.all(axis=1).sum()
                counts.append((file, len(df_subset) - invalid_count))
            
    counts.sort(key=lambda x:x[1])

    plt.figure()
    plt.bar([x[0] for x in counts], [x[1] for x in counts], width=0.04, edgecolor='olivedrab', align='center')
    plt.xlabel("Tasks")
    plt.ylabel("Valid Sequence Lengths")
    plt.title(f"Valid Sequence Lengths of Tasks at {time} Seconds Window at {validity_percentage} Threshold")
    plt.xticks([])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def task_count_chart_validity_time(directory_path:str, validity_percentage:float, time:int, time_cutoff:int) -> None:
    """
    Builds a histogram that shows that number of tasks under a specific time-cutoff given a specific validity threshold

    Args:
        directory_path: Path to the pickle file containing the DataFrame.
        validity_percentage: % of tasks, that is valid in that specific time window
        time: Duration in seconds for which the data is expected.
        time_cutoff: time threshold we are doing eda on

    Returns:
        None: (Plots histogram)
    """
    counts = []
    for file in os.listdir(directory_path):
        if file.endswith('.pkl'):
            # Ex. if we have like 0.9 validiy over time threshold 29 seconds > 0.75 <determines heights>
            if get_proportion_validity_with_time(os.path.join(directory_path, file), time) >= validity_percentage:
                df = pd.read_pickle(os.path.join(directory_path, file))
                df_subset = df.iloc[:time*120]
                mask = (df_subset < 0) | (df_subset.isna())
                invalid_count = mask.all(axis=1).sum()
                counts.append((file, len(df_subset) - invalid_count))

    counts.sort(key=lambda x:x[1])

    sequence_lengths = [x[1] for x in counts if x[1] <= 120 * time_cutoff]
    plt.hist(sequence_lengths, bins=30, edgecolor='black')
    plt.xlabel("Valid Sequence Lengths")
    plt.ylabel("Number of Tasks")
    plt.title(f"Distribution of Valid Sequence Lengths at {validity_percentage} Threshold Shorter than {time_cutoff} Seconds")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
