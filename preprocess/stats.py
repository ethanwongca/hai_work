'''
Computes the stats of the .out from the HPC 
'''

import re
import numpy as np

def parse_metrics(file_path):
    """
    Parses the output file to extract test set accuracy (the combined metric)
    and AUC values from lines with the expected format.
    
    Expected line format (example):
    Test set sens. : 0.52941, spec. : 0.47991, combined: 0.50466, auc: 0.48805
    
    Args:
        file_path: Path to the output file.
        
    Returns:
        Tuple of two lists: (accuracies, aucs)
    """
    accuracies = []
    aucs = []
    
    # Regex pattern to match the test set metrics.
    pattern = re.compile(
        r"Test set sens\.\s*:\s*([\d\.]+),\s*spec\.\s*:\s*([\d\.]+),\s*combined:\s*([\d\.]+),\s*auc:\s*([\d\.]+)"
    )
    
    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    # Extract the combined accuracy (group 3) and AUC (group 4)
                    accuracy = float(match.group(3))
                    auc = float(match.group(4))
                    accuracies.append(accuracy)
                    aucs.append(auc)
                except ValueError:
                    print(f"Warning: Could not convert metrics to float in line: {line.strip()}")
    return accuracies, aucs

def compute_statistics(values):
    """
    Computes and returns the mean and standard deviation of a list of numbers.
    
    Args:
        values: List of numeric values.
        
    Returns:
        A tuple (mean, std_dev).
    """
    return np.mean(values), np.std(values)

# Specify the path to your results file.
file_path = "vtnet_29_ctrl_210504.out"  # Update this path if needed

# Parse the metrics from the file.
accuracies, aucs = parse_metrics(file_path)

# Check if any valid metrics were found.
if not accuracies or not aucs:
    print("No valid test set metrics found in the file.")
else:
    mean_acc, std_acc = compute_statistics(accuracies)
    mean_auc, std_auc = compute_statistics(aucs)
    
    print("Test Set Metrics across folds:")
    print(f"Accuracy (combined): Mean = {mean_acc:.5f}, Std Dev = {std_acc:.5f}")
    print(f"AUC: Mean = {mean_auc:.5f}, Std Dev = {std_auc:.5f}")
