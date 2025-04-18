import re
import numpy as np

def parse_file(filename):
    """Read the entire file content."""
    with open(filename, 'r') as f:
        return f.read()

def extract_fold_metrics(block):
    """
    Extract test metrics from each fold.
    Expected pattern per fold:
    Test set sens. : 0.67123, spec. : 0.54000, combined: 0.60562, auc: 0.61687
    """
    pattern = r"Test set sens\. : ([\d\.]+), spec\. : ([\d\.]+), combined: ([\d\.]+), auc: ([\d\.]+)"
    matches = re.findall(pattern, block)
    fold_metrics = []
    for m in matches:
        fold_metrics.append({
            "sensitivity": float(m[0]),
            "specificity": float(m[1]),
            "combined": float(m[2]),
            "auc": float(m[3])
        })
    return fold_metrics

def compute_overall_metrics(fold_metrics):
    """Compute overall mean and standard deviation from a list of fold metrics."""
    overall = {}
    if not fold_metrics:
        return None
    for metric in ["sensitivity", "specificity", "combined", "auc"]:
        values = [fm[metric] for fm in fold_metrics]
        overall[metric] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }
    return overall

def process_section(header, block):
    """Process one cognitive ability section."""
    print(f"=== {header.strip()} ===")
    
    # Extract metrics from each fold in the section.
    fold_metrics = extract_fold_metrics(block)
    
    if fold_metrics:
        overall = compute_overall_metrics(fold_metrics)
        print("Fold-level Test Metrics:")
        for i, fm in enumerate(fold_metrics, 1):
            print(f"  Fold {i}: Sensitivity: {fm['sensitivity']:.5f}, "
                  f"Specificity: {fm['specificity']:.5f}, Combined: {fm['combined']:.5f}, "
                  f"AUC: {fm['auc']:.5f}")
        print("\nOverall CV Metrics (aggregated from folds):")
        for key, stats in overall.items():
            print(f"  {key.capitalize()} - Mean: {stats['mean']:.5f}, Std Dev: {stats['std']:.5f}")
    else:
        print("No fold-level test metrics found in this section.")
    print("\n")

filename = "vtnet_29seconds_209798.out"  

# Read and parse file
data = parse_file(filename)

# Split file into sections using header lines 
sections = re.split(r'(VTNET.*Results.*\n)', data)
if not sections[0].strip():
    sections = sections[1:]

# Process each section (header + block)
for i in range(0, len(sections), 2):
    header = sections[i]
    block = sections[i+1] if i+1 < len(sections) else ""
    process_section(header, block)
