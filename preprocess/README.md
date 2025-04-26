# VTNet Pre-Processing Pipeline

## Overview

This documentation covers the pre-processing pipeline designed for preparing eye-tracking data to train VTNet models. 

The pipeline consists of four primary modules that transform raw Tobii eye-tracker data into properly formatted, validated, and augmented datasets ready for model training.

## Pipeline Modules

### 1. Task Separation (`tasks.py`)

**Purpose:** Segments raw Tobii eye-tracking data into individual task datasets

**Key Functionality:**
- Parses raw eye-tracking data from Tobii exports
- Separates data into 14 distinct cognitive tasks
- Cleans eye tracking data
- Filters out invalid tracking samples
- Prepares task-specific datasets for validity analysis

**Output:** Individual and cleaned CSV files for each participant-task combination

### 2. Validity Analysis (`validity.py`)

**Purpose:** Assesses data quality and ensures sufficient valid samples across tasks

**Key Functionality:**
- Calculates validity metrics for each task and participant
- Generates validity visualization charts (as shown below)
- Computes sequence length statistics across the dataset
- Identifies tasks meeting quality thresholds
- Flags problematic recordings for exclusion

**Key Metrics:**
- Percent of valid samples per participant-task
- Distribution of valid sequence lengths

**Output:** Statistical reports and visualization charts similar to those published in *Further Results on Predicting Cognitive Abilities for Adaptive Visualizations* (Conati et al. 2017)

![Validity Chart Example](https://github.com/user-attachments/assets/8fc09d90-32a0-46a0-9adc-f85d5528a4d2)

### 3. Data Augmentation (`input.py`)

**Purpose:** Expands dataset size and generates visual scan path representations

**Key Functionality:**
- Implements cyclic splitting to quadruple data quantity
- Generates scanpath visualizations for CNN input
- Takes a subset of features of the temporal sequences for RNN input
- Filters tasks based on sequence length thresholds

**Augmentation Process:**
1. Data validation against thresholds
2. Cyclic splitting of temporal sequences
3. Scanpath image generation

**Output:** 
- Visual scanpath images (.png)
- Processed time-series data (.pickle)

### 4. Training Setup (`setup.py`)

**Purpose:** Organizes processed data for model training with cross-validation

**Key Functionality:**
- Structures data into VTNet-compatible directory hierarchy
- Implements 10-fold grouped cross-validation splits
- Separates data by task and difficulty level
- Ensures participant data integrity across splits
- Generates configuration files for VTNet training

**Data Organization:**
- Task-specific subdirectories
- Difficulty-level separation (high/low)
- Train/validation/test splits
- Participant grouping for valid cross-validation

**Output:**
- Complete directory structure for VTNet training
- Cross-validation split files (.pickle)

## Execution Workflow

For complete pipeline execution, follow this workflow:

1. **Task Separation:** tasks.py

2. **Validity Analysis:** validity.py 

3. **Data Augmentation:** input.py
4. **Training Setup:** steup.py

## Configuration Options

Each module supports various configuration parameters to customize processing:

| Parameter | Description | Default | Module |
|-----------|-------------|---------|--------|
| `validity_threshold` | Minimum percentage of valid samples required | 0.80 | validity.py |
| `sequence_length` | Time duration in seconds for analysis | 29 | input.py |
| `sampling_rate` | Eye-tracker sampling frequency (Hz) | 120 | input.py |
| `cyclic_factor` | Number of cyclic splits to generate | 4 | input.py |
| `cv_folds` | Number of cross-validation folds | 10 | setup.py |
| `random_seed` | Seed for reproducible splitting | 42 | setup.py |

## Integration with VTNet

After completing the pre-processing pipeline, the output directory structure will be ready for VTNet training:

```
vtnet_data/
├── TaskName_label/
│   ├── high/
│   │   └── images/
│   │       └── participant_trial_0.png
│   └── low/
│       └── images/
│           └── participant_trial_0.png
├── TaskName.pickle     # Processed time-series data
└── TaskName_cv.pickle  # Cross-validation splits
```

This structure directly integrates with VTNet's data loading mechanisms as described in the VTNet documentation.

## Extended Functionality

All modules include additional utilities for:
- Data visualization and exploration
- Quality control and validation
- Batch processing multiple datasets
- Generating detailed statistics and reports

## Documentation

All functions include comprehensive docstrings specifying:
- Parameter descriptions and types
- Return value details
- Usage examples
- Edge case handling

The files follow Google's Style Guide. Please look at the code for further information. 

## References

For methodological details on validity analysis and visualization techniques, refer to:
1. *Further Results on Predicting Cognitive Abilities for Adaptive Visualizations* (Conati et al. 2017)
2. *Classification of Alzheimer's Disease with Deep Learning on Eye-tracking Data*
