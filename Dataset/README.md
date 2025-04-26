# Dataset Documentation

This guide explains how to access the eye-tracking datasets for use with VTNet.

## Table of Contents
- [Access Requirements](#access-requirements)
- [Dataset Overview](#dataset-overview)
- [Accessing the Data](#accessing-the-data)
- [Data Structure](#data-structure)
- [Data Retrieval Instructions](#data-retrieval-instructions)
- [Data Preprocessing](#data-preprocessing)
- [Required Data Columns](#required-data-columns)
- [Troubleshooting](#troubleshooting)

## Access Requirements

Before beginning, ensure you have:

- **SSH/SFTP access** to UBC lab server (`remote.cs.ubc.ca`)
- **UBC credentials** with appropriate permissions for `/ubc/cs/research/conati/ATUAV/`
- **Local storage** sufficient for raw eye-tracking data (approximately 10GB)

## Dataset Overview

The dataset contains eye-tracking data from three distinct experiments:

| Dataset Name | Description | Location |
|--------------|-------------|----------|
| **Adapt Bar** | Adaptive bar chart visualization study | MSNV UBC Study 2 - Adaptive |
| **Adapt Link** | Adaptive network visualization study | MSNV UBC Study 2 - Adaptive |
| **Control** | Non-adaptive visualization baseline | MSNV UBC Study 1 |

Each dataset includes:
- Raw Tobii eye-tracking exports (120 Hz sampling rate)
- Segmentation files (`.seg`) marking task boundaries
- Cognitive ability labels for classification tasks

## Accessing the Data

### Step 1: Connect to the UBC Lab Server

```bash
ssh <username>@remote.cs.ubc.ca
```

### Step 2: Navigate to Project Root

```bash
cd /ubc/cs/research/conati/ATUAV/
```

## Data Structure

### Adaptive Datasets Location (Bar & Link)

```
/ubc/cs/research/conati/ATUAV/
└── User Studies/
    └── MSNV UBC Study 2 - Adaptive/
        └── Processing and Analysis Scripts/
            └── Alireza - Combining Study Datasets/
                └── Data/
                    ├── Adapt Bar/               
                    │   ├── bar_segs/            # Segmentation files
                    │   └── MSNV Bar Tobii Exports/   # Raw eye-tracking data
                    │
                    └── Adapt Link/             
                        ├── link_segs/           # Segmentation files
                        └── MSNV Link Tobii Exports/   # Raw eye-tracking data
```

### Control Dataset Location

```
/ubc/cs/research/conati/ATUAV/
└── User Studies/
    └── MSNV UBC Study 1/
        └── Raw Experiment Data/
            └── Tobii Export/ 
                └── Segs/      # Segmentation files
                # Raw eye-tracking data is in parent directory
```

### Cognitive Ability Labels

```
/ubc/cs/research/conati/ATUAV/
└── User Studies/
    └── MSNV UBC Study 2 - Adaptive/
        └── Processing and Analysis Scripts/
            └── Alireza - Combining Study Datasets/
                └── Data/
                    └── Labels/
                        ├── all_across_labels.csv    # For across-task classification
                        └── all_within_labels.csv    # For within-task classification
```

## Data Retrieval Instructions

### Using SFTP

1. **Connect to server:**
   ```bash
   sftp <username>@remote.cs.ubc.ca
   ```

2. **Navigate to the dataset directory:**
   ```bash
   cd '/ubc/cs/research/conati/ATUAV/User Studies/MSNV UBC Study 2 - Adaptive/Processing and Analysis Scripts/Alireza - Combining Study Datasets/Data'
   ```

3. **Download Adapt Bar dataset:**
   ```bash
   # Create local directories first
   !mkdir -p "local_data/Adapt_Bar/seg_files" "local_data/Adapt_Bar/tobii_exports"
   
   # Download segmentation files
   get -r 'Adapt Bar/bar_segs/*' "local_data/Adapt_Bar/seg_files/"
   
   # Download eye-tracking data
   get -r 'Adapt Bar/MSNV Bar Tobii Exports/*' "local_data/Adapt_Bar/tobii_exports/"
   ```

4. **Download Adapt Link dataset:**
   ```bash
   !mkdir -p "local_data/Adapt_Link/seg_files" "local_data/Adapt_Link/tobii_exports"
   get -r 'Adapt Link/link_segs/*' "local_data/Adapt_Link/seg_files/"
   get -r 'Adapt Link/MSNV Link Tobii Exports/*' "local_data/Adapt_Link/tobii_exports/"
   ```

5. **Download cognitive ability labels:**
   ```bash
   !mkdir -p "local_data/Labels"
   get -r 'Labels/all_across_labels.csv' "local_data/Labels/"
   get -r 'Labels/all_within_labels.csv' "local_data/Labels/"
   ```

6. **Navigate to Control dataset:**
   ```bash
   cd '/ubc/cs/research/conati/ATUAV/User Studies/MSNV UBC Study 1/Raw Experiment Data/Tobii Export'
   ```

7. **Download Control dataset:**
   ```bash
   !mkdir -p "local_data/Control/seg_files" "local_data/Control/tobii_exports"
   get -r 'Segs/*' "local_data/Control/seg_files/"
   get -r '*.csv' "local_data/Control/tobii_exports/"
   ```

## Data Preprocessing

After downloading the raw data, use the preprocessing scripts in this repository to prepare it for VTNet:

```bash
# Navigate to the preprocessing directory
cd preprocess/
```

For custom preprocessing, ensure your data includes all the required columns listed below.

## Required Data Columns

VTNet requires the following columns from the raw Tobii data for its RNN component:

| Category               | Column Name                          | Description                                                                                   |
|------------------------|--------------------------------------|-----------------------------------------------------------------------------------------------|
| **Gaze Point (ADCSpx)**    | GazePointLeftX (ADCSpx)              | X‑coordinate of the left eye gaze in device‑centered screen pixels                            |
|                        | GazePointLeftY (ADCSpx)              | Y‑coordinate of the left eye gaze in device‑centered screen pixels                            |
|                        | GazePointRightX (ADCSpx)             | X‑coordinate of the right eye gaze in device‑centered screen pixels                           |
|                        | GazePointRightY (ADCSpx)             | Y‑coordinate of the right eye gaze in device‑centered screen pixels                           |
|                        | GazePointX (ADCSpx)                  | Averaged X‑coordinate of both eyes in device‑centered screen pixels                           |
|                        | GazePointY (ADCSpx)                  | Averaged Y‑coordinate of both eyes in device‑centered screen pixels                           |
| **Gaze Point (MCSpx)**     | GazePointX (MCSpx)                   | Averaged X‑coordinate of both eyes in monitor‑centered screen pixels                          |
|                        | GazePointY (MCSpx)                   | Averaged Y‑coordinate of both eyes in monitor‑centered screen pixels                          |
| **Gaze Point (ADCSmm)**    | GazePointLeftX (ADCSmm)              | X‑coordinate of the left eye gaze in device‑centered millimeters                              |
|                        | GazePointLeftY (ADCSmm)              | Y‑coordinate of the left eye gaze in device‑centered millimeters                              |
|                        | GazePointRightX (ADCSmm)             | X‑coordinate of the right eye gaze in device‑centered millimeters                             |
|                        | GazePointRightY (ADCSmm)             | Y‑coordinate of the right eye gaze in device‑centered millimeters                             |
| **Distance**               | DistanceLeft                         | Distance from left eye to screen (mm)                                                         |
|                        | DistanceRight                        | Distance from right eye to screen (mm)                                                        |
| **Pupil**                  | PupilLeft                            | Diameter of left pupil (pixels)                                                               |
|                        | PupilRight                           | Diameter of right pupil (pixels)                                                              |
| **Fixation Point**         | FixationPointX (MCSpx)               | X‑coordinate of fixation in monitor‑centered screen pixels (estimated by Tobii's fixation filter) |
|                        | FixationPointY (MCSpx)               | Y‑coordinate of fixation in monitor‑centered screen pixels (estimated by Tobii's fixation filter) |

## Troubleshooting

### Common Issues

1. **Permission denied errors**
   - Make sure that you are on UBC's VPN or on UBC Secure 


2. **Data format issues**
   - If Tobii data format seems inconsistent, refer to the data dictionary columns above
   - Some files may require additional cleaning steps before processing


