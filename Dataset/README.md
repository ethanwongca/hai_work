# 1. Dataset

This folder explains how to pull the raw MSNV UBC Study 2 data from the lab server and prepare it for VTNet.


## 1.1 Prerequisites

- **SSH access** to UBC lab server (`remote.cs.ubc.ca`)  
- **UBC credentials** with read permissions on `/ubc/cs/research/conati/ATUAV/`  


## 1.2 Accessing the Lab Server
Log in and navigate to the project root:
```bash
ssh <username>@remote.cs.ubc.ca
cd /ubc/cs/research/conati/ATUAV/
```

## 1.3 Locating the MSNV Data
To find the segmentation files and raw tobii exports for the adaptive and link dataset:
```
/ubc/cs/research/conati/ATUAV/
└── User Studies/
    └── MSNV UBC Study 2 - Adaptive/
        └── Processing and Analysis Scripts/
            └── Alireza - Combining Study Datasets/
                └── Data/
                    ├── Adapt Bar/               # Dataset “Adapt Bar”
                    │   ├── bar_segs/            # *.seg files for each task
                    │   └── MSNV Bar Tobii Exports/   # 120 Hz Tobii CSV exports
                    └── Adapt Link/             # Dataset “Adapt Link”
                        ├── link_segs/
                        └── MSNV Link Tobii Exports/

```
To find the segmentation files and raw tobii exports for the control dataset:
```
/ubc/cs/research/conati/ATUAV/
└── User Studies/
    └── MSNV UBC Study 1/
        └── Raw Experiment Data/
            └── Tobii Export/ # Folder contains Dataset "Control" and .seg file folder
                └── Segs/ # *.seg files for each task
```
To find the classification of long-term cognitive ability per task:
```
/ubc/cs/research/conati/ATUAV/
└── User Studies/
    └── MSNV UBC Study 2 - Adaptive/
        └── Processing and Analysis Scripts/
            └── Alireza - Combining Study Datasets/
                └── Data/
                    └── Labels/
                        ├── all_across_labels.csv
                        └── all_within_labels.csv
```

## Copying Files Locally
Use `sftp` to download both the segment files and Tobii CSVs:
```
sftp <username>@remote.cs.ubc.ca
cd '/ubc/cs/research/conati/ATUAV/MSNV UBC Study 2 - Adaptive/Processing and Analysis Scripts/Alireza - Combining Study Datasets/Data'
get -r 'Adapt Bar/bar_segs/*'
get -r 'Adapt Bar/MSNV Bar Tobii Exports/*'
# (repeat for Adapt Link and Control)
```

## 1.4 Running Preprocessing 
Now that your raw data is local, run the functions within the preprocessing directory in this repository to clean the data for VTNet. <br/>

You can also do your own preprocessing but note that the RNN component for VTnet takes these columns from the raw Tobii data. 
## 1.5 Running Preprocessing

Once the raw files are local, you can:

- **Use our scripts** in `preprocessing/` to clean and format data for VTNet.  
- **Or preprocess on your own** make sure your output includes the following columns from the raw tobii data for the RNN component:

| Category               | Column Name                          | Description                                                                                   |
|------------------------|--------------------------------------|-----------------------------------------------------------------------------------------------|
| Gaze Point (ADCSpx)    | GazePointLeftX (ADCSpx)              | X‑coordinate of the left eye gaze in device‑centered screen pixels                            |
|                        | GazePointLeftY (ADCSpx)              | Y‑coordinate of the left eye gaze in device‑centered screen pixels                            |
|                        | GazePointRightX (ADCSpx)             | X‑coordinate of the right eye gaze in device‑centered screen pixels                           |
|                        | GazePointRightY (ADCSpx)             | Y‑coordinate of the right eye gaze in device‑centered screen pixels                           |
|                        | GazePointX (ADCSpx)                  | Averaged X‑coordinate of both eyes in device‑centered screen pixels                           |
|                        | GazePointY (ADCSpx)                  | Averaged Y‑coordinate of both eyes in device‑centered screen pixels                           |
| Gaze Point (MCSpx)     | GazePointX (MCSpx)                   | Averaged X‑coordinate of both eyes in monitor‑centered screen pixels                          |
|                        | GazePointY (MCSpx)                   | Averaged Y‑coordinate of both eyes in monitor‑centered screen pixels                          |
| Gaze Point (ADCSmm)    | GazePointLeftX (ADCSmm)              | X‑coordinate of the left eye gaze in device‑centered millimeters                              |
|                        | GazePointLeftY (ADCSmm)              | Y‑coordinate of the left eye gaze in device‑centered millimeters                              |
|                        | GazePointRightX (ADCSmm)             | X‑coordinate of the right eye gaze in device‑centered millimeters                             |
|                        | GazePointRightY (ADCSmm)             | Y‑coordinate of the right eye gaze in device‑centered millimeters                             |
| Distance               | DistanceLeft                         | Distance from left eye to screen (mm)                                                         |
|                        | DistanceRight                        | Distance from right eye to screen (mm)                                                        |
| Pupil                  | PupilLeft                            | Diameter of left pupil (pixels)                                                               |
|                        | PupilRight                           | Diameter of right pupil (pixels)                                                              |
| Fixation Point         | FixationPointX (MCSpx)               | X‑coordinate of fixation in monitor‑centered screen pixels (estimated by Tobii’s fixation filter) |
|                        | FixationPointY (MCSpx)               | Y‑coordinate of fixation in monitor‑centered screen pixels (estimated by Tobii’s fixation filter) |

