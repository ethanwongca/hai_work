# 5. Dataset

This folder explains how to pull the raw MSNV UBC Study 2 data from the lab server and prepare it for VTNet.

---

## 5.1 Prerequisites

- **SSH access** to UBC lab server (`remote.cs.ubc.ca`)  
- **UBC credentials** with read permissions on `/ubc/cs/research/conati/ATUAV/`  

---

## 5.2 Accessing the Lab Server

```bash
ssh <username>@remote.cs.ubc.ca
cd /ubc/cs/research/conati/ATUAV/
```
---

## 5.3 Locating the MSNV Data
On the server, the raw files live here:
```
/ubc/cs/research/conati/ATUAV/
└── User Studies/
    └── MSNV UBC Study 2 - Adaptive/
        └── Processing and Analysis Scripts/
            └── Alireza - Combining Study Datasets/
                └── Data/
                    └── Adapt Bar/Adapt Link/Control # Each folder is a unique dataset
                      ├── bar_segs/                 # *.seg files for each task
                      └── MSNV <Bar, Link, Control> Tobii Exports/   # 120 Hz Tobii CSV exports
```
## Copying Files Locally
```
sftp <username>@remote.cs.ubc.ca
cd '/ubc/cs/research/conati/ATUAV/MSNV UBC Study 2 - Adaptive/Processing and Analysis Scripts/Alireza - Combining Study Datasets/Data'
get '<Adapt Bar, Adapt Link, or Control>/bar_segs/*'
get '<Adapt Bar, Adapt Link, or Control>/MSNV <Bar, Link, Control> Tobii Exports/*'
```

## 5.4 Running Preprocessing 
Now that your raw data is local, run the functions within the preprocessing directory in this repository to clean the data for VTNet
