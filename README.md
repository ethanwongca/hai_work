# Human Artificial Intelligence Interaction <img height=50 width=40 src="https://github.com/user-attachments/assets/235fba2a-59e6-4092-8068-b6306fb6e993">

This repository contains all the code developed and documentation written during my directed studies on Human-AI Interaction. The code is organized into several directories, each addressing different aspects of the project.

## Directory Overview

### 1. `preprocessing`
This folder includes all the functions and scripts used to transform raw Tobii data, sampling at 120 Hz, into the necessary inputs for VTNet. It covers:
- **Data Preprocessing:** Routines for cleaning and formatting raw data.
- **Exploratory Data Analysis (EDA):** Functions for visualizing and understanding the data.
- **Dataset Construction:** Scripts to build the within-task dataset.
- **Utility Functions:** Including scanpath extraction and cyclic splitting functions.

### 2. `hpc`
This folder contains resources and **documentation** for high-performance computing, including:
- **Slurm Template:** A template for running jobs on the ubc_ml cluster.
- **Statistics Script:** A script to compute the standard deviation of the AUC (Area Under the Curve) and accuracy metrics from VTNet's output file.

### 3. `VTNet_within_tasks`
This folder holds various implementations of VTNet for the within-task dataset (14 seconds, 29 seconds, and sequence length at 1000).  
**Note:** The `st_pickle_loader` function’s `max_length` parameter is critical—it adjusts VTNet for different sequence lengths (e.g., 1000 to 3000).

### 4. `VTNet_across_tasks`
This folder contains the VTNet implementations for the across-task dataset. 

### 5. `Dataset`
This folder contains documentation to find the proper files within the lab repository.

## VTNet Suggested Directory Structure Setup (HPC)
Although there are many possible directory structure setups that can be used for VTNet here is the structure I found most effective and organized:

```
VTNet/
├── dataset/                 # Dataset pre-processed via our functions in `preprocessing`        
├── utils.py                 # Utility functions used in VTNet
├── vislit.pickle            # Pickle file for grouped CV in classifying visual literacy
├── readp.pickle             # Pickle file for grouped CV in classifying reading proficiency
├── verbalwm.pickle          # Pickle file for grouped CV in classifying verbal working memory 
└── vtenv/                   # vtenv environments, one per new length cut-off (ex. vtenv14)
    ├── VisLit/
    │   ├── VTNet.py         # VTNet modified to be trained on VisLit 
    │   └── run.sh           # Bash script to execute VTNet for Visual Literacies tasks
    ├── ReadP/
    │   ├── VTNet.py         # VTNet modified to be trained on ReadP
    │   └── run.sh           # Bash script to execute VTNet for Reading Proficiency tasks
    └── VerbalWM/
        ├── VTNet.py         # VTNet modified to be trained on VerbalWM
        └── run.sh           # Bash script to execute VTNet for Verbal Working Memory tasks
```

## Environment Setup 
To setup up the necessary dependencies to use VTNet use the following commands
### Clone the Repository 
```
git clone https://github.com/ethanwongca/hai_work
```
### Go into the Cloned Repository 
```
cd hai_work 
```
### Build the Conda Enviornment 
```
conda env create -f environment.yml
```
