# Human Artificial Intelligence Interaction <img height=50 width=50 src="https://github.com/user-attachments/assets/a47c4f55-2fe8-458c-b356-2b6b61e12008">


This repository contains all the code developed and documentation written during my directed studies on Human-AI Interaction. The code is organized into several directories, each addressing different aspects of the project.

## Directory Overview

### 1. `preprocessing`
This [folder](./preprocess/) includes all the functions and scripts used to transform raw Tobii data, sampling at 120 Hz, into the necessary inputs for VTNet. It covers:
- **Data Preprocessing:** Routines for cleaning and formatting raw data.
- **Exploratory Data Analysis (EDA):** Functions for visualizing and understanding the data.
- **Dataset Construction:** Scripts to build the within-task dataset.
- **Utility Functions:** Including scanpath extraction and cyclic splitting functions.

### 2. `hpc`
This [folder](./hpc/) contains resources and **documentation** for high-performance computing, including:
- **Slurm Template:** A template for running jobs on the `ubc_ml` cluster.
- **Statistics Script:** A script to compute the standard deviation of the AUC (Area Under the Curve) and accuracy metrics from VTNet's output file.

### 3. `VTNet_within_tasks`
This [folder](./VTNet_within_tasks/) holds various implementations of VTNet for the within-task dataset (14 seconds, 29 seconds, and sequence length at 1000).  
**Note:** The `st_pickle_loader` function’s `max_length` parameter is critical—it adjusts VTNet for different sequence lengths (e.g., 1000 to 3000).

### 4. `VTNet_across_tasks`
This [folder](./VTNet_across_tasks/) contains the VTNet implementations for the across-task dataset. 

### 5. `Dataset`
This [folder](./Dataset/) contains **documentation** to find the proper files within the lab repository.

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

## References
These papers are good to read to understand VTNet: 
- Barral, O., Lallé, S., Guz, G., Iranpour, A., & Conati, C. (2020). Eye‑tracking to predict user cognitive abilities and performance for user‑adaptive narrative visualizations. *In Proceedings of the 2020 International Conference on Multimodal Interaction* (pp. 163–173). Association for Computing Machinery. https://doi.org/10.1145/3382507.3418884

- Sims, S. D., & Conati, C. (2020). A neural architecture for detecting user confusion in eye‑tracking data. *In Proceedings of the 2020 International Conference on Multimodal Interaction* (pp. 15–23). Association for Computing Machinery. https://doi.org/10.1145/3382507.3418828

- Sriram, H., Conati, C., & Field, T. (2023). Classification of Alzheimer’s disease with deep learning on eye‑tracking data. *In Proceedings of the 25th International Conference on Multimodal Interaction* (pp. 104–113). Association for Computing Machinery. https://doi.org/10.1145/3577190.3614149


