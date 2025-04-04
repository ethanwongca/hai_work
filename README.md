# Human AI-Interaction Work

This repository contains all the code developed during my directed studies on Human-AI Interaction. The code is organized into several directories, each addressing different aspects of the project.

## Directory Structure

### 1. `preprocessing`
This folder includes all the functions and scripts used to transform raw Tobii data into the necessary inputs for VTNet. It covers:
- **Data Preprocessing:** Routines for cleaning and formatting raw data.
- **Exploratory Data Analysis (EDA):** Functions for visualizing and understanding the data.
- **Dataset Construction:** Scripts to build the within-task dataset.
- **Utility Functions:** Including scanpath extraction and cyclic splitting functions.

### 2. `hpc`
This folder contains resources for high-performance computing, including:
- **Slurm Template:** A template for running jobs on the ubc_ml cluster.
- **Statistics Script:** A script to compute the standard deviation of the AUC (Area Under the Curve) and accuracy metrics.

### 3. `VTNet_within_tasks`
This folder holds various implementations of VTNet experiments for the within-task dataset.  
**Note:** The `st_pickle_loader` function’s `max_length` parameter is critical—it adjusts VTNet for different sequence lengths (e.g., 1000 to 3000).

### 4. `VTNet_across_tasks`
This folder contains the VTNet implementations and experiments for the across-task dataset.
