# Human-AI Interaction: Eye-Tracking Analysis with VTNet 

This repository contains code and documentation from research on Human-AI Interaction, focusing on eye-tracking data analysis using VTNet. T

### Key Features

- **Eye-tracking Data Processing**: Transform raw Tobii data (120 Hz) into VTNet-compatible formats
- **Cross-validation Frameworks**: Specialized CV approaches for different classification scenarios
- **HPC Integration**: Ready-to-use scripts for high-performance computing environments
- **Documentation**: Documents what files to look for in the lab server, how to use VTNet, HPCs, and more

## 📁 Repository Structure

### [`Dataset`](./Dataset/)
Contains documentation on locating the necessary data files within the lab repository.

### [`preprocess`](./preprocess/) 
Tools for transforming raw eye-tracking data:
- 🧹 **Data Cleaning & Formatting**: Scripts to clean and standardize raw Tobii data
- 📊 **Exploratory Data Analysis**: Functions for visualizing and understanding eye-tracking patterns
- 🏗️ **Dataset Construction**: Tools for building within-task and across-task datasets
- 🛠️ **Utility Functions**: Including scanpath extraction and cyclic data splitting

### [`hpc`](./hpc/)
Resources for high-performance computing:
- 📝 **Slurm Templates**: Ready-to-use templates for the `ubc_ml` cluster
- 📈 **Statistical Analysis**: Scripts for computing performance metrics from VTNet outputs

### [`VTNet_within_tasks`](./VTNet_within_tasks/)
Implementations of VTNet for the within-task dataset:
- ⏱️ Multiple sequence length configurations (14s, 29s, and length 1000)
- 📄 Documentation on required VTNet modifications
- ⚠️ **Important**: The `st_pickle_loader` function's `max_length` parameter is critical for adjusting sequence lengths

### [`VTNet_User_Aggregate`](./VTNet_across_tasks/)
VTNet implementations for the evaluating per user VTNet dataset:
- 🔄 Cross-task prediction configurations
- 📄 Documentation on required model modifications
- 📝 The extra preprocessing done to make the within-tasks dataset to per user

## 🛠️ Recommended Directory Structure for VTNet (HPC)

While multiple directory structures are possible, here is our recommended setup:

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

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Conda (for environment setup)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ethanwongca/hai_work
   ```

2. **Navigate to the project directory**
   ```bash
   cd hai_work
   ```

3. **Create and activate the Conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate hai-env
   ```

## 📚 Key Concepts

### VTNet Architecture
VTNet is a neural architecture that processes eye-tracking data to predict user confusion and Alzheimer disease. It transforms sequential eye movement data into features useful for classification tasks.

### Sequence Length Configuration
When using VTNet, pay attention to the `max_length` parameter in the `st_pickle_loader` function. This parameter adjusts the model for different sequence lengths (e.g., from 1000 to 3000).

## 📖 References

For a deeper understanding of the methods used in this repository, we recommend these papers:

- Barral, O., Lallé, S., Guz, G., Iranpour, A., & Conati, C. (2020). [Eye‑tracking to predict user cognitive abilities and performance for user‑adaptive narrative visualizations](https://doi.org/10.1145/3382507.3418884). *In Proceedings of the 2020 International Conference on Multimodal Interaction* (pp. 163–173).

- Sims, S. D., & Conati, C. (2020). [A neural architecture for detecting user confusion in eye‑tracking data](https://doi.org/10.1145/3382507.3418828). *In Proceedings of the 2020 International Conference on Multimodal Interaction* (pp. 15–23).

- Sriram, H., Conati, C., & Field, T. (2023). [Classification of Alzheimer's disease with deep learning on eye‑tracking data](https://doi.org/10.1145/3577190.3614149). *In Proceedings of the 25th International Conference on Multimodal Interaction* (pp. 104–113).


## 🤝 Acknowledgments

- UBC's Human Artificial Intelligence Interaction Lab

