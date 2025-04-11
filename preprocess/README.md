# Pre-Processing Modules
These modules do the necessary exploratory data analysis and data pre-processing to train VTNet (A RNN-CNN model specializing in Eye-Tracking data). 

1. `tasks.py`: Separates raw Tobii data into the 14 tasks needed in the study. It prepares the data for validity.py to run a validity analysis on the tasks. 

2. `validity.py`: Checks the validity of all the tasks in our dataset. This script includes functions to build validity charts, check overall total sequence lengths of tasks, and more. 

3. `input.py`: Cyclic splits the tasks that meet our validity and sequence length threshold. Creates 4x the amount of data points. Also generates the scanpaths of each task that meet the threshold. 

4. `setup.py`: Sets up the data for VTNet, separates the data into the proper corresponding folders for VTNet and does the 10 CV grouped split.

## Validity Module Charts 

In the valdity module you can build validity charts similar to the charts in this paper: **_Further Results on Predicting Cognitive Abilities for Adaptive Visualizations_ (Conati et al. 2017)** 

![download](https://github.com/user-attachments/assets/8fc09d90-32a0-46a0-9adc-f85d5528a4d2)
