# HPC Directory 
## Files Include:
1. `cv_split.ipynb`: Contains the original 10-fold grouped cross validation process
2. `slurm_template.sh`: A sample slurm script for UBC's HPC. See **Running a SLURM Script on HPCs** for more info on how to run this script.
3. `stats.py`: Calculates the AUC standard deviation and Accuracy standard deviation from the .out file produced from VTNet when training VTNet on a HPC.

## Running a SLURM Script on HPCs 

1. Insert information in slurm_template.sh
3. Run the command `sbatch slurm_template.sh` or whatever set file name your slurm script is 
4. Check if thing is running via `squeue -u $USER`

**For more info:** https://docs.alliancecan.ca/wiki/Running_jobs 
