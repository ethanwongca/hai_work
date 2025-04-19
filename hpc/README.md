# HPC Documentation
## Files Include:
1. `cv_split.ipynb`: Contains the original 10-fold grouped cross-validation process
2. `slurm_template.sh`: A sample slurm script for UBC's HPC. See **Running a SLURM Script on HPCs** for more info on how to run this script.
3. `stats.py`: Calculates the AUC standard deviation and Accuracy standard deviation from the .out file produced from VTNet when training VTNet on a HPC.

## Running a SLURM Script on HPCs 

1. Insert information in slurm_template.sh
2. Run the command `sbatch slurm_template.sh` or whatever set file name your slurm script is 
3. Check if the script is running via `squeue -u $USER`
4. Once the files are complete SLURM will generate an output file: `<file_name>.out` 

**For more info:** https://docs.alliancecan.ca/wiki/Running_jobs 

## Allocation Codes

All of UBC's labs have their own separate clusters, please check this documentation to use the appropriate allocation code and insert that code in `slurm_template.sh`: [UBC HPC Allocation Codes](https://github.com/plai-group/cluster-docs/wiki/UBC-ML-Cluster)

## Accessing UBC's HPC

Follow the commands below to access the HPC 
```bash
ssh <username>@remote.cs.ubc.ca
ssh submit-ml
```

## Post-HPC Workflow 
Once SLURM jobs have finished:
1. Run stats.py with the SLURM-generated output file to get the AUC standard deviation and accuracy.

## Notes for New Lab Members
1. Double-check your `#SBATCH` headers (time, memory, CPUs) this has to be within UBC's specs
2. Change the name of the `.out` files per every run or add `%j` to the file name to prevent experiments being overwritten 
3. To use conda on UBC's HPC cluster please use miniconda, then follow the environment setup instructions on the main page of this repository 
