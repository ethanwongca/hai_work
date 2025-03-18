#!/bin/bash
#SBATCH --job-name=vtnet_trainer         # Job name
#SBATCH --output=vtnet_1000_%j.out    # Standard output file
#SBATCH --error=vtnet_1000_%j.err     # Standard error file
#SBATCH --time=10:00:00                  # Walltime (adjust as needed)
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=24G                        # Memory per node
#SBATCH --gres=gpu:1                     # Request one GPU
#SBATCH --partition=<alloc-code>            # Allocation code from depo 
#SBATCH --mail-user=<email>               # Email for notifications
#SBATCH --mail-type=ALL                   # When to send email notifications

# Set bash as the shell
export SHELL=/bin/bash

# Change to your working directory
cd <working_directory>

# Activate conda environment (update the path as necessary)
source <working_directory>/miniconda3/etc/profile.d/conda.sh
conda activate vtenv

# Run your training script (replace with your actual command)
echo "VTNET_att_1000 Meara Results"
python vtnet_att_meara.py

echo "VTNET_att_1000 BarChart Results"
python vtnet_att_barchart_lit.py  

echo "VTNET_att_1000 VerbalWM Results"
python vtnet_att_verbalwm.py
