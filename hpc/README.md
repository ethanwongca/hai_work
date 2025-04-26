# High-Performance Computing (HPC) Documentation

## Introduction
This guide provides comprehensive instructions for utilizing UBC's High-Performance Computing (HPC) resources for machine learning experiments, with a focus on running VTNet models. It covers everything from accessing the cluster to analyzing results.

## Repository Contents
- `slurm_template.sh`: Template SLURM script for job submission on UBC's HPC
- `stats.py`: Utility script for calculating AUC and Accuracy standard deviations from VTNet training output files

## Accessing UBC's HPC

### SSH Configuration for Easy Access
Save the following configuration in your local machine's ~/.ssh/config file to enable easy SSH access:
```
Host remote.cs.ubc.ca
    HostName remote.cs.ubc.ca
    User <username>
    PreferredAuthentications password
    PubkeyAuthentication no
    ConnectTimeout 1000

Host submit-ml
    HostName submit-ml
    User <username>
    ProxyJump remote.cs.ubc.ca
    PreferredAuthentications password
    PubkeyAuthentication no
    ConnectTimeout 60
```
Replace username with your UBC CWL. With this configuration, you can simply use:

```bash
ssh submit-ml
```

### Connection Process
If the above is not done you can also connect to the `submit-ml` environment using:
```bash
ssh <username>@remote.cs.ubc.ca
ssh submit-ml
```

## Environment Setup

### Miniconda Installation 
To use conda environments within the HPC miniconda must be installed. Below is the commands on how to set that up. 
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make the installer executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run the installer
./Miniconda3-latest-Linux-x86_64.sh
```

## Cluster Machines

This is a list of all computational resources available for UBC-ML users. All nodes run Ubuntu 20.04.4 LTS with the following software preinstalled:
* Nvidia driver version: 510.54
* CUDA version: 11.6
* Singularity version: 3.9.2
* zsh (to be added)

Installation of additional software for particular needs should be done in user environments by users.

### Hardware Specifications
| Name | CPU | CPU cores | RAM | Disk | GPU | GPU memory |
|------|-----|-----------|-----|------|-----|------------|
| ubc-ml 01-06 | 2 x AMD EPYC 7313 CPU 16C/3.0G/155W/128M | 32 physical / 64 logical | 16 x 16GB DDR4-3200 RDIMM | 2 x 3.84TB Gen3 NVMe U.2 SSD | 8 x NVIDIA RTX A5000 PCIe 4.0 GPU | 24 GB GDDR6 with ECC |

## HPC Resource Allocation

### UBC HPC Allocation Codes
Before submitting jobs, identify the appropriate allocation code for your lab from:
[UBC ML Cluster Allocation Codes](https://github.com/plai-group/cluster-docs/wiki/UBC-ML-Cluster)

## SLURM Job Submission

### Customizing the SLURM Template
Edit `slurm_template.sh` to specify:
1. Resource requirements (time, memory, CPUs, GPUs)
2. Job name and output file path
3. Email notifications (optional)
4. Allocation code
5. Environment activation commands
6. Experiment parameters and command

### Example SLURM Script
See the provided `slurm_template.sh` file for a sample script, which looks similar to:

```bash
#!/bin/bash
#SBATCH --job-name=vtnet_training
#SBATCH --output=results/vtnet_%j.out
#SBATCH --error=results/vtnet_error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=<your_lab_allocation_code>
#SBATCH --mail-user=your.email@ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules and activate environment
module load cuda/11.6
source ~/miniconda3/bin/activate vtnet

# Create output directory if it doesn't exist
mkdir -p results

# Run the training script
python train_vtnet.py
```

### Submitting Jobs
```bash
# Submit a job
sbatch slurm_template.sh

# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>
```

## Post HPC Workflow

Once SLURM jobs have finished:

Run stats.py with the SLURM-generated output file to get AUC's and accuracy's standard deviation.


## Transferring Files To/From HPC

### Using SFTP
SFTP is the recommended method for transferring files to and from UBC's HPC system:

```bash
# Connect to the server with SFTP
sftp <username>@remote.cs.ubc.ca

# Basic SFTP commands:
# Download a file from remote to local
get remote_file local_file

# Upload a file from local to remote
put local_file remote_file

# Download a directory recursively
get -r remote_directory

# Upload a directory recursively
put -r local_directory

# List remote files
ls

# Change remote directory
cd remote_directory

# Change local directory
lcd local_directory

# Exit SFTP
exit
```

## Best Practices for Lab Members

### Job Management
1. **Descriptive naming**: Use meaningful job names and output file patterns
2. Include `%j` in output file names to prevent overwriting (e.g., `vtnet_%j.out`)
3. Set appropriate time and resource limits to avoid job failures
4. Submit smaller test jobs before long production runs

### Effective Job Management
1. Use descriptive job names to easily identify your experiments
2. Clean up temporary files regularly to maintain disk space
3. Cancel idle or failed jobs promptly with `scancel`
4. Monitor job status with `squeue -u $USER`

### Troubleshooting Common Issues
1. **Out of memory errors**: Request more memory or optimize your code
2. **Timeout**: Increase the time limit or implement checkpointing
3. **Module errors**: Check if required modules are loaded correctly
4. **Permission denied**: Ensure files have correct permissions (use `chmod +x`)

## Additional Resources
- [UBC CS HPC Documentation](https://docs.alliancecan.ca/wiki/Running_jobs)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [Lab-specific Guidelines](https://github.com/plai-group/cluster-docs/wiki/UBC-ML-Cluster)

## Getting Help
For issues with the cluster, contact UBC IT via help@cs.ubc.ca
