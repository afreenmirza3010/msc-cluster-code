#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=vnet_test   # sensible name for the job
#SBATCH --mem=196G             # Default memory per CPU is 3GB.
#SBATCH --partition=gpu   # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --mail-user=afmi@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/vnet-%A.out
#SBATCH --error=outputs/vnet-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
echo "Copying data..."
bash copy_dataset.sh
echo "Copy finished"
singularity exec --nv deoxys.sif python experiment.py