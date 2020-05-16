#!/bin/bash
#
#SBATCH --job-name=lab2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:p40:1
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=test
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out

module load python3/intel/3.6.3
module load pytorch/python3.6/0.3.0_4

python3 ./lab2.py --title C6 --data ./data --optim adam