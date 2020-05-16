# HPC lab 2
Code runs on

python 3.6.3

pytorch 0.3.0_4

To run on HPC Prince: srun --nodes=1 --cpus-per-task=2 --gres=gpu:p40:1 --time=5:00:00 --mem=8GB --pty /bin/bash
