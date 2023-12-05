#!/bin/sh

#SBATCH -J sim-run
#SBATCH --output=logs/sim_run.out
#SBATCH --nodes=1
#SBATCH --partition=llm_h
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

python -m demo_gen.py