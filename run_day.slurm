#!/bin/sh

#SBATCH -J sim-run
#SBATCH --error=logs/sim_run-%A_%a.err
#SBATCH --output=logs/sim_run-%A_%a.out
#SBATCH --nodes=1
#SBATCH --partition=llm_h
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

python -m run_day.py