#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p sched_mit_pog_r8
#SBATCH -e /orcd/archive/pog/001/ju26596/COAST/QG2L/logs/COAST_%A_%a_error.out
#SBATCH -o /orcd/archive/pog/001/ju26596/COAST/QG2L/logs/COAST_%A_%a_output.out
#SBATCH --mem-per-cpu=64GB

module load julia/1.9.1 
julia COAST_QG2L.jl $SLURM_ARRAY_TASK_ID 

