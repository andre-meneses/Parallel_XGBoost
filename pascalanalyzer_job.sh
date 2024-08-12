#!/bin/bash
#SBATCH --partition=intel-128
#SBATCH --job-name=PaScal_job
#SBATCH --output=PaScal_job%j.out
#SBATCH --error=PaScal_job%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-1:30
#SBATCH --exclusive

pascalanalyzer -t aut -c 1,2,4,8,16,32 -i 8123,16246,32492,64984,129968,259936,519872,1039744 -o output.json ./Parallel-XGBoost
