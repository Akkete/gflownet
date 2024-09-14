#!/bin/bash
#SBATCH --job-name=rte
#SBATCH --account=project_2009954
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=END,FAIL

cd /scratch/project_2009954/gflownet
module load python-data
source ../envs/gfn/bin/activate

export PYTHONPATH=. 
export RUNPATH='2024-07-10_17-11-23_s_unfiltered'

srun python scripts/eval_gflownet.py \
    --run_path external/logs/gflownet/${RUNPATH}/ \
    --n_samples 10000 \
    --sampling_batch_size 100 \
    --print_config \
    --randominit