#!/bin/bash
#SBATCH --job-name=at25b5k
#SBATCH --account=project_2009954
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END,FAIL

cd /scratch/project_2009954/gflownet
module load python-data
source ../envs/gfn/bin/activate

srun python main.py logger.do.online=True \
    env=arithmetictree \
    env.max_int=25 \
    env.min_int=2 \
    env.max_operations=5 \
    env.targets=\[19,20,21,23,25\] \
    proxy=arithmetictree \
    proxy.base_reward=0.0 \
    proxy.completion_reward=1.0 \
    gflownet.optimizer.n_train_steps=5000