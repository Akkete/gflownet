#!/bin/bash
#SBATCH --job-name=rts_b5k_d20tb
#SBATCH --account=project_2009954
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END,FAIL

cd /scratch/project_2009954/gflownet
module load python-data
source ../envs/gfn/bin/activate

srun python main.py logger.do.online=True \
    policy.forward.n_layers=20 \
    policy.forward.n_hid=256 \
    gflownet.random_action_prob=0.05 \
    gflownet.optimizer.n_train_steps=5000 \
    proxy.base_reward=0.0 \
    proxy.completion_reward=1.0 \
    gflownet=trajectorybalance