#!/bin/bash
#SBATCH --job-name=rtm_b20k_d20tb
#SBATCH --account=project_2009954
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END,FAIL

cd /scratch/project_2009954/gflownet
module load python-data
source ../envs/gfn/bin/activate

WANDB__SERVICE_WAIT=300 srun python main.py logger.do.online=True \
    policy.forward.checkpoint=forward \
    policy.forward.n_layers=20 \
    policy.forward.n_hid=256 \
    gflownet.random_action_prob=0.05 \
    gflownet.optimizer.n_train_steps=20000 \
    proxy.base_reward=0.0 \
    proxy.completion_reward=1.0 \
    gflownet=trajectorybalance \
    env.target_file=external/reactiontree_data/targets_20.txt \
    n_samples=100000
