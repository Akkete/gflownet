#!/bin/bash
#SBATCH --job-name=rt_xxs
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

srun python main.py \
    +experiments=reactiontree/reactiontree_xxs.yaml \
    logger.do.online=True 