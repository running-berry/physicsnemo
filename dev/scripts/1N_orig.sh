#!/bin/bash
#SBATCH -J job_name            # set job name
#SBATCH -N 1                   # number of nodes
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH -p NCDR_StormCast      # need partition name
#SBATCH -o %x.%j.out           # output log
#SBATCH -e %x.%j.err           # error log
#SBATCH --gpus-per-node=1      # number of GPU per node
# job runtime limit
#SBATCH -t 01:00:00

echo -e "Running on hosts: $(echo $(scontrol show hostname))"

srun -N 1 -p NCDR_StormCast  --mpi=pmix --gres=gpu:1 --ntasks-per-node 1 \
  --container-image /mnt/home/dczy-cmla-8ec31f/physicsnemo_25.03_02.sqsh \
  --container-writable \
  --container-remap-root \
  --container-mounts=/mnt/home/dczy-cmla-8ec31f/projects/physicsnemo:/workspace,\
/mnt/home/dczy-cmla-8ec31f/hyyeh/data:/mnt/data \
  --container-workdir=/workspace/dev \
  bash -c "
  make run
  "
