#!/bin/bash
#SBATCH -J job_name            # set job name
#SBATCH -N 1                   # number of nodes
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH -p <Group Name>     # need partition name
#SBATCH -w cnode3-003      # need partition name
#SBATCH -o %x.%j.out           # output log
#SBATCH -e %x.%j.err           # error log
#SBATCH --gpus-per-node=1      # number of GPU per node
# job runtime limit
#SBATCH -t 7-00:00:00             

echo -e "Running on hosts: $(echo $(scontrol show hostname))"

srun -N 1 -w cnode3-003 -p <Group Name> --mpi=pmix --gres=gpu:1 --ntasks-per-node 1 \
  --container-image /mnt/home/usr/<container>.sqsh \
  --container-writable \
  --container-remap-root \
  --container-mounts=/mnt/home/usr/projects/physicsnemo:/workspace,\
/mnt/home/usr/usr/data:/mnt/data \
  --container-workdir=/workspace/dev \
  bash -c "
  python rdd_run.py
  "