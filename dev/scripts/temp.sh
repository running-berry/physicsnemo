#!/bin/bash
source .env
#SBATCH -J job_name            # set job name
#SBATCH -N 1                   # number of nodes
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH -p NCDR_StormCast      # need partition name
#SBATCH -o %x.%j.out           # output log
#SBATCH -e %x.%j.err           # error log
#SBATCH --gpus-per-node=1      # number of GPU per node
# job runtime limit
#SBATCH -t 01:00:00

HOMEDIR=/mnt/home/dczy-cmla-8ec31f  # replace with actual home directory
SQSH=/physicsnemo_25.03_02.sqsh  # replace with actual SQSH file
PARTITION=NCDR_StormCast  # replace with actual partition name
NODE_ID=cnode3-003

echo -e "Running on hosts: $(echo $(scontrol show hostname))"

srun -N 1 -p ${PARTITION}  --mpi=pmix --gres=gpu:1 \
    --container-image ${HOMEDIR}${SQSH} \
    --container-writable \
    --container-remap-root \
    --container-mounts ${HOMEDIR}/projects/physicsnemo:/workspace \
    bash -c "
    cd dev
    make run
    "


