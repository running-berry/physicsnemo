#!/bin/bash
#SBATCH -J job_name            # set job name
#SBATCH -N 1                   # number of nodes
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH -p NCDR_StormCast      # partition name
#SBATCH -o %x.%j.out           # output log
#SBATCH -e %x.%j.err           # error log
#SBATCH --gpus-per-node=1      # number of GPUs per node
#SBATCH -t 01:00:00            # job runtime limit

HOMEDIR=<HOME_DIR>  # replace with actual home directory
SQSH=<SQSH_FILE>  # replace with actual SQSH file
PARTITION=<PARTITION_NAME>  # replace with actual partition name
DATAPATH=<DATA_PATH> # replace with actual data path

echo -e "Running on hosts: $(scontrol show hostname)"

srun -N 1 -p ${PARTITION}  --mpi=pmix --gres=gpu:1 \
  --container-image ${HOMEDIR}${SQSH} \
  --container-writable \
  --container-remap-root \
  --container-mounts=${HOMEDIR}/projects/physicsnemo:/workspace,\
${HOMEDIR}/${DATAPATH}/ \
  bash -c "
    cd dev
    #make run
  "
