#!/bin/bash
#SBATCH -J stormcast_train            # set job name
#SBATCH -N 1                   # number of nodes
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH -p NCDR_StormCast      # need partition name
#SBATCH -w cnode3-004          # specify node
#SBATCH -o %x.%j.out           # output log
#SBATCH -e %x.%j.err           # error log
#SBATCH --gpus-per-node=8      # number of GPU per node
#SBATCH -t 30-00:00:00                      # job runtime limit

echo -e "Running on hosts: $(echo $(scontrol show hostname))"

srun -N 1 -w cnode3-004 -p NCDR_StormCast --mpi=pmix --gres=gpu:8 --ntasks-per-node 1 \
  --container-image /mnt/home/dczy-cmla-8ec31f/physicsnemo_25.03_02.sqsh \
  --container-writable \
  --container-remap-root \
  --container-mounts=/mnt/home/dczy-cmla-8ec31f/projects/N4/physicsnemo:/workspace,\
/mnt/home/dczy-cmla-8ec31f/hyyeh/data:/mnt/data,\
/mnt/home/dczy-cmla-8ec31f/NCDR_StormCast:/mnt/ncdr \
  --container-workdir=/workspace/dev \
  bash -c "
    num_tests=\$(python3 - <<'EOF'
import yaml
with open('test_plan.yaml') as f:
    import yaml
    data = yaml.safe_load(f)
print(len(data['tests']))
EOF
    )

    echo Found \$num_tests tests

    for i in \$(seq 0 \$((num_tests-1))); do
        echo Running test \$i
        make run
    done
  "
