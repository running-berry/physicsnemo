#!/bin/bash
#SBATCH -J job_name            # set job name
#SBATCH -N 1                   # number of nodes
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH -p <Group Name>     # need partition name
#SBATCH -w cnode3-004          # specify node
#SBATCH -o %x.%j.out           # output log
#SBATCH -e %x.%j.err           # error log
#SBATCH --gpus-per-node=1      # number of GPU per node
#SBATCH -t 7-00:00:00                      # job runtime limit

echo -e "Running on hosts: $(echo $(scontrol show hostname))"

num_tests=$(python3 - <<'EOF'
import yaml
with open("../test_plan.yaml") as f:
    data = yaml.safe_load(f)
print(len(data["tests"]))
EOF
)

echo "Found $num_tests tests"
make -C .. reset


srun -N 1 -w cnode3-004 -p <Group Name>--mpi=pmix --gres=gpu:1 --ntasks-per-node 1 \
  --container-image /mnt/home/usr/<container>.sqsh \
  --container-writable \
  --container-remap-root \
  --container-mounts=/mnt/home/usr/projects/physicsnemo:/workspace,\
/mnt/home/usr/usr/data:/mnt/data,\
/mnt/home/usr/<Group Name>:/mnt/ncdr \
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
