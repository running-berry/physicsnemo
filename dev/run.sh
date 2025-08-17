#!/usr/bin/env bash
set -e

num_tests=$(python3 - <<'EOF'
import yaml
with open("test_plan.yaml") as f:
    data = yaml.safe_load(f)
print(len(data["tests"]))
EOF
)

echo "Found $num_tests tests"
make reset

for i in $(seq 0 $((num_tests-1))); do
    echo "Running test $i"
    make run
done