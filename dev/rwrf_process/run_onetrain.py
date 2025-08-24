import sys
from configger import ConfigProcesser
import subprocess
from datetime import datetime
import argparse
import os
import shlex

def create_data():
    subprocess.run(["python", "create_data.py"])

def train(stormcast_dir, log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    env = {**os.environ, "PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1"}
    bash_cmd = (
        f'set -o pipefail; '
        f'torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py '
        f'2>&1 | tee "{log_file}"'
    )
    subprocess.run(
        ["bash", "-lc", bash_cmd],
        cwd=stormcast_dir,
        env=env,
        check=True,
    )

def reconfig(cfgp):
    test_plan = cfgp.load(plan_path)

    idx = test_plan["progress"]["next_test"]
    if idx is not None:
        test = test_plan["tests"][idx]
    else:
        print("[warn] No next test available in the test plan.")
        return (None, None)
    
    cfgp.dump({
        "fruits": test["fruits"],
        "quantity": test["quantity"]      
    }, cfg_path)

    test_name = test.get("name")
    print(f"Configuration updated for {test_name}.")
    return idx, test_name

def save_progress_config(cfgp, test_id, test_name, plan_path):
    test_plan = cfgp.load(plan_path)
    total_tests = len(test_plan.get("tests", []))
    test_plan.setdefault("progress", {})
    test_plan["progress"]["complete"] = test_name
    test_plan["progress"]["next_test"] = test_id + 1 if test_id < total_tests - 1 else None

    cfgp.dump(test_plan, plan_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PLAN_PATH', type=str, default="test_plan.yaml")
    parser.add_argument('--RWRF_DIR', type=str, default=None)
    parser.add_argument('--STORMCAST_DIR', type=str, default=None)
    parser.add_argument('--LOG_DIR', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    # Initialize paths and config processor
    args = parse_args()
    plan_path = args.PLAN_PATH
    cfg_path = f"{args.RWRF_DIR}/config.yaml"
    cfgp = ConfigProcesser()

    # Reconfigure, create_data and train
    #test_id, test_name = reconfig(cfgp, plan_path, cfg_path)
    #if test_id is None:
    #    sys.exit(0)
    create_data()
    train(args.STORMCAST_DIR, args.LOG_DIR)

    # Save progress
    #save_progress_config(cfgp, test_id, test_name, plan_path)
