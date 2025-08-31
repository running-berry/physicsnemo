import sys
from configger import ConfigProcesser
import subprocess
from datetime import datetime
import argparse
import os
import shlex

def make_cache(rwrf_dir):
    subprocess.run(
        ["python", "nc_to_npz.py"],
        cwd=rwrf_dir   # this runs the command as if you had cd'ed there
    )
def create_data(rwrf_dir):
    subprocess.run(
        ["python", "create_data.py"],
        cwd=rwrf_dir
    )

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

def reconfig(cfgp, plan_path, cfg_path, stormcast_path):
    test_plan = cfgp.load(plan_path)

    idx = test_plan["progress"]["next_test"]
    if idx is not None:
        test = test_plan["tests"][idx]
    else:
        print("[warn] No next test available in the test plan.")
        return (None, None)
    
    dir_path = os.path.dirname(cfg_path)
   
    # Update config.yaml with test parameters 
    cfg_set = cfgp.load(cfg_path)

    cfg_set["invariants"] = test["invariants"]
    cfg_set["var_highres"] = test["var_highres"]
    cfg_set["var_lowres"] = test["var_lowres"]
    cfg_set["lon_min"] = test["lon_min"]
    cfg_set["lon_max"] = test["lon_max"]
    cfg_set["lat_min"] = test["lat_min"]
    cfg_set["lat_max"] = test["lat_max"]
    
    cfgp.dump(cfg_set, cfg_path)

    # Update dataset/small.yaml with test parameters
    cfg_path = f"{stormcast_path}/config/dataset/small.yaml" 
    cfg_set = cfgp.load(cfg_path)

    cfg_set["invariants"] = test["invariants"]
    cfg_set["exp_train_zarrs"] = test["exp_train_zarrs"]
    cfg_set["train_dates"] = test["train_dates"]
    cfg_set["valid_dates"] = test["valid_dates"]
    
    cfgp.dump(cfg_set, cfg_path)

    # Update training/small.yaml with test parameters
    cfg_path = f"{stormcast_path}/config/training/small.yaml" 
    cfg_set = cfgp.load(cfg_path)

    cfg_set["total_train_steps"] = test["total_train_steps"]
    cfg_set["validation_freq"] = test["validation_freq"]
    cfg_set["checkpoint_freq"] = test["checkpoint_freq"]
    cfg_set["print_progress_freq"] = test["print_progress_freq"]
    cfg_set["validation_plot_variables"] = test["validation_plot_variables"]
    cfg_set["loss"] = test["loss"]
    if cfg_set["loss"] == 'edm':
        _revise_cfg_diff(cfgp, dir_path, stormcast_path, test["reg_weights"])

    cfgp.dump(cfg_set, cfg_path)

    test_name = test.get("name")
    print(f"Configuration updated for {test_name}.")
    return idx, test_name

def _revise_cfg_diff(cfgp, dir_path, stormcast_path, reg_weights_path):
    reg_weights_path = os.path.join(dir_path, "../../", reg_weights_path) 
    # Update training/small.yaml with pretrained regression model
    cfg_path = f"{stormcast_path}/config/diffusion.yaml" 
    cfg_set = cfgp.load(cfg_path)
    cfg_set["model"]["regression_weights"] = reg_weights_path
    cfgp.dump(cfg_set, cfg_path)

    # Update model/stormcast.yaml with pretrained regression model
    cfg_path = f"{stormcast_path}/config/model/stormcast.yaml" 
    cfg_set = cfgp.load(cfg_path)
    cfg_set["regression_weights"] = reg_weights_path
    cfgp.dump(cfg_set, cfg_path)

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
    test_id, test_name = reconfig(cfgp, plan_path, cfg_path, args.STORMCAST_DIR)
    if test_id is None:
        sys.exit(0)

    #make_cache(args.RWRF_DIR)
    create_data(args.RWRF_DIR)
    train(args.STORMCAST_DIR, args.LOG_DIR)

    # Save progress
    save_progress_config(cfgp, test_id, test_name, plan_path)
