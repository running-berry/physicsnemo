from configger import ConfigProcesser
import argparse

def reset_progress_config(cfgp, path="test_plan.yaml"):
    test_plan = cfgp.load(path)
    test_plan["progress"]["complete"] = 'null'
    test_plan["progress"]["next_test"] = 0

    cfgp.dump(test_plan, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--PLAN_PATH', type=str, default="test_plan.yaml")
    cfgp = ConfigProcesser()
    reset_progress_config(cfgp, parser.parse_args().PLAN_PATH)
    print("Progress reset to initial state.")