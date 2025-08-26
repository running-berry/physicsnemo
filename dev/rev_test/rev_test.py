import sys
from configger import ConfigProcesser
from copy import deepcopy
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

PLAN_PATH = "test_plan.yaml"

def _flow_list(items):
    cs = CommentedSeq([dq(str(x)) for x in items])
    cs.fa.set_flow_style()   # [ "a", "b", "c" ]
    return cs

def append_test(tests):
    """Copy first test and append as new one with incremented name."""
    base = deepcopy(tests[0])
    new_idx = len(tests)
    base["name"] = f"test{new_idx}"
    tests.append(base)

def save_progress_config(cfgp, var, values, path=PLAN_PATH):
    plan = cfgp.load(path) or {}
    tests = plan.get("tests", [])
    while len(values) > len(tests):
        append_test(tests)

    if values and isinstance(values[0], tuple): # [(idx, val), ...]
        for idx, val in values:
            tests[idx][var] = _flow_list(val if isinstance(val, list) else [val])
    elif len(values) == len(tests):     # [val1, val2, ...]
        for i, val in enumerate(values):
            tests[i][var] = _flow_list(val if isinstance(val, list) else [val])
    else:
        raise ValueError("values must match #tests or be (idx, val) pairs")

    cfgp.width = 4096 #avoid line wrapping of long flow lists

    cfgp.dump(plan, path)

if __name__ == "__main__":
    cfgp = ConfigProcesser()
    var = "train_dates"
    val_list = [
        ['2019/08/03', '2019/08/03'],
        ['2019/08/01', '2019/08/31'],
        ['2019/08/01', '2019/12/31'],
        ['2019/08/01', '2020/07/31'],
         ]
    #val_list = [["t2m", v] for v in val_list]
    save_progress_config(cfgp, var, val_list)