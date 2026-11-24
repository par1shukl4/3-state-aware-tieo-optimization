"""Quantile-state TIEO with baseline-style fallback for tribes below five members."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "quantile_minimum_occupancy"
STATE_MODE = "quantile"
REWARD_MODE = "binary"
CONFIG_OVERRIDES = {"quantile_min_tribe_size": 5}


def run(**kwargs):
    return run_suite(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES, **kwargs)


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES)
