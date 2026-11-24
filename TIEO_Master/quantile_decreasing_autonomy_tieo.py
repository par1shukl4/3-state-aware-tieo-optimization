"""Quantile-state TIEO with autonomy weight decreasing from 0.5 to 0.15."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "quantile_decreasing_autonomy"
STATE_MODE = "quantile"
REWARD_MODE = "binary"
CONFIG_OVERRIDES = {"weight_schedule": "linear", "w_start": 0.5, "w_end": 0.15}


def run(**kwargs):
    return run_suite(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES, **kwargs)


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES)
