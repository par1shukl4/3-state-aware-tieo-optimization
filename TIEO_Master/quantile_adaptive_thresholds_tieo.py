"""Quantile-state TIEO with thresholds moving from q20/q80 to q33/q67."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "quantile_adaptive_thresholds"
STATE_MODE = "quantile"
REWARD_MODE = "binary"
CONFIG_OVERRIDES = {
    "quantile_schedule": "adaptive",
    "quantile_lower_start": 0.20,
    "quantile_upper_start": 0.80,
}


def run(**kwargs):
    return run_suite(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES, **kwargs)


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES)
