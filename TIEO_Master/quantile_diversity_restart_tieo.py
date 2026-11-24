"""Quantile-state TIEO with stagnation- and diversity-triggered 10 percent restart."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "quantile_diversity_restart"
STATE_MODE = "quantile"
REWARD_MODE = "binary"
CONFIG_OVERRIDES = {
    "restart_fraction": 0.10,
    "stagnation_patience": 1_500,
    "max_restarts": 2,
    "restart_requires_low_diversity": True,
    "restart_diversity_threshold": 0.05,
}


def run(**kwargs):
    return run_suite(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES, **kwargs)


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES)
