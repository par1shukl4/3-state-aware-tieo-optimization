"""Baseline TIEO with stagnation-triggered restart of the worst 5 percent."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "stagnation_restart"
STATE_MODE = "baseline"
REWARD_MODE = "binary"
CONFIG_OVERRIDES = {
    "restart_fraction": 0.05,
    "stagnation_patience": 1_500,
    "max_restarts": 2,
}


def run(**kwargs):
    return run_suite(
        METHOD_NAME,
        STATE_MODE,
        reward_mode=REWARD_MODE,
        config_overrides=CONFIG_OVERRIDES,
        **kwargs,
    )


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES)

