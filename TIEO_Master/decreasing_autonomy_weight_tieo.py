"""Baseline TIEO with autonomy weight decreasing linearly from 0.9 to 0.15."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "decreasing_autonomy_weight"
STATE_MODE = "baseline"
REWARD_MODE = "binary"
CONFIG_OVERRIDES = {
    "weight_schedule": "linear",
    "w_start": 0.9,
    "w_end": 0.15,
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

