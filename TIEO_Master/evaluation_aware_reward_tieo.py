"""Baseline TIEO with evaluation-aware, magnitude-aware Q rewards."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "evaluation_aware_reward"
STATE_MODE = "baseline"
REWARD_MODE = "evaluation_aware"


def run(**kwargs):
    return run_suite(
        METHOD_NAME,
        STATE_MODE,
        reward_mode=REWARD_MODE,
        **kwargs,
    )


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE, REWARD_MODE)

