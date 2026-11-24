"""Quantile-state TIEO with the final 5 percent reserved for local refinement."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "quantile_local_refinement"
STATE_MODE = "quantile"
REWARD_MODE = "binary"
CONFIG_OVERRIDES = {"local_refinement_fraction": 0.05, "local_refinement_method": "Powell"}


def run(**kwargs):
    return run_suite(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES, **kwargs)


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE, REWARD_MODE, CONFIG_OVERRIDES)
