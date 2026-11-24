"""Corrected TIEO using three quantile/rank-like fitness states."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "quantile"
STATE_MODE = "quantile"


def run(**kwargs):
    return run_suite(METHOD_NAME, STATE_MODE, **kwargs)


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE)

