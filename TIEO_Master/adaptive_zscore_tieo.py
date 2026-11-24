"""Corrected TIEO using three adaptive z-score fitness states."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "adaptive_zscore"
STATE_MODE = "adaptive_zscore"


def run(**kwargs):
    return run_suite(METHOD_NAME, STATE_MODE, **kwargs)


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE)

