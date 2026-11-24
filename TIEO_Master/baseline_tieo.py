"""Corrected two-state baseline TIEO experiment."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "baseline"
STATE_MODE = "baseline"


def run(**kwargs):
    return run_suite(METHOD_NAME, STATE_MODE, **kwargs)


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE)

