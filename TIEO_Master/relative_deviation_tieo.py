"""Corrected TIEO using three scale-safe relative-deviation states."""

from experiment_utils import method_main, run_suite


METHOD_NAME = "relative_deviation"
STATE_MODE = "relative_deviation"


def run(**kwargs):
    return run_suite(METHOD_NAME, STATE_MODE, **kwargs)


if __name__ == "__main__":
    method_main(METHOD_NAME, STATE_MODE)

