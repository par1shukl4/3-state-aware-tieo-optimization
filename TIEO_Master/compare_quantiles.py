"""Run and compare only the original quantile TIEO and its five variants."""

from compare_all import build_parser, compare_methods
from experiment_utils import QUANTILE_METHODS


if __name__ == "__main__":
    parser = build_parser()
    parser.description = "Run and compare the six quantile-state TIEO methods"
    destination = compare_methods(
        parser.parse_args(),
        QUANTILE_METHODS,
        reference_method="quantile",
        comparison_name="quantile_comparison",
    )
    print(f"\nQuantile comparison saved to: {destination.resolve()}")
