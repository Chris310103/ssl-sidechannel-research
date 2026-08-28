import argparse


def parse_range(value: str):
    try:
        start, end = value.split("_")
        return int(start), int(end)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "range must use the form START_END, for example 200_900"
        ) from error


def parse_count_pair(value: str):
    try:
        first, second = value.split("_")
        return int(first), int(second)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "count pair must use the form FIRST_SECOND, for example 500_25"
        ) from error
