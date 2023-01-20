
import argparse

def setup_argument_parser():

    parser = argparse.ArgumentParser()

    # OO implementation
    parser.add_argument("--run", help = " ", type = int, default = 1)
    parser.add_argument("--runs", help = " ", type = int, default = 1)
    return parser.parse_args()