
import argparse

def setup_argument_parser():

    parser = argparse.ArgumentParser()

    # OO implementation
    parser.add_argument("--run", help = " ", type = int, default = None)
    parser.add_argument("--runs", help = " ", type = int, default = 1)
    parser.add_argument("--form", help = " ", type = str, default = 'standalone')
    parser.add_argument("--dir", help = " ", type = str, default = 'testing')
    parser.add_argument("--beta", help = " ", type = int, default = 2)
    parser.add_argument("--tau", help = " ", type = int, default = 50)
    parser.add_argument("--tau_ref", help = " ", type = int, default = 500)

    parser.add_argument("--inhibit", help = " ", type = float, default = -.25)
    parser.add_argument("--verbose", help = " ", type = bool, default = False)
    return parser.parse_args()