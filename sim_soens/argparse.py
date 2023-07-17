import argparse

def setup_argument_parser():

    parser = argparse.ArgumentParser()

    # OO implementation
    parser.add_argument( "--ib",            type=float,     default = 1.8       )
    parser.add_argument( "--tau",           type=float,     default = 150       )
    parser.add_argument( "--beta",          type=float,     default = 2         )
    parser.add_argument( "--s_th",          type=float,     default = 0.5       )
    parser.add_argument( "--eta",           type=float,     default = .005      )
    parser.add_argument( "--elast",         type=str,       default = "None"    )
    parser.add_argument( "--valid",         type=str,       default = "True"    )
    parser.add_argument( "--exp_name",      type=str,       default = "pixels"  )
    parser.add_argument( "--inhibit",       type=float,     default = -1        )
    parser.add_argument( "--backend",       type=str,       default = "python"  )
    parser.add_argument( "--run",           type=int,       default = 0         )
    parser.add_argument( "--name",          type=str,       default = "test"    )
    parser.add_argument( "--digits",        type=int,       default = 3         )
    parser.add_argument( "--samples",       type=int,       default = 10        )
    parser.add_argument( "--elasticity",    type=str,       default = "elastic" )
    parser.add_argument( "--layers",        type=int,       default = 3         )

    return parser.parse_args()
