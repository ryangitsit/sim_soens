import argparse

def list_of_ints(arg):
    '''
    For parsing a command line argument as a list of integers with `setup_argument_parser`
    '''
    return list(map(int, arg.split(',')))

def setup_argument_parser():
    '''
    Command line arguments to run with sim_soens
    '''  

    parser = argparse.ArgumentParser()

    # OO implementation 
    parser.add_argument( "--ib",            type=float,         default = 1.8      )
    parser.add_argument( "--tau",           type=float,         default = 50      )
    parser.add_argument( "--beta",          type=float,         default = 2        )
    parser.add_argument( "--s_th",          type=float,         default = 0.5      )
    parser.add_argument( "--eta",           type=float,         default = .005     )
    parser.add_argument( "--elast",         type=str,           default = "None"   )
    parser.add_argument( "--valid",         type=str,           default = "True"   )
    parser.add_argument( "--exp_name",      type=str,           default = "pixels" )
    parser.add_argument( "--inhibit",       type=float,         default = -1       )
    parser.add_argument( "--backend",       type=str,           default = "julia " )
    parser.add_argument( "--run",           type=int,           default = 0        )
    parser.add_argument( "--name",          type=str,           default = "test"   )
    parser.add_argument( "--digits",        type=int,           default = 3        )
    parser.add_argument( "--samples",       type=int,           default = 10       )
    parser.add_argument( "--elasticity",    type=str,           default = None     )
    parser.add_argument( "--layers",        type=int,           default = 3        )
    parser.add_argument( "--decay",         type=str,           default = "False"  )
    parser.add_argument( "--probabilistic", type=float,         default = 1        )
    parser.add_argument( "--weights",       type=str,           default = "preset" )
    parser.add_argument( "--dataset",       type=str,           default = "MNIST"  )
    parser.add_argument( "--duration",      type=int,           default = 250      )
    parser.add_argument( "--low_bound",     type=float,         default = -0.5     )
    parser.add_argument( "--plotting",      type=str,           default = "sparse" )
    parser.add_argument( "--jul_threading", type=int,           default = 1        )
    parser.add_argument( "--hebbian",       type=str,           default = "False"  )
    parser.add_argument( "--exin",          type=list_of_ints,  default = None     )
    parser.add_argument( "--fixed",         type=float,         default = None     )
    parser.add_argument( "--rand_flux",     type=float,         default = None     )
    parser.add_argument( "--inh_counter",   type=bool,          default = None     )
    parser.add_argument( "--norm_fanin",    type=bool,          default = None     )
    parser.add_argument( "--lay_weighting", type=list_of_ints,  default = None     )
    parser.add_argument( "--fan_coeff",     type=float,         default = 1.5      )
    parser.add_argument( "--dt",            type=float,         default = .1       )
    parser.add_argument( "--target",        type=int,           default = 10       )
    parser.add_argument( "--max_offset",    type=float,         default = None     )
    
    


    return parser.parse_args()
