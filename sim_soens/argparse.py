import argparse
import numpy as np

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
    parser.add_argument( "--ib",                type=float,         default = 1.8      )
    parser.add_argument( "--tau",               type=float,         default = 50       )
    parser.add_argument( "--beta",              type=float,         default = 2        )
    parser.add_argument( "--s_th",              type=float,         default = 0.5      )
    parser.add_argument( "--eta",               type=float,         default = .005     )
    parser.add_argument( "--elast",             type=str,           default = "None"   )
    parser.add_argument( "--valid",             type=str,           default = "True"   )
    parser.add_argument( "--exp_name",          type=str,           default = "test"   )
    parser.add_argument( "--inhibit",           type=float,         default = -1       )
    parser.add_argument( "--backend",           type=str,           default = "julia " )
    parser.add_argument( "--run",               type=int,           default = 0        )
    parser.add_argument( "--digits",            type=int,           default = 3        )
    parser.add_argument( "--samples",           type=int,           default = 10       )
    parser.add_argument( "--elasticity",        type=str,           default = None     )
    parser.add_argument( "--layers",            type=int,           default = 3        )
    parser.add_argument( "--decay",             type=str,           default = "False"  )
    parser.add_argument( "--probabilistic",     type=float,         default = 1        )
    parser.add_argument( "--weights",           type=str,           default = "preset" )
    parser.add_argument( "--dataset",           type=str,           default = "MNIST"  )
    parser.add_argument( "--duration",          type=int,           default = 1000     )
    parser.add_argument( "--low_bound",         type=float,         default = -0.5     )
    parser.add_argument( "--plotting",          type=str,           default = "sparse" )
    parser.add_argument( "--jul_threading",     type=int,           default = 1        )
    parser.add_argument( "--hebbian",           type=str,           default = "False"  )
    parser.add_argument( "--exin",              type=list_of_ints,  default = None     )
    parser.add_argument( "--fixed",             type=float,         default = None     )
    parser.add_argument( "--rand_flux",         type=float,         default = None     )
    parser.add_argument( "--inh_counter",       type=bool,          default = None     )
    parser.add_argument( "--norm_fanin",        type=bool,          default = None     )
    parser.add_argument( "--norm_fanin_prime",  type=bool,          default = None     )
    parser.add_argument( "--lay_weighting",     type=list_of_ints,  default = None     )
    parser.add_argument( "--fan_coeff",         type=float,         default = 1.5      )
    parser.add_argument( "--fan_buffer",        type=float,         default = 0        )
    parser.add_argument( "--dt",                type=float,         default = .1       )
    parser.add_argument( "--target",            type=int,           default = 10       )
    parser.add_argument( "--off_target",        type=int,           default = 0        )
    parser.add_argument( "--max_offset",        type=str,           default = "phi_th" )    
    parser.add_argument( "--tiling",            type=bool,          default = None     )
    parser.add_argument( "--alternode",         type=str,           default = None     )
    parser.add_argument( "--weight_transfer",   type=str,           default = None     )
    parser.add_argument( "--offset_transfer",   type=str,           default = None     )
    parser.add_argument( "--no_negative_jij",   type=bool,          default = False    )



    parser.add_argument( '--N'                 , type=int,        default = 98            )
    parser.add_argument( '--C'                 , type=int,        default = 3             )
    parser.add_argument( '--seed'              , type=int,        default = 442           )
    parser.add_argument( '--runs'              , type=int,        default = 25            )


    parser.add_argument( '--nodes_tau'         , type=int,        default = 50            )
    parser.add_argument( '--nodes_beta'        , type=float,      default = 2*np.pi*10**3 )
    parser.add_argument( '--nodes_ib'          , type=float,      default = 1.8           )
    parser.add_argument( '--nodes_s_th'        , type=float,      default = 0.1           )
    parser.add_argument( '--fan_coeff_nodes'   , type=float,      default = 2.25          )

    parser.add_argument( '--codes_tau'         , type=int,        default = 250           )
    parser.add_argument( '--codes_beta'        , type=float,      default = 2*np.pi*10**3 )
    parser.add_argument( '--codes_ib'          , type=float,      default = 1.8           )
    parser.add_argument( '--codes_s_th'        , type=float,      default = 0.1           )
    parser.add_argument( '--fan_coeff_codes'   , type=float,      default = 1.5           )

    parser.add_argument( '--density'           , type=float,      default = 0.1           )
    parser.add_argument( '--res_connect_coeff' , type=float,      default = 0.1           )

    parser.add_argument( '--evolve',             type=bool,       default = False         )
    parser.add_argument( '--multi',              type=bool,       default = False         )

    parser.add_argument( '--neuromod',           type=bool,       default = False         )

    # parser.add_argument( "int_var1",        type=int,           default = None     )
    # parser.add_argument( "int_var2",        type=int,           default = None     )
    # parser.add_argument( "int_var3",        type=int,           default = None     )
    # parser.add_argument( "int_var4",        type=int,           default = None     )
    # parser.add_argument( "int_var5",        type=int,           default = None     )
    # parser.add_argument( "int_var6",        type=int,           default = None     )

    # parser.add_argument( "flt_var1",        type=float,         default = None     )
    # parser.add_argument( "flt_var2",        type=float,         default = None     )
    # parser.add_argument( "flt_var3",        type=float,         default = None     )
    # parser.add_argument( "flt_var4",        type=float,         default = None     )
    # parser.add_argument( "flt_var5",        type=float,         default = None     )
    # parser.add_argument( "flt_var6",        type=float,         default = None     )
    

    return parser.parse_args()
