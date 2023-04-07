import numpy as np
import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.super_net import PointReservoir
from sim_soens.super_functions import picklin, spks_to_txt
from sim_soens.super_argparse import setup_argument_parser
from sim_soens.soen_plotting import raster_plot
import time


def main():
    '''
    Reservoir Script
     - Meant to be swept over for many parameter values using sweeper_res.bat
    '''
    input = picklin("results","saccade_mnist_10")
    args = setup_argument_parser()
    args.seed=args.run
    params= {
        "N":72,
        "s_th":0.5,
        "beta":2,
        "tau":100,
        "tau_ref":50,
        "tf":3600
        }
    
    params.update(args.__dict__)
    params["beta"] = 2*np.pi*10**(params["beta"])
    path = f'reservoirs_3/'
    name = f'res_{int(params["beta"])}_{params["tau"]}_{params["tau_ref"]}_run_{args.run}'
    print(f"Run: {args.run} -- {name}")
    # print(params)
    s = time.perf_counter()
    res = PointReservoir(**params)
    res.connect_input(input)
    res.run_network()
    f = time.perf_counter()
    print(f"time to run = {f-s}")
    # picklit(res,path,name)
    # spks_to_txt(res.net.spikes,res.N,8,path,name)
    raster_plot(res.net.spikes)

if __name__=='__main__':
    main()
