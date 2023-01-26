import numpy as np
from super_net import PointReservoir
from super_functions import picklin, spks_to_txt
from super_argparse import setup_argument_parser
from soen_plotting import raster_plot


def main():

    input = picklin("results","saccade_mnist_10")
    args = setup_argument_parser()

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
    path = f'reservoirs_2/'
    name = f'res_{int(params["beta"])}_{params["tau"]}_{params["tau_ref"]}_run_{args.run}'
    print(f"Run: {args.run} -- {name}")
    # print(params)

    res = PointReservoir(**params)
    res.connect_input(input)
    res.run_network()
    # picklit(res,path,name)
    spks_to_txt(res.net.spikes,res.N,8,path,name)
    # raster_plot(res.net.spikes)

if __name__=='__main__':
    main()
