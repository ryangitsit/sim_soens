import numpy as np
from _util__soen import dend_load_arrays_thresholds_saturations
from _util import index_finder

ib__list__ri, phi_r__array__ri, i_di__array__ri, r_fq__array__ri, phi_th_plus__vec__ri, phi_th_minus__vec__ri, s_max_plus__vec__ri, s_max_minus__vec__ri, s_max_plus__array__ri, s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')

default_neuron_params = {
    # dendrites
    "beta_di": 2*np.pi*1e2,
    "tau_di": 1000,
    "ib": ib__list__ri[9], 

    # neurons
    "ib_n": ib__list__ri[9], 
    "s_th_factor_n": 0.1,
    # "phi_th_n":,
    "beta_ni": 2*np.pi*1e3,
    "tau_ni": 50,

    # connections
    "w_sd": 1,
    "w_sid": 1, 
    "w_dd": 0.5,
    "w_dn": .5, 

    # refraction loop
    "ib_ref": ib__list__ri[8], 
    "beta_ref": 2*np.pi*1e4,
    "tau_ref": 500,
}

default_neuron_params["s_max_n"]=s_max_plus__vec__ri[index_finder(default_neuron_params['ib_n'],ib__list__ri[:])]
