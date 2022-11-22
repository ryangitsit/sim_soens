import numpy as np
from _util__soen import dend_load_arrays_thresholds_saturations
from _util import index_finder

"""
Parameter dictionaries for call from other class initializations
"""

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

    'ib_list_ri':ib__list__ri[:]
}



nine_pixel_params = {

    'weights': [
        [[.5,.5,.5]],
        [[0.35,-0.65],[0.35,-0.65],[0.35,-0.65]],
        [[.6,.5],[.6,.5],[.6,.5],[.6,.5],[.6,.5],[.6,.5]]
    ],

    'betas': [
        [[2,2,2]],
        [[2,2],[2,2],[2,2]],
        [[2,2],[2,2],[2,2],[2,2],[2,2],[2,2]]
    ],

    'biases': [
        [[1,1,1]],
        [[5,5],[5,5],[5,5]],
        [[-4,3],[-4,3],[-4,3],[-4,3],[-4,3],[-4,3]]
    ],
    'types': [
        [['rtti','rtti','rtti']],
        [['ri','ri'],['ri','ri'],['ri','ri']],
        [['ri','ri'],['ri','ri'],['ri','ri'],['ri','ri'],['ri','ri'],['ri','ri']]
    ],
    'syns': [['2','5'],['4','6'],['5','8'],['4','6'],['1','3'],['7','9'],
             ['4','6'],['2','5'],['7','9'],['1','3'],['4','6'],['5','8']],

    "tau_di": 1000,
    "ib_n": ib__list__ri[9], 
    "s_th_factor_n": 0.1,
    "beta_ni": 2*np.pi*1e2,
    "tau_ni": 50,
    "w_sd": 1,
    # "w_sid": 1, 
    # "w_dd": 0.5,
    "w_dn": .5, 
    "ib_ref": ib__list__ri[8], 
    "beta_ref": 2*np.pi*1e4,
    "tau_ref": 100,
    'ib_list_ri':ib__list__ri[:]
}

nine_pixel_params["s_max_n"]=s_max_plus__vec__ri[index_finder(default_neuron_params['ib_n'],ib__list__ri[:])]


default_neuron_params["s_max_n"]=s_max_plus__vec__ri[index_finder(default_neuron_params['ib_n'],ib__list__ri[:])]


net_args = {
    # "N":100,
    # "ns": 100,
    "connectivity": "random",
    "in_connect": "ordered",
    "recurrence": None,
    # "sim": 500,
    "input_p": 1,
    "reservoir_p":0.2,

    "beta_di": 2*np.pi*1e2,
    "tau_di": [1,2], #[900,1100],
    "ib": 9, # int 0-9 to draw from ib__list__ri[i] list
    # "s_max":,
    # "phi_th":,
    "ib_n": 9, # int 0-9 to draw from ib__list__ri[i] list
    "s_th_factor_n": 0.1,
    # "s_max_n":,
    # "phi_th_n":,
    "beta_ni": 2*np.pi*1e3,
    "tau_ni": 50,

    "w_sd": [2.5],
    "w_sid": [2.5], # two numbers for rand float range or single value for consant
    "w_dn": [.75], # two numbers for rand float range or single value for consant
    "norm_dn": 1,
    "norm_sd": 1,

    "beta_ni": 2*np.pi*1e3,
    "tau_ni": 50,
    "ib_ref": 8, # int 0-9 to draw from ib__list__ri[i] list
    "beta_ref": 2*np.pi*1e4,
    "tau_ref": 500,
    "dt_soen": 1, # simulation time-step
    "_t_on": 5,

}