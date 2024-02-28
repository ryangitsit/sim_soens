import numpy as np
import time

from sim_soens.super_node import SuperNode

def arbor_update(nodes,config,digit,sample,errors):
    '''
    Updates all dendrites (except refractory) according to the arbor update rule
     - Paper on the arbor update rule: https://dl.acm.org/doi/abs/10.1145/3589737.3605972
    '''
    # print(config.max_offset)
    s = time.perf_counter()
    offset_sums = [0 for _ in range(config.digits)]
    max_hits = np.zeros(config.digits)
    
    # if config.inh_counter: print("inh counter")
    for n,node in enumerate(nodes):
        for l,layer in enumerate(node.dendrites):
            for g,group in enumerate(layer):
                for d,dend in enumerate(group):
                    if 'ref' not in dend.name and 'soma' not in dend.name:
                        
                        if hasattr(dend, 'hebb'):
                            hebb = dend.hebb*20
                        else:
                            hebb = 1

                        step = errors[n]*np.mean(dend.s)*config.eta*hebb
                        flux = np.mean(dend.phi_r) + step
                        
                        if config.hebbian == "True":
                            for in_dend in dend.dendritic_inputs.keys():
                                in_dendrite = dend.dendritic_inputs[in_dend]
                                if "ref" not in in_dend:
                                    in_dendrite.hebb = np.mean(dend.s)
                                    # print(np.mean(dend.s))

                        if config.elasticity=="elastic":
                            if flux > 0.5 or flux < config.low_bound:
                                step = -step
        
                        elif config.elasticity=="inelastic":
                            if flux > 0.5 or flux < config.low_bound:
                                step = 0

                        if config.inh_counter:
                            if dend.downstream_inhibition%2!=0:
                                step = -step
                            
                        dend.offset_flux += step
                        

                        if config.max_offset != None:
                            if config.max_offset=='phi_off':
                                max_off = dend.phi_th 
                            elif config.max_offset=='half':
                                max_off=0.5
                            elif config.max_offset=='inverse':
                                max_off=0.5-dend.phi_th
                            else:
                                print(f"Invalid maximum offset.")

                            # print(f"max offset == {dend.phi_th}")
                            if np.abs(dend.offset_flux) > max_off: 
                                old = dend.offset_flux
                                max_hits[n]+=1
                                #*** what about offset==0?
                                if dend.offset_flux < 0:
                                    dend.offset_flux = np.max([dend.offset_flux,-max_off])
                                else:
                                    dend.offset_flux = np.min([dend.offset_flux,max_off])
                                offset_sums[n] += dend.offset_flux - old 
                            else:
                                offset_sums[n] += step

                    dend.s = []
                    dend.phi_r = []
    # if max_hits > 0:
    #     print(f"{max_hits} max_update hits!")
    f = time.perf_counter()
    # print(f"Update time = {f-s}")
    return nodes, offset_sums,max_hits


def probablistic_arbor_update(nodes,config,digit,sample,errors):
    '''
    Makes arbor update rule updates on all dendrites (except refractory) with some probability p
    '''
    s = time.perf_counter()
    offset_sums = [0 for _ in range(config.digits)]
    bool_array = np.random.rand(len(nodes)*len(nodes[0].dendrite_list)) < config.probabilistic
    dend_counter = 0
    if config.elasticity=="elastic":
        if sample == 0 and config.run == 0: print("elastic")
        for n,node in enumerate(nodes):
            for l,layer in enumerate(node.dendrites):
                for g,group in enumerate(layer):
                    for d,dend in enumerate(group):
                        print(bool_array[dend_counter])
                        if bool_array[dend_counter] == True:
                            if 'ref' not in dend.name and 'soma' not in dend.name:
                                step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                                flux = np.mean(dend.phi_r) + step #dend.offset_flux
                                if flux > 0.5 or flux < config.low_bound:
                                    step = -step
                                dend.offset_flux += step
                                offset_sums[n] += step
                        dend.s = []
                        dend.phi_r = []
                        dend_counter += 1

    if config.elasticity=="inelastic":
        if sample == 0 and config.run == 0: print("inealstic")
        for n,node in enumerate(nodes):
            for l,layer in enumerate(node.dendrites):
                for g,group in enumerate(layer):
                    for d,dend in enumerate(group):
                        bool_array[dend_counter]
                        if bool_array[dend_counter] == True:
                            if 'ref' not in dend.name and 'soma' not in dend.name:
                                step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                                flux = np.mean(dend.phi_r) + step #dend.offset_flux
                                if flux > 0.5 or flux < config.low_bound:
                                    step = 0
                                dend.offset_flux += step
                                offset_sums[n] += step
                        dend.s = []
                        dend.phi_r = []
                        dend_counter += 1

    if config.elasticity=="unbounded":
        if sample == 0 and config.run == 0: print("unbounded")
        for n,node in enumerate(nodes):
            for l,layer in enumerate(node.dendrites):
                for g,group in enumerate(layer):
                    for d,dend in enumerate(group):
                        bool_array[dend_counter]
                        if bool_array[dend_counter] == True:
                            if 'ref' not in dend.name and 'soma' not in dend.name:
                                step = errors[n]*np.mean(dend.s)*config.eta #+(2-l)*.001
                                dend.offset_flux += step
                                offset_sums[n] += step #dend.offset_flux
                        dend.s = []
                        dend.phi_r = []
                        dend_counter += 1
    f = time.perf_counter()
    max_hits = 0
    # print(f"Update time = {f-s}")
    return nodes, offset_sums,max_hits