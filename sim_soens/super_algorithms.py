import numpy as np
import time

from sim_soens.super_node import SuperNode


def make_symmetric_update(nodes,eta,errors,max_offset):
    for n,node in enumerate(nodes):
        if errors[n] != 0:
            for dend in node.dendrite_list[2:]:
                if dend.output_connection_strength < 0: 
                    update_sign = -1
                else:
                    update_sign = 1    
                update = np.mean(dend.s)*errors[n]*eta*update_sign
                # dend.offset_flux += update
                step = update
                # print("--",dend.offset_flux,dend.offset_flux+step)
                if max_offset != "None":
                    if max_offset=='phi_off':
                        max_off = dend.phi_th 
                    elif max_offset=='half':
                        max_off=0.5
                    elif max_offset=='inverse':
                        max_off=0.5-dend.phi_th
                    else:
                        print(f"Invalid maximum offset.")

                    # print(f"max offset == {dend.phi_th}")
                    if np.abs(step+dend.offset_flux) > max_off: 
                        # print(dend.offset_flux,dend.offset_flux+step)
                        #*** what about offset==0?
                        if step < 0:
                            dend.offset_flux = np.max([dend.offset_flux+step,-max_off])
                        else:
                            dend.offset_flux = np.min([dend.offset_flux+step,max_off])
                    else:
                        dend.offset_flux += step
                else:
                    dend.offset_flux += step



def make_choosing_update(nodes,eta,errors):
    for n,node in enumerate(nodes):
        if errors[n] != 0:
            error = errors[n]
            for dend in node.dendrite_list[2:]:
                if 'lay7' in dend.name:
                    if dend.output_connection_strength < 0 and error<0: 
                        update = np.mean(dend.s)*error*eta*-1
                    elif dend.output_connection_strength > 0 and error>0: 
                        update = np.mean(dend.s)*error*eta
                    else: 
                        update = 0
                else:
                    update = np.mean(dend.s)*error*eta

                dend.offset_flux+=update



def arbor_update(nodes,config,digit,sample,errors,updater):
    '''
    Updates all dendrites (except refractory) according to the arbor update rule
     - Paper on the arbor update rule: https://dl.acm.org/doi/abs/10.1145/3589737.3605972
    '''

    offset_sums = [0 for _ in range(config.digits)]
    max_hits = np.zeros(config.digits)
        

    if updater=="classic":
        # print(config.max_offset)
        s = time.perf_counter()
        # if config.inh_counter: print("inh counter")
        for n,node in enumerate(nodes):
            if errors[n] != 0:
                for l,layer in enumerate(node.dendrites):
                    for g,group in enumerate(layer):
                        for d,dend in enumerate(group):
                            if 'ref' not in dend.name and 'soma' not in dend.name:
                                if config.double_dends != True:
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
                                        print("INH COUNTER")
                                        if dend.downstream_inhibition%2!=0:
                                            step = -step
                                        
                                    dend.offset_flux += step

                                else:
                                    if 'lay7' in dend.name:
                                        if dend.output_connection_strength < 0: 
                                            update_sign = -1
                                        else:
                                            update_sign = 1

                                        if np.mean(dend.s) > 0:
                                            # print(f"Signal based update -> {dend.name}")
                                            update = np.mean(dend.s)*errors[n]*config.eta*update_sign
                                            # if n==0:
                                            #     if update_sign==1: 
                                            #         print(f"Positive signal based update -> {dend.name, dend.offset_flux} <= {update}")
                                            #         print(f"  {np.mean(dend.s),errors[n],config.eta,update_sign}")
                                            #     else:
                                            #         print(f"Negative signal based update -> {dend.name, dend.offset_flux} <= {update}")

                                        else: 
                                            # print(f"Void based update -> {dend.name}")
                                            update = 0 # errors[n]*config.eta*update_sign*-1*.1
                                    else:
                                        # print(f"Arbor based update -> {dend.name}")
                                        update = errors[n]*np.mean(dend.s)*config.eta
                                    step = update                  
                                # if dend.name == 'node_0_lay7_branch329_den1':
                                # # if np.mean(dend.s) > 0 and update_sign == 1 and dend.offset_flux > 0: 
                                #     print(dend.name)
                                #     print(f"                                                            ",
                                #           f"offset = {dend.offset_flux}  ::  signal = {np.mean(dend.s)} :: error = {errors[n]} :: eta = {config.eta}  ::  sign = {update_sign}  ::  update = {update}")

                                if config.max_offset != "None":
                                    if config.max_offset=='phi_off':
                                        max_off = dend.phi_th 
                                    elif config.max_offset=='half':
                                        max_off=0.5
                                    elif config.max_offset=='inverse':
                                        max_off=0.5-dend.phi_th
                                    else:
                                        print(f"Invalid maximum offset.")

                                    # print(f"max offset == {dend.phi_th}")
                                    if np.abs(step+dend.offset_flux) > max_off: 
                                        old = dend.offset_flux
                                        max_hits[n]+=1
                                        #*** what about offset==0?
                                        if step < 0:
                                            dend.offset_flux = np.max([dend.offset_flux+step,-max_off])
                                        else:
                                            dend.offset_flux = np.min([dend.offset_flux+step,max_off])
                                        offset_sums[n] += dend.offset_flux - old 
                                    else:
                                        offset_sums[n] += step
                                else:
                                    dend.offset_flux += step
                                    offset_sums[n] += step

                            dend.s = []
                            dend.phi_r = []
            else:
                for dend in node.dendrite_list:
                    dend.s = []
                    dend.phi_r = []
    elif updater == "symmetric":
        make_symmetric_update(nodes,config.eta,errors,config.max_offset)
    elif updater == "chooser":
        make_choosing_update(nodes,config.eta,errors)

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