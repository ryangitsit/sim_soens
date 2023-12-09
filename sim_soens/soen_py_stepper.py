import numpy as np
import time


def net_step(net,tau_vec,d_tau):
    '''
    Time stepper for SOEN simulation
        - Can implement hardware in the loop for error and corrections
        - Steps through time tf/dt
        - At each time step, all elements (neurons, dendrites, synapses) updated
        - If any somatic dendrite crosses firing threshold
            - add spikes to neuron
            - send spikes to downstream neuron in the form of new input
    '''   
    if net.timer==True:
        _t0 = time.time()

    HW=None

    for ii in range(len(tau_vec)-1):
            
        # step through neurons
        for node in net.nodes:

            neuron = node.neuron

            # update all input synapses and dendrites       
            for dend in node.dendrite_list:
                # if hasattr(dend,'is_soma') and dend.threshold_flag == True:
                    
                dendrite_updater(dend,ii,tau_vec[ii+1],d_tau,HW)

            # update all output synapses
            output_synapse_updater(neuron,ii,tau_vec[ii+1])
            
            neuron = spike(neuron,ii,tau_vec,net.time_params['dt'])
                       
    if net.timer==True:
        _t1 = time.time()
        print(f'\nSimulation completed in time = {(_t1-_t0)} seconds \n')
        
    return net


def spike(neuron,ii,tau_vec,dt):
    # check if neuron integration loop has increased above threshold
    if neuron.dend_soma.s[ii+1] >= neuron.integrated_current_threshold:
        
        neuron.dend_soma.threshold_flag = True
        neuron.dend_soma.spike_times = np.append(
            neuron.dend_soma.spike_times,
            tau_vec[ii+1]
            )
        neuron.spike_times.append(tau_vec[ii+1])
        neuron.spike_indices.append(ii+1)
        
        # add spike to refractory dendrite
        neuron.dend__ref.synaptic_inputs[
            f'{neuron.name}__syn_refraction'
            ].spike_times_converted = np.append(
            neuron.dend__ref.synaptic_inputs[
            f'{neuron.name}__syn_refraction'
            ].spike_times_converted,
            tau_vec[ii+1]
            )

        # add spike to output synapses
        if neuron.source_type == 'qd' or neuron.source_type == 'ec':
            syn_out = neuron.synaptic_outputs
            num_samples = neuron.num_photons_out_factor*len(syn_out)
            random_numbers = np.random.default_rng().random(size = num_samples)
            
            photon_delay_tau_vec = np.zeros([num_samples])
            for qq in range(num_samples):
                lst = neuron.electroluminescence_cumulative_vec[:]
                val = random_numbers[qq]
                photon_delay_tau_vec[qq] = neuron.time_params[
                    'tau_vec__electroluminescence'
                    ][closest_index(lst,val)]
                                    
            # assign photons to synapses
            for synapse_name in syn_out:
                syn_out[synapse_name].photon_delay_times__temp = []
                
            while len(photon_delay_tau_vec) > 0:
                
                for synapse_name in syn_out:
                    # print(photon_delay_tau_vec[0]/779.5556478344771)
                    syn_out[synapse_name].photon_delay_times__temp.append(
                        photon_delay_tau_vec[0]
                        )
                    photon_delay_tau_vec = np.delete(photon_delay_tau_vec, 0)
                # print(syn_out[synapse_name].photon_delay_times__temp)
            for synapse_name in syn_out:
                # lst = tau_vec[ii+1]
                # val = tau_vec[ii+1] + np.min(
                #     syn_out[synapse_name].photon_delay_times__temp
                #     )
                # _ind = closest_index(lst,val)
                _ind = int(ii+10/dt)#closest_index(lst,val)
                if _ind < len(tau_vec)-1:

                    t_spk = tau_vec[_ind]
                    # a prior spd event has occurred at this synapse                        
                    if len(syn_out[synapse_name].spike_times_converted) > 0:
                        # the spd has had time to recover 
                        if (t_spk - syn_out[synapse_name].spike_times_converted[-1] >= 
                            syn_out[synapse_name].spd_reset_time_converted):                               
                            syn_out[synapse_name].spike_times_converted = np.append(
                                syn_out[synapse_name].spike_times_converted,
                                t_spk
                                )
                    # a prior spd event has not occurred at this synapse
                    else: 
                        syn_out[synapse_name].spike_times_converted = np.append(
                            syn_out[synapse_name].spike_times_converted,
                            t_spk
                            )
                                        
        elif neuron.source_type == 'delay_delta':
            lst = tau_vec[:]
            val = tau_vec[ii+1] + neuron.light_production_delay
            _ind = closest_index(lst,val)
            for synapse_name in syn_out:
                syn_out[synapse_name].spike_times_converted = np.append(
                    syn_out[synapse_name].spike_times_converted,
                    tau_vec[_ind]
                    )
                
    return neuron


def dendrite_updater(dend_obj,time_index,present_time,d_tau,HW=None):
    
    # make sure dendrite isn't a soma that reached threshold
    if hasattr(dend_obj, 'is_soma'):
        if dend_obj.threshold_flag == True:
            update = False
            # wait for absolute refractory period before resetting soma
            if (present_time - dend_obj.spike_times[-1] 
                > dend_obj.absolute_refractory_period_converted): 
                dend_obj.threshold_flag = False # reset threshold flag
        else: 
            update = True
    else:
        update = True
                        
    # directly applied flux
    dend_obj.phi_r[time_index+1] = dend_obj.phi_r_external__vec[time_index+1]
    
    # applied flux from dendrites
    for dendrite_key in dend_obj.dendritic_inputs:
        dend_obj.phi_r[time_index+1] += (
            dend_obj.dendritic_inputs[dendrite_key].s[time_index] * 
            dend_obj.dendritic_connection_strengths[dendrite_key]
            )  
        # if hasattr(dend_obj, 'is_soma') and time_index == 250:
        #     print(dendrite_key,dend_obj.dendritic_connection_strengths[dendrite_key])

    # self-feedback
    dend_obj.phi_r[time_index+1] += (
        dend_obj.self_feedback_coupling_strength * 
        dend_obj.s[time_index]
        )
    
    # applied flux from synapses
    for synapse_key in dend_obj.synaptic_inputs:
        syn_obj = dend_obj.synaptic_inputs[synapse_key]

        # find most recent spike time for this synapse
        _st_ind = np.where( present_time > syn_obj.spike_times_converted[:] )[0]

        if len(_st_ind) > 0:
            
            _st_ind = int(_st_ind[-1])
            if ( syn_obj.spike_times_converted[_st_ind] <= present_time # spike in past
                and (present_time - syn_obj.spike_times_converted[_st_ind]) < 
                syn_obj.spd_duration_converted  # spike within a relevant duration                
                ):
                    _dt_spk = present_time - syn_obj.spike_times_converted[_st_ind]
                    
                    _phi_spd = spd_response(syn_obj.phi_peak, 
                                            syn_obj.tau_rise_converted,
                                            syn_obj.tau_fall_converted,
                                            syn_obj.hotspot_duration_converted, 
                                            _dt_spk)

                    # to avoid going too low when a new spike comes in
                    if _st_ind - syn_obj._st_ind_last == 1:
                        _phi_spd = np.max([_phi_spd,syn_obj.phi_spd[time_index]])
                        syn_obj._phi_spd_memory = _phi_spd
                    if _phi_spd < syn_obj._phi_spd_memory:
                        syn_obj.phi_spd[time_index+1] = syn_obj._phi_spd_memory
                    else:
                        syn_obj.phi_spd[time_index+1] = (
                            _phi_spd * 
                            dend_obj.synaptic_connection_strengths[synapse_key]
                            )

                        syn_obj._phi_spd_memory = 0
                
            syn_obj._st_ind_last = _st_ind
                    
        dend_obj.phi_r[time_index+1] += syn_obj.phi_spd[time_index+1]
        

    new_bias=dend_obj.bias_current

    # find appropriate rate array indices
    lst = dend_obj.phi_r__vec[:] # old way    
    val = dend_obj.phi_r[time_index+1] 


    if val > np.max(dend_obj.phi_r__vec[:]):
        # print("High roll")
        val = val - np.max(dend_obj.phi_r__vec[:])
    elif val < np.min(dend_obj.phi_r__vec[:]):
        # print("Low roll")
        val = val - np.min(dend_obj.phi_r__vec[:])
    
    _ind__phi_r = closest_index(lst,val) 

    i_di__vec = np.asarray(dend_obj.i_di__subarray[_ind__phi_r]) # old way


    # if dend_obj.pri == True:
    #     lst = i_di__vec[:]
    #     val = 2.7 - dend_obj.bias_current + dend_obj.s[time_index]
    #     _ind__s = closest_index(lst,val)
    # elif dend_obj.loops_present=='ri':
    #     lst = i_di__vec[:]
    #     val = (dend_obj.ib_max-new_bias+dend_obj.s[time_index])
    #     _ind__s = closest_index(lst,val)
    # else:

    lst =i_di__vec[:]
    val = dend_obj.s[time_index]
    _ind__s = closest_index(lst,val)
        
    dend_obj.ind_phi.append(_ind__phi_r) # temp
    dend_obj.ind_s.append(_ind__s) # temp

    r_fq = dend_obj.r_fq__subarray[_ind__phi_r][_ind__s] # old way 

    # update the signal of the dendrite
    if update == True:
        dend_obj.s[time_index+1] = dend_obj.s[time_index] * ( 
            1 - d_tau*dend_obj.alpha/dend_obj.beta
            ) + (d_tau/dend_obj.beta) * r_fq

    return


def output_synapse_updater(neuron_object,time_index,present_time):
    
    for synapse_key in neuron_object.synaptic_outputs:
        syn_out = neuron_object.synaptic_outputs[synapse_key]
        # find most recent spike time for this synapse
        _st_ind = np.where( present_time > syn_out.spike_times_converted[:] )[0]
        
        if len(_st_ind) > 0:
            # print(synapse_key)
            _st_ind = int(_st_ind[-1])
            if ( syn_out.spike_times_converted[_st_ind] <= present_time 
                and (present_time - syn_out.spike_times_converted[_st_ind]) < 
                syn_out.spd_duration_converted ): # the case that counts    
                _dt_spk = present_time - syn_out.spike_times_converted[_st_ind]
                _phi_spd = spd_response( syn_out.phi_peak, 
                                        syn_out.tau_rise_converted,
                                        syn_out.tau_fall_converted,
                                        syn_out.hotspot_duration_converted, _dt_spk)
                    
                # to avoid going too low when a new spike comes in
                if _st_ind - syn_out._st_ind_last == 1: 
                    _phi_spd = np.max( [ _phi_spd , syn_out.phi_spd[time_index] ])
                    syn_out._phi_spd_memory = _phi_spd
                if _phi_spd < syn_out._phi_spd_memory:
                    syn_out.phi_spd[time_index+1] = syn_out._phi_spd_memory
                else:
                    # * neuron_object.synaptic_connection_strengths[synapse_key]
                    syn_out.phi_spd[time_index+1] = _phi_spd 
                    syn_out._phi_spd_memory = 0
                
            syn_out._st_ind_last = _st_ind
                        
    return

def closest_index(lst,val):
    return (np.abs(lst-val)).argmin()

def spd_response(phi_peak,tau_rise,tau_fall,hotspot_duration,t):

    if t <= hotspot_duration:
        phi = phi_peak * ( 
            1 - tau_rise/tau_fall 
            ) * ( 1 - np.exp( -t / tau_rise ) )
    elif t > hotspot_duration:
        phi = phi_peak * ( 
            1 - tau_rise/tau_fall 
            ) * (
            1 - np.exp( -hotspot_duration / tau_rise )
            ) * np.exp( -( t - hotspot_duration ) / tau_fall )
    
    return phi

def spd_static_response(phi_peak,tau_rise,tau_fall,hotspot_duration,t):
    '''
    Rewrite time stepper to reference one static spd response by time offeset
    '''
    print(hotspot_duration)
    if t <= hotspot_duration:
        phi = phi_peak * (
            1 - tau_rise/tau_fall
            ) * ( 1 - np.exp( -t / tau_rise ) )
    elif t > hotspot_duration:
        phi = phi_peak * ( 
            1 - tau_rise/tau_fall 
            ) * ( 
            1 - np.exp( -hotspot_duration / tau_rise ) 
            ) * np.exp( -( t - hotspot_duration ) / tau_fall )
    
    return phi