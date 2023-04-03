import numpy as np
import time


from numpy.random import default_rng
rng = default_rng()

from .soen_initialize import (
    dendrite_drive_construct,
    rate_array_attachment,
    synapse_initialization,
    output_synapse_initialization,
    transmitter_initialization,
    dendrite_data_attachment
)

np.random.seed(10)
def run_soen_sim(net):
    '''
    Runs SOEN simulation
    - Initializes simulation parameters through soen_intitialize.py
    - Runs simulation through net_step()
    '''
    time_vec = np.arange(0,net.tf+net.dt,net.dt)          
        
    # network
    if type(net).__name__ == 'network':
        
        # convert to dimensionless time
        net.time_params = {
            'dt': net.dt, 
            'tf': net.tf, 
            'time_vec': time_vec, 
            't_tau_conversion': 1e-9/net.jj_params['tau_0']
            }
        t_tau_conversion = net.time_params['t_tau_conversion']
        tau_vec = time_vec*t_tau_conversion
        d_tau = net.time_params['dt']*t_tau_conversion
        net.time_params.update({'tau_vec': tau_vec, 'd_tau': d_tau})
        
        # interate through all network nodes and initialize all related elements
        for node in net.nodes:
            # print("Initializing neuron: ", neuron.name)
            node.neuron.time_params = net.time_params
            node.neuron.dend_soma.threshold_flag = False

            for dend in node.dendrite_list:
                # print(" Initializing dendrite: ", dend.name)
                dendrite_drive_construct(dend,tau_vec,t_tau_conversion,d_tau)
                rate_array_attachment(dend)
                synapse_initialization(dend,tau_vec,t_tau_conversion)

            output_synapse_initialization(node.neuron,tau_vec,t_tau_conversion)
            transmitter_initialization(node.neuron,t_tau_conversion)

        # run the simulation one time step at a time
        net = net_step(net,tau_vec,d_tau)

        # attach results to dendrite objects
        for node in net.nodes:
            for dend in node.dendrite_list:
                dendrite_data_attachment(dend,net)
        
    # formerly, there were unique sim methods for each element
    else:
        print('''
        Error: Simulations no longer supported for individual components.
                --> Instead run component in the context of a network
        ''')

    return net



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


    if "hardware" in net.__dict__.keys():
        print("Hardware in the loop.")
        HW = net.hardware
        HW.conversion = net.time_params['t_tau_conversion']
    else:
        # print("No hardware in the loop.")
        net.hardware=None
        HW=None
    if net.timer==True:
        _t0 = time.time()
    for ii in range(len(tau_vec)-1):

        if net.hardware:
            # print("BACKWARD ERROR")
            if ii == HW.check_time/net.dt:
                HW.forward_error(net.nodes)
                HW.backward_error(net.nodes)
            
        # step through neurons
        for node in net.nodes:

            neuron = node.neuron

            # update all input synapses and dendrites       
            for dend in node.dendrite_list:
                # if hasattr(dend,'is_soma') and dend.threshold_flag == True:
                    
                dendrite_updater(dend,ii,tau_vec[ii+1],d_tau,HW)

            # update all output synapses
            output_synapse_updater(neuron,ii,tau_vec[ii+1])
            
            neuron = spike(neuron,ii,tau_vec)
                       
    if net.timer==True:
        _t1 = time.time()
        print(f'\nSimulation completed in time = {(_t1-_t0)} seconds \n')
        
    return net

def spike(neuron,ii,tau_vec):
    # check if neuron integration loop has increased above threshold
    if neuron.dend_soma.s[ii+1] >= neuron.integrated_current_threshold:
        
        neuron.dend_soma.threshold_flag = True
        neuron.dend_soma.spike_times.append(tau_vec[ii+1])
        neuron.spike_times.append(tau_vec[ii+1])
        neuron.spike_indices.append(ii+1)
        
        # add spike to refractory dendrite
        neuron.dend__ref.synaptic_inputs[f'{neuron.name}__syn_refraction'].spike_times_converted = np.append(
            neuron.dend__ref.synaptic_inputs[f'{neuron.name}__syn_refraction'].spike_times_converted,
            tau_vec[ii+1]
            )
        
        # if neuron.second_ref == True:
        #     neuron.dend__ref_2.synaptic_inputs...

        # add spike to output synapses
        if neuron.source_type == 'qd' or neuron.source_type == 'ec':

            num_samples = neuron.num_photons_out_factor*len(neuron.synaptic_outputs)
            random_numbers = rng.random(size = num_samples)
            
            photon_delay_tau_vec = np.zeros([num_samples])
            for qq in range(num_samples):
                lst = neuron.electroluminescence_cumulative_vec[:]
                val = random_numbers[qq]
                photon_delay_tau_vec[qq] = neuron.time_params['tau_vec__electroluminescence'][closest_index(lst,val)]
                                    
            # assign photons to synapses
            for synapse_name in neuron.synaptic_outputs:
                neuron.synaptic_outputs[synapse_name].photon_delay_times__temp = []
                
            while len(photon_delay_tau_vec) > 0:
                
                for synapse_name in neuron.synaptic_outputs:
                    neuron.synaptic_outputs[synapse_name].photon_delay_times__temp.append(photon_delay_tau_vec[0])
                    photon_delay_tau_vec = np.delete(photon_delay_tau_vec, 0)
                    
            for synapse_name in neuron.synaptic_outputs:
                lst = tau_vec[ii+1]
                val = tau_vec[ii+1] + np.min(neuron.synaptic_outputs[synapse_name].photon_delay_times__temp)
                _ind = closest_index(lst,val)
                # a prior spd event has occurred at this synapse                        
                if len(neuron.synaptic_outputs[synapse_name].spike_times_converted) > 0:
                    # the spd has had time to recover 
                    if (tau_vec[_ind] - neuron.synaptic_outputs[synapse_name].spike_times_converted[-1] >= 
                        neuron.synaptic_outputs[synapse_name].spd_reset_time_converted):                               
                        neuron.synaptic_outputs[synapse_name].spike_times_converted = np.append(
                            neuron.synaptic_outputs[synapse_name].spike_times_converted,
                            tau_vec[_ind]
                            )
                # a prior spd event has not occurred at this synapse
                else: 
                    neuron.synaptic_outputs[synapse_name].spike_times_converted = np.append(
                        neuron.synaptic_outputs[synapse_name].spike_times_converted,
                        tau_vec[_ind]
                        )
                                        
        elif neuron.source_type == 'delay_delta':
            lst = tau_vec[:]
            val = tau_vec[ii+1] + neuron.light_production_delay
            _ind = closest_index(lst,val)
            for synapse_name in neuron.synaptic_outputs:
                neuron.synaptic_outputs[synapse_name].spike_times_converted = np.append(
                    neuron.synaptic_outputs[synapse_name].spike_times_converted,
                    tau_vec[_ind]
                    )
                
    return neuron

def dendrite_updater(dend_obj,time_index,present_time,d_tau,HW=None):
    
    # make sure dendrite isn't a soma that reached threshold
    if hasattr(dend_obj, 'is_soma'):
        if dend_obj.threshold_flag == True:
            update = False
            # wait for absolute refractory period before resetting soma
            if present_time - dend_obj.spike_times[-1] > dend_obj.absolute_refractory_period_converted: 
                dend_obj.threshold_flag = False # reset threshold flag
        else: 
            update = True
    else:
        update = True
                        
    # directly applied flux
    dend_obj.phi_r[time_index+1] = dend_obj.phi_r_external__vec[time_index+1]

    # applied flux from dendrites
    for dendrite_key in dend_obj.dendritic_inputs:
        dend_obj.phi_r[time_index+1] += dend_obj.dendritic_inputs[dendrite_key].s[time_index] * dend_obj.dendritic_connection_strengths[dendrite_key]        


    # self-feedback
    dend_obj.phi_r[time_index+1] += dend_obj.self_feedback_coupling_strength * dend_obj.s[time_index]
    
    # applied flux from synapses
    for synapse_key in dend_obj.synaptic_inputs:
        # print(dend_obj.synaptic_inputs[synapse_key])
        # find most recent spike time for this synapse
        _st_ind = np.where( present_time > dend_obj.synaptic_inputs[synapse_key].spike_times_converted[:] )[0]
        
        if len(_st_ind) > 0:
            
            _st_ind = int(_st_ind[-1])
            if ( dend_obj.synaptic_inputs[synapse_key].spike_times_converted[_st_ind] <= present_time # spike in past
                and (present_time - dend_obj.synaptic_inputs[synapse_key].spike_times_converted[_st_ind]) < 
                dend_obj.synaptic_inputs[synapse_key].spd_duration_converted  # spike within a relevant duration                
                ):
                    _dt_spk = present_time - dend_obj.synaptic_inputs[synapse_key].spike_times_converted[_st_ind]
                    _phi_spd = spd_response(dend_obj.synaptic_inputs[synapse_key].phi_peak, 
                                            dend_obj.synaptic_inputs[synapse_key].tau_rise_converted,
                                            dend_obj.synaptic_inputs[synapse_key].tau_fall_converted,
                                            dend_obj.synaptic_inputs[synapse_key].hotspot_duration_converted, 
                                            _dt_spk)

                    # to avoid going too low when a new spike comes in
                    if _st_ind - dend_obj.synaptic_inputs[synapse_key]._st_ind_last == 1: 
                        _phi_spd = np.max( [ _phi_spd , dend_obj.synaptic_inputs[synapse_key].phi_spd[time_index] ])
                        dend_obj.synaptic_inputs[synapse_key]._phi_spd_memory = _phi_spd
                    if _phi_spd < dend_obj.synaptic_inputs[synapse_key]._phi_spd_memory:
                        dend_obj.synaptic_inputs[synapse_key].phi_spd[time_index+1] = dend_obj.synaptic_inputs[synapse_key]._phi_spd_memory
                    else:
                        dend_obj.synaptic_inputs[synapse_key].phi_spd[time_index+1] = _phi_spd * dend_obj.synaptic_connection_strengths[synapse_key]

                        # dend_obj.synaptic_connection_strengths[synapse_key] = dend_obj.synaptic_connection_strengths[synapse_key]*2
                        # print(dend_obj.synaptic_connection_strengths[synapse_key])

                        dend_obj.synaptic_inputs[synapse_key]._phi_spd_memory = 0
                
            dend_obj.synaptic_inputs[synapse_key]._st_ind_last = _st_ind
                    
        dend_obj.phi_r[time_index+1] += dend_obj.synaptic_inputs[synapse_key].phi_spd[time_index+1]
        
    # for counting moments any types of flex rollover
    # if np.abs(dend_obj.phi_r[time_index+1]) > .5:
    #     dend_obj.rollover+=1
    #     if np.abs(dend_obj.phi_r[time_index+1]) > 1:
    #         dend_obj.valleyedout+=1
    #         if np.abs(dend_obj.phi_r[time_index+1]) > 1.5:
    #             dend_obj.doubleroll+=1

    new_bias=dend_obj.bias_current
    # if 'ib_ramp' in list(dend_obj.__dict__.keys()):
    #     if dend_obj.ib_ramp == True:
    #         new_bias= 1.4 + (dend_obj.ib_max-1.4)*time_index/dend_obj.time_steps

    if HW:

        if 'trace' not in dend_obj.name:

            for trace in HW.traces:
                if dend_obj.name == list(trace.dendritic_inputs.keys())[0]:
                    # print(dend_obj.name, list(trace.dendritic_inputs.keys())[0], trace.name)

                    if "minus" in trace.name:
                        if trace.s[time_index] > 0:
                            new_bias = (1-trace.s[time_index]) * (dend_obj.ib_max-.99) + HW.baseline
                        # if time_index == 2500 or time_index == 7500: 
                        #     print("minus",trace.name,dend_obj.name,new_bias)

                    elif "plus" in trace.name:
                        if trace.s[time_index] > 0:
                            new_bias = trace.s[time_index] * (dend_obj.ib_max-.99) + HW.baseline
                        # if time_index == 2500 or time_index == 7500: 
                        #     print("plus",trace.name,dend_obj.name,new_bias)

                # if (time_index == 2500 or time_index == 7500): 
                    # print("BIAS: ",dend_obj.name,new_bias)

                dend_obj.bias_current = new_bias
                HW.trace_biases[trace.name].append(new_bias)

        # track how bias changes over time
        dend_obj.bias_dynamics.append(new_bias)

    # find appropriate rate array indices
    lst = dend_obj.phi_r__vec[:]
    val = dend_obj.phi_r[time_index+1] 
    _ind__phi_r = closest_index(lst,val)
    i_di__vec = np.asarray(dend_obj.i_di__subarray[_ind__phi_r])

    if dend_obj.pri == True:
        lst = i_di__vec[:]
        val = 2.7 - dend_obj.bias_current + dend_obj.s[time_index]
        _ind__s = closest_index(lst,val)
    elif dend_obj.loops_present=='ri':
        lst = i_di__vec[:]
        val = (dend_obj.ib_max-new_bias+dend_obj.s[time_index])
        _ind__s = closest_index(lst,val)
    else:
        lst =i_di__vec[:]
        val = dend_obj.s[time_index]
        _ind__s = closest_index(lst,val)
    r_fq = dend_obj.r_fq__subarray[_ind__phi_r][_ind__s]
        
    # get alpha 
    # skip this if/else
    # if hasattr(dend_obj,'alpha_list'):
    #     _ind = np.where(dend_obj.s_list > dend_obj.s[time_index])
    #     alpha = dend_obj.alpha_list[_ind[0][0]]
    # else:
    #     alpha = dend_obj.alpha    

    # update the signal of the dendrite
    if update == True:
        dend_obj.s[time_index+1] = dend_obj.s[time_index] * ( 1 - d_tau*dend_obj.alpha/dend_obj.beta) + (d_tau/dend_obj.beta) * r_fq

    return


def output_synapse_updater(neuron_object,time_index,present_time):
    
    for synapse_key in neuron_object.synaptic_outputs:
        
        # find most recent spike time for this synapse
        _st_ind = np.where( present_time > neuron_object.synaptic_outputs[synapse_key].spike_times_converted[:] )[0]
        
        if len(_st_ind) > 0:
            
            _st_ind = int(_st_ind[-1])
            if ( neuron_object.synaptic_outputs[synapse_key].spike_times_converted[_st_ind] <= present_time 
                and (present_time - neuron_object.synaptic_outputs[synapse_key].spike_times_converted[_st_ind]) < 
                neuron_object.synaptic_outputs[synapse_key].spd_duration_converted ): # the case that counts    
                _dt_spk = present_time - neuron_object.synaptic_outputs[synapse_key].spike_times_converted[_st_ind]
                _phi_spd = spd_response( neuron_object.synaptic_outputs[synapse_key].phi_peak, 
                                        neuron_object.synaptic_outputs[synapse_key].tau_rise_converted,
                                        neuron_object.synaptic_outputs[synapse_key].tau_fall_converted,
                                        neuron_object.synaptic_outputs[synapse_key].hotspot_duration_converted, _dt_spk)
                    
                # to avoid going too low when a new spike comes in
                if _st_ind - neuron_object.synaptic_outputs[synapse_key]._st_ind_last == 1: 
                    _phi_spd = np.max( [ _phi_spd , neuron_object.synaptic_outputs[synapse_key].phi_spd[time_index] ])
                    neuron_object.synaptic_outputs[synapse_key]._phi_spd_memory = _phi_spd
                if _phi_spd < neuron_object.synaptic_outputs[synapse_key]._phi_spd_memory:
                    neuron_object.synaptic_outputs[synapse_key].phi_spd[time_index+1] = neuron_object.synaptic_outputs[synapse_key]._phi_spd_memory
                else:
                    # * neuron_object.synaptic_connection_strengths[synapse_key]
                    neuron_object.synaptic_outputs[synapse_key].phi_spd[time_index+1] = _phi_spd 
                    neuron_object.synaptic_outputs[synapse_key]._phi_spd_memory = 0
                
            neuron_object.synaptic_outputs[synapse_key]._st_ind_last = _st_ind
                        
    return

def closest_index(lst,val):
    return (np.abs(lst-val)).argmin()

def spd_response(phi_peak,tau_rise,tau_fall,hotspot_duration,t):
        
    if t <= hotspot_duration:
        phi = phi_peak * ( 1 - tau_rise/tau_fall ) * ( 1 - np.exp( -t / tau_rise ) )
    elif t > hotspot_duration:
        phi = phi_peak * ( 1 - tau_rise/tau_fall ) * ( 1 - np.exp( -hotspot_duration / tau_rise ) ) * np.exp( -( t - hotspot_duration ) / tau_fall )
    
    return phi

def spd_static_response(phi_peak,tau_rise,tau_fall,hotspot_duration,t):
    '''
    Rewrite time stepper to reference one static spd response by time offeset
    '''
    if t <= hotspot_duration:
        phi = phi_peak * ( 1 - tau_rise/tau_fall ) * ( 1 - np.exp( -t / tau_rise ) )
    elif t > hotspot_duration:
        phi = phi_peak * ( 1 - tau_rise/tau_fall ) * ( 1 - np.exp( -hotspot_duration / tau_rise ) ) * np.exp( -( t - hotspot_duration ) / tau_fall )
    
    return phi