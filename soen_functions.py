import numpy as np
import pickle
import time
import sys

from numpy.random import default_rng
rng = default_rng()

from soen_utilities import dend_load_rate_array, dend_load_arrays_thresholds_saturations, physical_constants, index_finder


ib__list__ri, phi_r__array__ri, i_di__array__ri, r_fq__array__ri, phi_th_plus__vec__ri, phi_th_minus__vec__ri, s_max_plus__vec__ri, s_max_minus__vec__ri, s_max_plus__array__ri, s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')
ib__list__rtti, phi_r__array__rtti, i_di__array__rtti, r_fq__array__rtti, phi_th_plus__vec__rtti, phi_th_minus__vec__rtti, s_max_plus__vec__rtti, s_max_minus__vec__rtti, s_max_plus__array__rtti, s_max_minus__array__rtti = dend_load_arrays_thresholds_saturations('default_rtti')
# ib__list__rtti, phi_r__array__rtti, i_di__array__rtti, r_fq__array__rtti, phi_th_plus__vec__rtti, phi_th_minus__vec__rtti, s_max_plus__vec__rtti, s_max_minus__vec__rtti, s_max_plus__array__rtti, s_max_minus__array__rtti = dend_load_arrays_thresholds_saturations('default_pri')
ib__vec__ri = np.asarray(ib__list__ri[:])
ib__vec__rtti = np.asarray(ib__list__rtti[:])



def run_soen_sim(obj, **kwargs):
    
    # set up time vecx
    if 'dt' in kwargs:
        dt = kwargs['dt']
    else:
        dt = 1 # ns
    if 'tf' in kwargs:
        tf = kwargs['tf']
    else:
        tf = 100
    
    time_vec = np.arange(0,tf+dt,dt) 
    # obj.time_vec = time_vec            
        
    # network
    if type(obj).__name__ == 'network':
        
        # convert to dimensionless time
        obj.time_params = {'dt': dt, 'tf': tf, 'time_vec': time_vec, 't_tau_conversion': 1e-9/obj.jj_params['tau_0']}
        t_tau_conversion = obj.time_params['t_tau_conversion']
        tau_vec = time_vec*t_tau_conversion
        d_tau = obj.time_params['dt']*t_tau_conversion
        obj.time_params.update({'tau_vec': tau_vec, 'd_tau': d_tau})
        
        if obj.new_way==False:
            print("old way")
            # initialize all neurons
            for neuron_key in obj.neurons:
                
                # add time params to each neuron (shouldn't have to do this; do it once for the network)
                obj.neurons[neuron_key].time_params = obj.time_params
                            
                # initialize dendrites (phi_r vector) and make drive signals
                # print('\n')
                recursive_dendrite_initialization_and_drive_construction(obj.neurons[neuron_key].dend__nr_ni,tau_vec,t_tau_conversion,d_tau) # go through all other dendrites in the neuron's arbor
                # print('\n')
                
                # load rate arrays to all dendrites
                recursive_rate_array_attachment(obj.neurons[neuron_key].dend__nr_ni)
                
                # initialize all input synapses
                recursive_synapse_initialization(obj.neurons[neuron_key].dend__nr_ni,tau_vec,t_tau_conversion)
                
                # initialize all output synapses
                output_synapse_initialization(obj.neurons[neuron_key],tau_vec,t_tau_conversion)
                
                # initialize transmitter
                transmitter_initialization(obj.neurons[neuron_key],t_tau_conversion)
                
            # step through time
            # print('\nrunning network time stepper for {:4.2e}ns ({:d} time steps) ...'.format(time_vec[-1],len(tau_vec)))
            obj = network_time_stepper(obj,tau_vec,d_tau)
            
            # add output data to dendrites for plotting and diagnostics
            for neuron_key in obj.neurons:
                recursive_dendrite_data_attachment(obj.neurons[neuron_key].dend__nr_ni,obj)
        else:
            print("new way")
            # for name,neuron in obj.neurons.items():
            for neuron in obj.nodes:
                # print("Initializing neuron: ", neuron.name)
                neuron.neuron.time_params = obj.time_params
                # print(neuron.dendrite_list)
                for dend in neuron.dendrite_list:
                    # print(" Initializing dendrite: ", dend.name)

                    dendrite_init_and_drive_construct(dend,tau_vec,t_tau_conversion,d_tau)
                    rate_array_attachment(dend)
                    synapse_initialization(dend,tau_vec,t_tau_conversion)

                output_synapse_initialization(neuron.neuron,tau_vec,t_tau_conversion)
                transmitter_initialization(neuron.neuron,t_tau_conversion)

            obj = net_step(obj,tau_vec,d_tau)
            for dend in neuron.dendrite_list:
                dendrite_data_attachment(dend,obj)
            # print(obj.neurons)
            # print("Here!")
            
    return obj

def dendrite_init_and_drive_construct(dendrite_object,tau_vec,t_tau_conversion,d_tau):
    # print("  Constructing:",dendrite_object.name)          
    dendrite_object.phi_r_external__vec = np.zeros([len(tau_vec)]) # from external drives
    dendrite_object.phi_r = np.zeros([len(tau_vec)]) # from synapses and dendrites
    dendrite_object.s = np.zeros([len(tau_vec)]) # output variable
    dendrite_object.beta = dendrite_object.circuit_betas[-1]
    
    # add external drives to this dendrite if they're present
    dendrite_object = construct_dendritic_drives(dendrite_object) 
    
    # turn external drives to this dendrite into flux
    dendrite_object.phi_r_external__vec[:] = dendrite_object.offset_flux
    for external_input in dendrite_object.external_inputs:
        dendrite_object.phi_r_external__vec += dendrite_object.external_inputs[external_input].drive_signal * dendrite_object.external_connection_strengths[external_input]
        
    # prepare somas for absolute refractory period
    if hasattr(dendrite_object, 'is_soma'):
        dendrite_object.absolute_refractory_period_converted = dendrite_object.absolute_refractory_period * t_tau_conversion
        
    # normalize inputs
    if dendrite_object.normalize_input_connection_strengths:        
        J_ij_e__init = 0 # excitatory
        J_ij_i__init = 0 # inhibitory
        J_ij_e = dendrite_object.total_excitatory_input_connection_strength
        J_ij_i = dendrite_object.total_inhibitory_input_connection_strength
        # print('J_ij = {}'.format(J_ij))
        for external_input in dendrite_object.external_inputs:
            if dendrite_object.external_connection_strengths[external_input] >= 0:
                J_ij_e__init += dendrite_object.external_connection_strengths[external_input]
            elif dendrite_object.external_connection_strengths[external_input] < 0:
                J_ij_i__init += dendrite_object.external_connection_strengths[external_input]
        for synapse in dendrite_object.synaptic_inputs:
            if dendrite_object.synaptic_connection_strengths[synapse] >= 0:
                J_ij_e__init += dendrite_object.synaptic_connection_strengths[synapse]
            elif dendrite_object.synaptic_connection_strengths[synapse] < 0:
                J_ij_i__init += dendrite_object.synaptic_connection_strengths[synapse]
        for dendrite in dendrite_object.dendritic_inputs:
            if dendrite_object.dendritic_connection_strengths[dendrite] >= 0:
                J_ij_e__init += dendrite_object.dendritic_connection_strengths[dendrite]
            elif dendrite_object.dendritic_connection_strengths[dendrite] < 0:
                if dendrite_object.dendritic_inputs[dendrite].name[-15:] != 'dend_refraction': # make sure this isn't the refractory dendrite. that one doesn't get included in this normalization.
                    J_ij_i__init += dendrite_object.dendritic_connection_strengths[dendrite]
        if J_ij_e__init > 0:
            factor_e = J_ij_e/J_ij_e__init
        else:
            factor_e = 0
        if J_ij_i__init > 0:
            factor_i = J_ij_i/J_ij_i__init
        else:
            factor_i = 0
        # print('J_ij__init = {}'.format(J_ij__init))
        # print('factor = {}'.format(factor))
        for external_input in dendrite_object.external_inputs:
            if dendrite_object.external_connection_strengths[external_input] >= 0:
                dendrite_object.external_connection_strengths[external_input] = factor_e * dendrite_object.external_connection_strengths[external_input]
            elif dendrite_object.external_connection_strengths[external_input] < 0:
                dendrite_object.external_connection_strengths[external_input] = factor_i * dendrite_object.external_connection_strengths[external_input]
        for synapse in dendrite_object.synaptic_inputs:
            if dendrite_object.synaptic_connection_strengths[synapse] >= 0:
                dendrite_object.synaptic_connection_strengths[synapse] = factor_e * dendrite_object.synaptic_connection_strengths[synapse]
            elif dendrite_object.synaptic_connection_strengths[synapse] < 0:
                dendrite_object.synaptic_connection_strengths[synapse] = factor_i * dendrite_object.synaptic_connection_strengths[synapse]
        for dendrite in dendrite_object.dendritic_inputs:
            if dendrite_object.dendritic_connection_strengths[dendrite] >= 0:
                dendrite_object.dendritic_connection_strengths[dendrite] = factor_e * dendrite_object.dendritic_connection_strengths[dendrite]
            elif dendrite_object.dendritic_connection_strengths[dendrite] < 0:
                if dendrite_object.dendritic_inputs[dendrite].name[-15:] != 'dend_refraction': # make sure this isn't the refractory dendrite. that one doesn't get included in this normalization.
                    dendrite_object.dendritic_connection_strengths[dendrite] = factor_i * dendrite_object.dendritic_connection_strengths[dendrite]
        
    # check that timestep is sufficiently small:
    if dendrite_object.loops_present == 'ri':
        ib_list = ib__list__ri
        r_fq_array = r_fq__array__ri
    elif dendrite_object.loops_present == 'rtti':
        ib_list = ib__list__rtti
        r_fq_array = r_fq__array__rtti
    elif dendrite_object.loops_present == 'pri':
        ib_list = ib__list__rtti
        r_fq_array = r_fq__array__rtti


    _ib_ind = index_finder(ib_list,dendrite_object.ib)
    _flat_rate = [item for sublist in r_fq_array[_ib_ind] for item in sublist]
    _r_max = np.max(_flat_rate)
    _min = 0.01*dendrite_object.beta/dendrite_object.alpha
    _max = 0.1*dendrite_object.beta/dendrite_object.alpha
    if d_tau > 0.1*dendrite_object.beta/dendrite_object.alpha:
        _str = 'Warning: d_tau should be << beta/alpha.'
        _str = '{} For dendrite {} with beta = {:4.2e} and alpha = {:4.2e}'.format(_str,dendrite_object.name,dendrite_object.beta,dendrite_object.alpha)        
        _str = '{} the recommended d_tau is {:5.3e}-{:5.3e} (dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min,_max,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    elif d_tau <= 0.1*dendrite_object.beta/dendrite_object.alpha:
        _str = 'For dendrite {} d_tau = {:5.3e} = {:5.3e} x beta/alpha'.format(dendrite_object.name,d_tau,d_tau*dendrite_object.alpha/dendrite_object.beta)
        _str = '{} (0.01-0.1 x beta/alpha is recommended, dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    _min = 0.01*dendrite_object.beta/_r_max
    _max = 0.1*dendrite_object.beta/_r_max
    if d_tau > 0.1*dendrite_object.beta/_r_max:
        _str = 'Warning: d_tau should be << beta/r_max.'
        _str = '{} For dendrite {} with beta = {:4.2e} and r_max = {:4.2e}'.format(_str,dendrite_object.name,dendrite_object.beta,_r_max)        
        _str = '{} the recommended d_tau is {:5.3e}-{:5.3e} (dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min,_max,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    elif d_tau <= 0.1*dendrite_object.beta/_r_max:
        _str = 'For dendrite {} d_tau = {:5.3e} = {:5.3e} x beta/r_max'.format(dendrite_object.name,d_tau,d_tau*_r_max/dendrite_object.beta)
        _str = '{} (0.01-0.1 x beta/r_max is recommended, dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    return


def rate_array_attachment(dendrite_object):
        
    load_string = 'default_{}'.format(dendrite_object.loops_present)
        
    ib__list, phi_r__array, i_di__array, r_fq__array, _, _ = dend_load_rate_array(load_string) 
    ib__vec = np.asarray(ib__list)
    
    # attach data to this dendrite

    # bias_current = 2.2 #***
    # _ind__ib = -1 #( np.abs( ib__vec[:] - bias_current ) ).argmin() #***
    # print(ib__vec[-1])
    # _ind__ib = ( np.abs( ib__vec[:] - dendrite_object.bias_current ) ).argmin()

    if dendrite_object.loops_present == 'pri':
        # print("slide pri")
        _ind__ib = ( np.abs( ib__vec[:] - dendrite_object.phi_p ) ).argmin()
    else:
        # print("slide bias")
        _ind__ib = -1 #( np.abs( ib__vec[:] - bias_current ) ).argmin() #***
        # _ind__ib = ( np.abs( ib__vec[:] - dendrite_object.bias_current ) ).argmin()
    dendrite_object.phi_r__vec = np.asarray(phi_r__array[_ind__ib])
    dendrite_object.i_di__subarray = np.asarray(i_di__array[_ind__ib],dtype=object)
    dendrite_object.r_fq__subarray = np.asarray(r_fq__array[_ind__ib],dtype=object) 
    
    return

def synapse_initialization(dendrite_object,tau_vec,t_tau_conversion):
    
    for synapse_key in dendrite_object.synaptic_inputs:
        # print("   Initializing synapse: ", dendrite_object.synaptic_inputs[synapse_key].name)
        
        # print('recursive_synapse_initialization:\n  dend_name = {}\n  syn_name = {}\n  in_name = {}\n  spike_times = {}'.format(dendrite_object.name,dendrite_object.synaptic_inputs[synapse_key].name,dendrite_object.synaptic_inputs[synapse_key].input_signal.name,dendrite_object.synaptic_inputs[synapse_key].input_signal.spike_times))          
        
        dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory = 0
        dendrite_object.synaptic_inputs[synapse_key]._st_ind_last = 0
        dendrite_object.synaptic_inputs[synapse_key].phi_spd = np.zeros([len(tau_vec)])
        
        # print('dend = {}, syn = {}'.format(dendrite_object.name,dendrite_object.synaptic_inputs[synapse_key].name))
        
        if hasattr(dendrite_object.synaptic_inputs[synapse_key],'input_signal'):
            if hasattr(dendrite_object.synaptic_inputs[synapse_key].input_signal,'input_temporal_form'):
                if dendrite_object.synaptic_inputs[synapse_key].input_signal.input_temporal_form == 'constant_rate':
                
                    rate = dendrite_object.synaptic_inputs[synapse_key].input_signal.rate * 1e6 # 1e6 because inputs are in MHz
                    # print('rate = {:3.1e}'.format(rate))
                    isi = (1/rate) * 1e9 # 1e9 to convert to ns
                    # print('isi = {:3.1e}'.format(isi))
                    t_f = tau_vec[-1]/t_tau_conversion
                    t_on = dendrite_object.synaptic_inputs[synapse_key].input_signal.t_first_spike
                    dendrite_object.synaptic_inputs[synapse_key].input_signal.spike_times = np.arange(t_on,t_f+isi,isi)
        else:
            dendrite_object.synaptic_inputs[synapse_key].input_signal = dict()
            # dendrite_object.synaptic_inputs[synapse_key].input_signal
        # print(dendrite_object.synaptic_inputs)#[synapse_key].input_signal,synapse_key)
        dendrite_object.synaptic_inputs[synapse_key].spike_times_converted = np.asarray(dendrite_object.synaptic_inputs[synapse_key].input_signal.spike_times) * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].tau_rise_converted = dendrite_object.synaptic_inputs[synapse_key].tau_rise * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].tau_fall_converted = dendrite_object.synaptic_inputs[synapse_key].tau_fall * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].hotspot_duration_converted = dendrite_object.synaptic_inputs[synapse_key].hotspot_duration * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].spd_duration_converted = dendrite_object.synaptic_inputs[synapse_key].spd_duration * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].spd_reset_time_converted = dendrite_object.synaptic_inputs[synapse_key].spd_reset_time * t_tau_conversion
        
        # remove spike times that came in faster than spd can respond
        if len(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted) > 1:
            _spike_times_converted = [dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[0]]
            for ii in range(len(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[1:])):
                if dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[ii] - _spike_times_converted[-1] >= dendrite_object.synaptic_inputs[synapse_key].spd_reset_time_converted:
                    _spike_times_converted.append(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[ii])
            dendrite_object.synaptic_inputs[synapse_key].spike_times_converted = _spike_times_converted     
    
    return


def output_synapse_initialization(neuron_object,tau_vec,t_tau_conversion):
    
    for synapse_key in neuron_object.synaptic_outputs:
        
        neuron_object.synaptic_outputs[synapse_key]._phi_spd_memory = 0
        neuron_object.synaptic_outputs[synapse_key]._st_ind_last = 0
        neuron_object.synaptic_outputs[synapse_key].phi_spd = np.zeros([len(tau_vec)])
        
        neuron_object.synaptic_outputs[synapse_key].spike_times_converted = []
        neuron_object.synaptic_outputs[synapse_key].tau_rise_converted = neuron_object.synaptic_outputs[synapse_key].tau_rise * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].tau_fall_converted = neuron_object.synaptic_outputs[synapse_key].tau_fall * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].hotspot_duration_converted = neuron_object.synaptic_outputs[synapse_key].hotspot_duration * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].spd_duration_converted = neuron_object.synaptic_outputs[synapse_key].spd_duration * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].spd_reset_time_converted = neuron_object.synaptic_outputs[synapse_key].spd_reset_time * t_tau_conversion
    
    return

def transmitter_initialization(neuron_object,t_tau_conversion):
    
    if neuron_object.source_type == 'qd' or neuron_object.source_type == 'ec':
    
        from soen_utilities import pathfinder
        _path = pathfinder()
        
        if neuron_object.source_type == 'qd':
            load_string = 'source_qd_Nph_1.0e+04'
        elif neuron_object.source_type == 'ec':
            load_string = 'source_ec_Nph_1.0e+04'
            
        with open('{}{}{}.soen'.format(_path,'/soen_sim_data/',load_string), 'rb') as data_file:         
            data_dict_imported = pickle.load(data_file) 
            
        time_vec__el = data_dict_imported['time_vec']#*t_tau_conversion
        el_vec = data_dict_imported['dNphdt']
        t_on_tron = data_dict_imported['t_on_tron']*1e9
        tau_rad = data_dict_imported['tau_rad']
        t_off = np.min([ t_on_tron+5*tau_rad , time_vec__el[-1] ]) 
        
        _ind_on = ( np.abs(t_on_tron-time_vec__el) ).argmin()
        _ind_off = ( np.abs(t_off-time_vec__el) ).argmin()
        
        t_vec__el = time_vec__el[_ind_on:_ind_off] - time_vec__el[_ind_on]
        neuron_object.time_params['tau_vec__electroluminescence'] = t_vec__el * t_tau_conversion
        dt_vec = np.diff(t_vec__el)
        el_vec = el_vec[_ind_on:_ind_off]
    
        # form probability distribution
        el_cumulative_vec = np.cumsum(el_vec[:-1]*dt_vec[:])
        el_cumulative_vec = el_cumulative_vec/np.max(el_cumulative_vec)
        neuron_object.electroluminescence_cumulative_vec = el_cumulative_vec
        
    elif neuron_object.source_type == 'delay_delta':
        
        neuron_object.light_production_delay = neuron_object.light_production_delay * t_tau_conversion 
            
    return

def dendrite_data_attachment(dendrite_object,neuron_object):
    
    # attach data to this dendrite
    dendrite_object.output_data = {'s': dendrite_object.s, 'phi_r': dendrite_object.phi_r, 'tau_vec': neuron_object.time_params['tau_vec'], 'time_vec': neuron_object.time_params['tau_vec']/neuron_object.time_params['t_tau_conversion']}
    dendrite_object.time_params = {'t_tau_conversion': neuron_object.time_params['t_tau_conversion']}
            
    return        





def net_step(network_object,tau_vec,d_tau):
               
    # set neuron threshold flags
    for neuron_key in network_object.neurons:
        network_object.neurons[neuron_key].dend__nr_ni.threshold_flag = False
    conversion = network_object.time_params['t_tau_conversion']
    # step through time
    _t0 = time.time_ns()

    if "hardware" in network_object.__dict__.keys():
    # if hasattr('network_object','hardware'):
    # if hasattr('network_object','hardware'):
        print("Hardware in the loop.")
        HW = network_object.hardware
        HW.traces = []
        HW.ib__list, HW.phi_r__array, HW.i_di__array, HW.r_fq__array, _, _ = dend_load_rate_array('default_ri') 
        HW.ib__vec = np.asarray(HW.ib__list)
        HW.conversion = network_object.time_params['t_tau_conversion']
    else:
        # print("No hardware in the loop.")
        network_object.hardware=None
        HW=None
    # print(len(tau_vec))
    for ii in range(len(tau_vec)-1):

        # error check with hardware in the loop at defined moment

        # if hasattr('network_object','hardware'):
        # if "hardware" in network_object.__dict__.keys():
        if network_object.hardware:
            # print("BACKWARD ERROR")
            if ii == HW.check_time/network_object.dt:
                HW.forward_error(network_object.nodes)
                HW.backward_error(network_object.nodes)
            
        # step through neurons
        for node in network_object.nodes:

            neuron = node.neuron

            # update all input synapses and dendrites       
            for dend in node.dendrite_list:
                dendrite_updater(dend,ii,tau_vec[ii+1],d_tau,HW)
                # if hasattr('node',"trace_dendrites"):
                #     if ii == 10000: print("yes",node.name)
                #     if dend not in node.trace_dendrites:
                #         dendrite_updater(dend,ii,tau_vec[ii+1],d_tau,HW)
                # else:
                #     if ii == 10000: print("no",node.name)
                #     dendrite_updater(dend,ii,tau_vec[ii+1],d_tau,HW)

            # update all output synapses
            output_synapse_updater(neuron,ii,tau_vec[ii+1])
            
            # check if ni loop has increased above threshold
            if neuron.dend__nr_ni.s[ii+1] >= neuron.integrated_current_threshold:
                
                neuron.dend__nr_ni.threshold_flag = True
                neuron.dend__nr_ni.spike_times.append(tau_vec[ii+1])
                neuron.spike_times.append(tau_vec[ii+1])
                neuron.spike_indices.append(ii+1)
                
                # add spike to refractory dendrite
                neuron.dend__ref.synaptic_inputs[f'{neuron.name}__syn_refraction'].spike_times_converted = np.append(neuron.dend__ref.synaptic_inputs[f'{neuron.name}__syn_refraction'].spike_times_converted,tau_vec[ii+1])

                if neuron.second_ref == True:
                    neuron.dend__ref_2.synaptic_inputs[f'{neuron.name}__syn_refraction'].spike_times_converted = np.append(neuron.dend__ref_2.synaptic_inputs[f'{neuron.name}__syn_refraction'].spike_times_converted,tau_vec[ii+1])


                # add spike to output synapses
                if neuron.source_type == 'qd' or neuron.source_type == 'ec':
    
                    num_samples = neuron.num_photons_out_factor*len(neuron.synaptic_outputs)
                    random_numbers = rng.random(size = num_samples)
                    
                    photon_delay_tau_vec = np.zeros([num_samples])
                    for qq in range(num_samples):
                        photon_delay_tau_vec[qq] = neuron.time_params['tau_vec__electroluminescence'][ ( np.abs( neuron.electroluminescence_cumulative_vec[:] - random_numbers[qq] ) ).argmin() ]
                                           
                    # assign photons to synapses
                    for synapse_name in neuron.synaptic_outputs:
                        neuron.synaptic_outputs[synapse_name].photon_delay_times__temp = []
                        
                    while len(photon_delay_tau_vec) > 0:
                        
                        for synapse_name in neuron.synaptic_outputs:
                            neuron.synaptic_outputs[synapse_name].photon_delay_times__temp.append(photon_delay_tau_vec[0])
                            photon_delay_tau_vec = np.delete(photon_delay_tau_vec, 0)
                           
                    for synapse_name in neuron.synaptic_outputs:
                        _ind = ( np.abs( tau_vec[:] - ( tau_vec[ii+1] + np.min(neuron.synaptic_outputs[synapse_name].photon_delay_times__temp) ) ) ).argmin()
                                                
                        if len(neuron.synaptic_outputs[synapse_name].spike_times_converted) > 0: # a prior spd event has occurred at this synapse
                            if tau_vec[_ind] - neuron.synaptic_outputs[synapse_name].spike_times_converted[-1] >= neuron.synaptic_outputs[synapse_name].spd_reset_time_converted: # the spd has had time to recover                                
                                neuron.synaptic_outputs[synapse_name].spike_times_converted = np.append(neuron.synaptic_outputs[synapse_name].spike_times_converted,tau_vec[_ind])
                        else: # a prior spd event has not occurred at this synapse
                            neuron.synaptic_outputs[synapse_name].spike_times_converted = np.append(neuron.synaptic_outputs[synapse_name].spike_times_converted,tau_vec[_ind])
                                                
                elif neuron.source_type == 'delay_delta':
                    
                    _ind = ( np.abs( tau_vec[:] - ( tau_vec[ii+1] + neuron.light_production_delay ) ) ).argmin()
                    
                    for synapse_name in neuron.synaptic_outputs:
                        
                        neuron.synaptic_outputs[synapse_name].spike_times_converted = np.append(neuron.synaptic_outputs[synapse_name].spike_times_converted,tau_vec[_ind])
                # end add spike to output synapses
                       
    _t1 = time.time_ns()
    # print('done running network time stepper. t_sim = {:7.5e}s\n'.format(1e-9*(_t1-_t0))) # {:7.5e}
        
    return network_object


def dendrite_updater(dendrite_object,time_index,present_time,d_tau,HW=None):
    
    # make sure dendrite isn't a soma that reached threshold
    if hasattr(dendrite_object, 'is_soma'):
        if dendrite_object.threshold_flag == True:
            update = False
            if present_time - dendrite_object.spike_times[-1] > dendrite_object.absolute_refractory_period_converted: # wait for absolute refractory period before resetting soma
                dendrite_object.threshold_flag = False # reset threshold flag
        else: 
            update = True
    else:
        update = True
                        
    # directly applied flux
    dendrite_object.phi_r[time_index+1] = dendrite_object.phi_r_external__vec[time_index+1]
    

    ### VVV May insert simulation learning here VVV ###

    # applied flux from dendrites
    for dendrite_key in dendrite_object.dendritic_inputs:
        dendrite_object.phi_r[time_index+1] += dendrite_object.dendritic_inputs[dendrite_key].s[time_index] * dendrite_object.dendritic_connection_strengths[dendrite_key]        
        # print(dendrite_object.name,dendrite_object.s[time_index],dendrite_object.dendritic_inputs[dendrite_key].name,dendrite_object.dendritic_inputs[dendrite_key].s[time_index])

    ### ^^^ May insert simulation learning here ^^^ ###

    # self-feedback
    dendrite_object.phi_r[time_index+1] += dendrite_object.self_feedback_coupling_strength * dendrite_object.s[time_index]
    
    # applied flux from synapses
    for synapse_key in dendrite_object.synaptic_inputs:
        # print(dendrite_object.synaptic_inputs[synapse_key])
        # find most recent spike time for this synapse
        _st_ind = np.where( present_time > dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[:] )[0]
        
        if len(_st_ind) > 0:
            
            _st_ind = int(_st_ind[-1])
            if ( dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[_st_ind] <= present_time # the spike happened in the past
                and (present_time - dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[_st_ind]) < dendrite_object.synaptic_inputs[synapse_key].spd_duration_converted  # the spike happened within a relevant duration                
                ):
                    
                    _dt_spk = present_time - dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[_st_ind]
                    _phi_spd = spd_response( dendrite_object.synaptic_inputs[synapse_key].phi_peak, 
                                            dendrite_object.synaptic_inputs[synapse_key].tau_rise_converted,
                                            dendrite_object.synaptic_inputs[synapse_key].tau_fall_converted,
                                            dendrite_object.synaptic_inputs[synapse_key].hotspot_duration_converted, _dt_spk)
                    
                    # to avoid going too low when a new spike comes in
                    if _st_ind - dendrite_object.synaptic_inputs[synapse_key]._st_ind_last == 1: 
                        _phi_spd = np.max( [ _phi_spd , dendrite_object.synaptic_inputs[synapse_key].phi_spd[time_index] ])
                        dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory = _phi_spd
                    if _phi_spd < dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory:
                        dendrite_object.synaptic_inputs[synapse_key].phi_spd[time_index+1] = dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory
                    else:
                        dendrite_object.synaptic_inputs[synapse_key].phi_spd[time_index+1] = _phi_spd * dendrite_object.synaptic_connection_strengths[synapse_key]

                        # dendrite_object.synaptic_connection_strengths[synapse_key] = dendrite_object.synaptic_connection_strengths[synapse_key]*2
                        # print(dendrite_object.synaptic_connection_strengths[synapse_key])

                        dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory = 0
                
            dendrite_object.synaptic_inputs[synapse_key]._st_ind_last = _st_ind
                    
        dendrite_object.phi_r[time_index+1] += dendrite_object.synaptic_inputs[synapse_key].phi_spd[time_index+1]
        
    if np.abs(dendrite_object.phi_r[time_index+1]) > .5:
        # print('\nWarning: absolute value of flux drive to dendrite {} exceeded 1 on time step {} (phi_r = {:5.3f})'.format(dendrite_object.name,time_index+1,dendrite_object.phi_r[time_index+1]))
        dendrite_object.rollover+=1
        if np.abs(dendrite_object.phi_r[time_index+1]) > 1:
            dendrite_object.valleyedout+=1
            if np.abs(dendrite_object.phi_r[time_index+1]) > 1.5:
                dendrite_object.doubleroll+=1
    #         print('phi_r = {:5.3f}? Calm the fuck down, bro.'.format(dendrite_object.phi_r[time_index+1]))
    #     print('\n')``
    # print("*")
    # find relevant entry in r_fq__array

    
    # new_bias=None
    new_bias=dendrite_object.bias_current
    # print(dendrite_object.ib_ramp)
    # if time_index == 250: print(dendrite_object.ib_ramp)
    if 'ib_ramp' in list(dendrite_object.__dict__.keys()):
        if dendrite_object.ib_ramp == True:
            # print("Bias Current Ramp!")
            # new_bias= 2.0523958588352214 - (2.0523958588352214-1.4)*time_index/dendrite_object.time_steps
            new_bias= 1.4 + (2.0523958588352214-1.4)*time_index/dendrite_object.time_steps
    # else:
    #     new_bias=dendrite_object.bias_current
    if HW:
        # if HW.expect[HW.phase][0] != None and HW.expect[HW.phase][1] != None:
        # if len(HW.traces) > 0:
        if 'trace' not in dendrite_object.name:

            for trace in HW.traces:
                if dendrite_object.name == list(trace.dendritic_inputs.keys())[0]:
                    # print(dendrite_object.name, list(trace.dendritic_inputs.keys())[0], trace.name)

                    if "minus" in trace.name:
                        if trace.s[time_index] > 0:
                            new_bias = (1-trace.s[time_index]) * (2.0523958588352214-.99) + HW.baseline
                        # if time_index == 2500 or time_index == 7500: 
                        #     print("minus",trace.name,dendrite_object.name,new_bias)

                    elif "plus" in trace.name:
                        if trace.s[time_index] > 0:
                            new_bias = trace.s[time_index] * (2.0523958588352214-.99) + HW.baseline
                    # if time_index == 2500 or time_index == 7500: 
                    #     print("plus",trace.name,dendrite_object.name,new_bias)

                # if (time_index == 2500 or time_index == 7500): 
                    # print("BIAS: ",dendrite_object.name,new_bias)

                dendrite_object.bias_current = new_bias
                HW.trace_biases[trace.name].append(new_bias)

    dendrite_object.bias_dynamics.append(new_bias)
    _ind__phi_r = ( np.abs( dendrite_object.phi_r__vec[:] - dendrite_object.phi_r[time_index+1] ) ).argmin()
    i_di__vec = np.asarray(dendrite_object.i_di__subarray[_ind__phi_r])

    if dendrite_object.pri == True:
        # if time_index == 100: print("PRI Bias Regime")
        _ind__s = ( np.abs( i_di__vec[:] - (2.7 - dendrite_object.bias_current + dendrite_object.s[time_index] ) )).argmin()

    elif HW:
        # if time_index == 100: print("Dynamic Bias Regime")
        _ind__s = ( np.abs( i_di__vec[:] - (2.0523958588352214-new_bias+dendrite_object.s[time_index]) ) ).argmin() #*** (2.2-new_bias+dendrite_object.s[time_index])
        # _ind__s = ( np.abs( i_di__vec[:] - (2.7 - dendrite_object.bias_current + dendrite_object.s[time_index] ) )).argmin()

    # elif flux_offset...

    else:
        # if time_index == 100: print("Default Bias Regime")
        _ind__s = ( np.abs( i_di__vec[:] - (2.0523958588352214-new_bias+dendrite_object.s[time_index]) ) ).argmin()
        # _ind__s = ( np.abs( i_di__vec[:] - dendrite_object.s[time_index] ) ).argmin()

    r_fq = dendrite_object.r_fq__subarray[_ind__phi_r][_ind__s]
        
    # get alpha
    if hasattr(dendrite_object,'alpha_list'):
        _ind = np.where(dendrite_object.s_list > dendrite_object.s[time_index])
        alpha = dendrite_object.alpha_list[_ind[0][0]]
    else:
        alpha = dendrite_object.alpha    
    
    ### vvv SIGNAL UPDATE vvv ###
    if update == True:
        dendrite_object.s[time_index+1] = dendrite_object.s[time_index] * ( 1 - d_tau*alpha/dendrite_object.beta) + (d_tau/dendrite_object.beta) * r_fq
    ### ^^^ SIGNAL UPDATE ^^^ ###

    return


def output_synapse_updater(neuron_object,time_index,present_time):
    
    for synapse_key in neuron_object.synaptic_outputs:
        
        # find most recent spike time for this synapse
        _st_ind = np.where( present_time > neuron_object.synaptic_outputs[synapse_key].spike_times_converted[:] )[0]
        
        if len(_st_ind) > 0:
            
            _st_ind = int(_st_ind[-1])
            if ( neuron_object.synaptic_outputs[synapse_key].spike_times_converted[_st_ind] <= present_time 
                and (present_time - neuron_object.synaptic_outputs[synapse_key].spike_times_converted[_st_ind]) < neuron_object.synaptic_outputs[synapse_key].spd_duration_converted ): # the case that counts    
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
                    neuron_object.synaptic_outputs[synapse_key].phi_spd[time_index+1] = _phi_spd # * neuron_object.synaptic_connection_strengths[synapse_key]
                    neuron_object.synaptic_outputs[synapse_key]._phi_spd_memory = 0
                
            neuron_object.synaptic_outputs[synapse_key]._st_ind_last = _st_ind
                        
    return




















################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################








def recursive_dendrite_initialization_and_drive_construction(dendrite_object,tau_vec,t_tau_conversion,d_tau):
    # print("initializing")          
    dendrite_object.phi_r_external__vec = np.zeros([len(tau_vec)]) # from external drives
    dendrite_object.phi_r = np.zeros([len(tau_vec)]) # from synapses and dendrites
    dendrite_object.s = np.zeros([len(tau_vec)]) # output variable
    dendrite_object.beta = dendrite_object.circuit_betas[-1]
    
    # add external drives to this dendrite if they're present
    dendrite_object = construct_dendritic_drives(dendrite_object) 
    
    # turn external drives to this dendrite into flux
    dendrite_object.phi_r_external__vec[:] = dendrite_object.offset_flux
    for external_input in dendrite_object.external_inputs:
        dendrite_object.phi_r_external__vec += dendrite_object.external_inputs[external_input].drive_signal * dendrite_object.external_connection_strengths[external_input]
        
    # prepare somas for absolute refractory period
    if hasattr(dendrite_object, 'is_soma'):
        dendrite_object.absolute_refractory_period_converted = dendrite_object.absolute_refractory_period * t_tau_conversion
        
    # normalize inputs
    if dendrite_object.normalize_input_connection_strengths:        
        J_ij_e__init = 0 # excitatory
        J_ij_i__init = 0 # inhibitory
        J_ij_e = dendrite_object.total_excitatory_input_connection_strength
        J_ij_i = dendrite_object.total_inhibitory_input_connection_strength
        # print('J_ij = {}'.format(J_ij))
        for external_input in dendrite_object.external_inputs:
            if dendrite_object.external_connection_strengths[external_input] >= 0:
                J_ij_e__init += dendrite_object.external_connection_strengths[external_input]
            elif dendrite_object.external_connection_strengths[external_input] < 0:
                J_ij_i__init += dendrite_object.external_connection_strengths[external_input]
        for synapse in dendrite_object.synaptic_inputs:
            if dendrite_object.synaptic_connection_strengths[synapse] >= 0:
                J_ij_e__init += dendrite_object.synaptic_connection_strengths[synapse]
            elif dendrite_object.synaptic_connection_strengths[synapse] < 0:
                J_ij_i__init += dendrite_object.synaptic_connection_strengths[synapse]
        for dendrite in dendrite_object.dendritic_inputs:
            if dendrite_object.dendritic_connection_strengths[dendrite] >= 0:
                J_ij_e__init += dendrite_object.dendritic_connection_strengths[dendrite]
            elif dendrite_object.dendritic_connection_strengths[dendrite] < 0:
                if dendrite_object.dendritic_inputs[dendrite].name[-15:] != 'dend_refraction': # make sure this isn't the refractory dendrite. that one doesn't get included in this normalization.
                    J_ij_i__init += dendrite_object.dendritic_connection_strengths[dendrite]
        if J_ij_e__init > 0:
            factor_e = J_ij_e/J_ij_e__init
        else:
            factor_e = 0
        if J_ij_i__init > 0:
            factor_i = J_ij_i/J_ij_i__init
        else:
            factor_i = 0
        # print('J_ij__init = {}'.format(J_ij__init))
        # print('factor = {}'.format(factor))
        for external_input in dendrite_object.external_inputs:
            if dendrite_object.external_connection_strengths[external_input] >= 0:
                dendrite_object.external_connection_strengths[external_input] = factor_e * dendrite_object.external_connection_strengths[external_input]
            elif dendrite_object.external_connection_strengths[external_input] < 0:
                dendrite_object.external_connection_strengths[external_input] = factor_i * dendrite_object.external_connection_strengths[external_input]
        for synapse in dendrite_object.synaptic_inputs:
            if dendrite_object.synaptic_connection_strengths[synapse] >= 0:
                dendrite_object.synaptic_connection_strengths[synapse] = factor_e * dendrite_object.synaptic_connection_strengths[synapse]
            elif dendrite_object.synaptic_connection_strengths[synapse] < 0:
                dendrite_object.synaptic_connection_strengths[synapse] = factor_i * dendrite_object.synaptic_connection_strengths[synapse]
        for dendrite in dendrite_object.dendritic_inputs:
            if dendrite_object.dendritic_connection_strengths[dendrite] >= 0:
                dendrite_object.dendritic_connection_strengths[dendrite] = factor_e * dendrite_object.dendritic_connection_strengths[dendrite]
            elif dendrite_object.dendritic_connection_strengths[dendrite] < 0:
                if dendrite_object.dendritic_inputs[dendrite].name[-15:] != 'dend_refraction': # make sure this isn't the refractory dendrite. that one doesn't get included in this normalization.
                    dendrite_object.dendritic_connection_strengths[dendrite] = factor_i * dendrite_object.dendritic_connection_strengths[dendrite]
        
    # check that timestep is sufficiently small:
    if dendrite_object.loops_present == 'ri':
        ib_list = ib__list__ri
        r_fq_array = r_fq__array__ri
    elif dendrite_object.loops_present == 'rtti':
        ib_list = ib__list__rtti
        r_fq_array = r_fq__array__rtti
    elif dendrite_object.loops_present == 'pri':
        ib_list = ib__list__pri
        r_fq_array = r_fq__array__pri
    _ib_ind = index_finder(ib_list,dendrite_object.ib)
    _flat_rate = [item for sublist in r_fq_array[_ib_ind] for item in sublist]
    _r_max = np.max(_flat_rate)
    _min = 0.01*dendrite_object.beta/dendrite_object.alpha
    _max = 0.1*dendrite_object.beta/dendrite_object.alpha
    if d_tau > 0.1*dendrite_object.beta/dendrite_object.alpha:
        _str = 'Warning: d_tau should be << beta/alpha.'
        _str = '{} For dendrite {} with beta = {:4.2e} and alpha = {:4.2e}'.format(_str,dendrite_object.name,dendrite_object.beta,dendrite_object.alpha)        
        _str = '{} the recommended d_tau is {:5.3e}-{:5.3e} (dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min,_max,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    elif d_tau <= 0.1*dendrite_object.beta/dendrite_object.alpha:
        _str = 'For dendrite {} d_tau = {:5.3e} = {:5.3e} x beta/alpha'.format(dendrite_object.name,d_tau,d_tau*dendrite_object.alpha/dendrite_object.beta)
        _str = '{} (0.01-0.1 x beta/alpha is recommended, dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    _min = 0.01*dendrite_object.beta/_r_max
    _max = 0.1*dendrite_object.beta/_r_max
    if d_tau > 0.1*dendrite_object.beta/_r_max:
        _str = 'Warning: d_tau should be << beta/r_max.'
        _str = '{} For dendrite {} with beta = {:4.2e} and r_max = {:4.2e}'.format(_str,dendrite_object.name,dendrite_object.beta,_r_max)        
        _str = '{} the recommended d_tau is {:5.3e}-{:5.3e} (dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min,_max,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    elif d_tau <= 0.1*dendrite_object.beta/_r_max:
        _str = 'For dendrite {} d_tau = {:5.3e} = {:5.3e} x beta/r_max'.format(dendrite_object.name,d_tau,d_tau*_r_max/dendrite_object.beta)
        _str = '{} (0.01-0.1 x beta/r_max is recommended, dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
        
    # step through all dendrites input to this one and call the present function recursively
    # for dendrite_key in dendrite_object.dendritic_inputs:
    #     recursive_dendrite_initialization_and_drive_construction(dendrite_object.dendritic_inputs[dendrite_key],tau_vec,t_tau_conversion,d_tau) 
    for dendrite_key in dendrite_object.dendritic_inputs:
        recursive_dendrite_initialization_and_drive_construction(dendrite_object.dendritic_inputs[dendrite_key],tau_vec,t_tau_conversion,d_tau) 

    return

def recursive_rate_array_attachment(dendrite_object):
        
    load_string = 'default_{}'.format(dendrite_object.loops_present)
        
    ib__list, phi_r__array, i_di__array, r_fq__array, _, _ = dend_load_rate_array(load_string) 
    ib__vec = np.asarray(ib__list)
    
    # attach data to this dendrite
    _ind__ib = ( np.abs( ib__vec[:] - dendrite_object.bias_current ) ).argmin()
    dendrite_object.phi_r__vec = np.asarray(phi_r__array[_ind__ib])
    dendrite_object.i_di__subarray = np.asarray(i_di__array[_ind__ib],dtype=object)
    dendrite_object.r_fq__subarray = np.asarray(r_fq__array[_ind__ib],dtype=object)
        
    # iterate recursively through all input dendrites
    for dendrite_key in dendrite_object.dendritic_inputs:
        recursive_rate_array_attachment(dendrite_object.dendritic_inputs[dendrite_key])        
    
    return

def recursive_synapse_initialization(dendrite_object,tau_vec,t_tau_conversion):
    
    for synapse_key in dendrite_object.synaptic_inputs:
        
        # print('recursive_synapse_initialization:\n  dend_name = {}\n  syn_name = {}\n  in_name = {}\n  spike_times = {}'.format(dendrite_object.name,dendrite_object.synaptic_inputs[synapse_key].name,dendrite_object.synaptic_inputs[synapse_key].input_signal.name,dendrite_object.synaptic_inputs[synapse_key].input_signal.spike_times))          
        
        dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory = 0
        dendrite_object.synaptic_inputs[synapse_key]._st_ind_last = 0
        dendrite_object.synaptic_inputs[synapse_key].phi_spd = np.zeros([len(tau_vec)])
        
        # print('dend = {}, syn = {}'.format(dendrite_object.name,dendrite_object.synaptic_inputs[synapse_key].name))
        
        if hasattr(dendrite_object.synaptic_inputs[synapse_key],'input_signal'):
            if hasattr(dendrite_object.synaptic_inputs[synapse_key].input_signal,'input_temporal_form'):
                if dendrite_object.synaptic_inputs[synapse_key].input_signal.input_temporal_form == 'constant_rate':
                
                    rate = dendrite_object.synaptic_inputs[synapse_key].input_signal.rate * 1e6 # 1e6 because inputs are in MHz
                    # print('rate = {:3.1e}'.format(rate))
                    isi = (1/rate) * 1e9 # 1e9 to convert to ns
                    # print('isi = {:3.1e}'.format(isi))
                    t_f = tau_vec[-1]/t_tau_conversion
                    t_on = dendrite_object.synaptic_inputs[synapse_key].input_signal.t_first_spike
                    dendrite_object.synaptic_inputs[synapse_key].input_signal.spike_times = np.arange(t_on,t_f+isi,isi)
        else:
            dendrite_object.synaptic_inputs[synapse_key].input_signal = dict()
            # dendrite_object.synaptic_inputs[synapse_key].input_signal
        # print(dendrite_object.synaptic_inputs)#[synapse_key].input_signal,synapse_key)
        dendrite_object.synaptic_inputs[synapse_key].spike_times_converted = np.asarray(dendrite_object.synaptic_inputs[synapse_key].input_signal.spike_times) * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].tau_rise_converted = dendrite_object.synaptic_inputs[synapse_key].tau_rise * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].tau_fall_converted = dendrite_object.synaptic_inputs[synapse_key].tau_fall * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].hotspot_duration_converted = dendrite_object.synaptic_inputs[synapse_key].hotspot_duration * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].spd_duration_converted = dendrite_object.synaptic_inputs[synapse_key].spd_duration * t_tau_conversion
        dendrite_object.synaptic_inputs[synapse_key].spd_reset_time_converted = dendrite_object.synaptic_inputs[synapse_key].spd_reset_time * t_tau_conversion
        
        # remove spike times that came in faster than spd can respond
        if len(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted) > 1:
            _spike_times_converted = [dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[0]]
            for ii in range(len(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[1:])):
                if dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[ii] - _spike_times_converted[-1] >= dendrite_object.synaptic_inputs[synapse_key].spd_reset_time_converted:
                    _spike_times_converted.append(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[ii])
            dendrite_object.synaptic_inputs[synapse_key].spike_times_converted = _spike_times_converted
        
    # iterate recursively through all input dendrites
    for dendrite_key in dendrite_object.dendritic_inputs:
        # if dendrite_object.dendritic_inputs[dendrite_key].synaptic_inputs:
        recursive_synapse_initialization(dendrite_object.dendritic_inputs[dendrite_key],tau_vec,t_tau_conversion)        
    
    return

def output_synapse_initialization(neuron_object,tau_vec,t_tau_conversion):
    
    for synapse_key in neuron_object.synaptic_outputs:
        
        neuron_object.synaptic_outputs[synapse_key]._phi_spd_memory = 0
        neuron_object.synaptic_outputs[synapse_key]._st_ind_last = 0
        neuron_object.synaptic_outputs[synapse_key].phi_spd = np.zeros([len(tau_vec)])
        
        neuron_object.synaptic_outputs[synapse_key].spike_times_converted = []
        neuron_object.synaptic_outputs[synapse_key].tau_rise_converted = neuron_object.synaptic_outputs[synapse_key].tau_rise * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].tau_fall_converted = neuron_object.synaptic_outputs[synapse_key].tau_fall * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].hotspot_duration_converted = neuron_object.synaptic_outputs[synapse_key].hotspot_duration * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].spd_duration_converted = neuron_object.synaptic_outputs[synapse_key].spd_duration * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].spd_reset_time_converted = neuron_object.synaptic_outputs[synapse_key].spd_reset_time * t_tau_conversion
    
    return

def transmitter_initialization(neuron_object,t_tau_conversion):
    
    if neuron_object.source_type == 'qd' or neuron_object.source_type == 'ec':
    
        from soen_utilities import pathfinder
        _path = pathfinder()
        
        if neuron_object.source_type == 'qd':
            load_string = 'source_qd_Nph_1.0e+04'
        elif neuron_object.source_type == 'ec':
            load_string = 'source_ec_Nph_1.0e+04'
            
        with open('{}{}{}.soen'.format(_path,'/soen_sim_data/',load_string), 'rb') as data_file:         
            data_dict_imported = pickle.load(data_file) 
            
        time_vec__el = data_dict_imported['time_vec']#*t_tau_conversion
        el_vec = data_dict_imported['dNphdt']
        t_on_tron = data_dict_imported['t_on_tron']*1e9
        tau_rad = data_dict_imported['tau_rad']
        t_off = np.min([ t_on_tron+5*tau_rad , time_vec__el[-1] ]) 
        
        _ind_on = ( np.abs(t_on_tron-time_vec__el) ).argmin()
        _ind_off = ( np.abs(t_off-time_vec__el) ).argmin()
        
        t_vec__el = time_vec__el[_ind_on:_ind_off] - time_vec__el[_ind_on]
        neuron_object.time_params['tau_vec__electroluminescence'] = t_vec__el * t_tau_conversion
        dt_vec = np.diff(t_vec__el)
        el_vec = el_vec[_ind_on:_ind_off]
    
        # form probability distribution
        el_cumulative_vec = np.cumsum(el_vec[:-1]*dt_vec[:])
        el_cumulative_vec = el_cumulative_vec/np.max(el_cumulative_vec)
        neuron_object.electroluminescence_cumulative_vec = el_cumulative_vec
        
    elif neuron_object.source_type == 'delay_delta':
        
        neuron_object.light_production_delay = neuron_object.light_production_delay * t_tau_conversion 
            
    return

def recursive_dendrite_updater(dendrite_object,time_index,present_time,d_tau):
    
    # make sure dendrite isn't a soma that reached threshold
    if hasattr(dendrite_object, 'is_soma'):
        if dendrite_object.threshold_flag == True:
            update = False
            if present_time - dendrite_object.spike_times[-1] > dendrite_object.absolute_refractory_period_converted: # wait for absolute refractory period before resetting soma
                dendrite_object.threshold_flag = False # reset threshold flag
        else: 
            update = True
    else:
        update = True
                        
    # directly applied flux
    dendrite_object.phi_r[time_index+1] = dendrite_object.phi_r_external__vec[time_index+1]
    

    ### VVV May insert simulation learning here VVV ###

    # applied flux from dendrites
    for dendrite_key in dendrite_object.dendritic_inputs:
        dendrite_object.phi_r[time_index+1] += dendrite_object.dendritic_inputs[dendrite_key].s[time_index] * dendrite_object.dendritic_connection_strengths[dendrite_key]        
        # print(dendrite_object.name,dendrite_object.s[time_index],dendrite_object.dendritic_inputs[dendrite_key].name,dendrite_object.dendritic_inputs[dendrite_key].s[time_index])

    ### ^^^ May insert simulation learning here ^^^ ###

    # self-feedback
    dendrite_object.phi_r[time_index+1] += dendrite_object.self_feedback_coupling_strength * dendrite_object.s[time_index]
    
    # applied flux from synapses
    for synapse_key in dendrite_object.synaptic_inputs:
        
        # find most recent spike time for this synapse
        _st_ind = np.where( present_time > dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[:] )[0]
        
        if len(_st_ind) > 0:
            
            _st_ind = int(_st_ind[-1])
            if ( dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[_st_ind] <= present_time # the spike happened in the past
                and (present_time - dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[_st_ind]) < dendrite_object.synaptic_inputs[synapse_key].spd_duration_converted  # the spike happened within a relevant duration                
                ):
                    
                    _dt_spk = present_time - dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[_st_ind]
                    _phi_spd = spd_response( dendrite_object.synaptic_inputs[synapse_key].phi_peak, 
                                            dendrite_object.synaptic_inputs[synapse_key].tau_rise_converted,
                                            dendrite_object.synaptic_inputs[synapse_key].tau_fall_converted,
                                            dendrite_object.synaptic_inputs[synapse_key].hotspot_duration_converted, _dt_spk)
                    
                    # to avoid going too low when a new spike comes in
                    if _st_ind - dendrite_object.synaptic_inputs[synapse_key]._st_ind_last == 1: 
                        _phi_spd = np.max( [ _phi_spd , dendrite_object.synaptic_inputs[synapse_key].phi_spd[time_index] ])
                        dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory = _phi_spd
                    if _phi_spd < dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory:
                        dendrite_object.synaptic_inputs[synapse_key].phi_spd[time_index+1] = dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory
                    else:
                        dendrite_object.synaptic_inputs[synapse_key].phi_spd[time_index+1] = _phi_spd * dendrite_object.synaptic_connection_strengths[synapse_key]

                        # dendrite_object.synaptic_connection_strengths[synapse_key] = dendrite_object.synaptic_connection_strengths[synapse_key]*2
                        # print(dendrite_object.synaptic_connection_strengths[synapse_key])

                        dendrite_object.synaptic_inputs[synapse_key]._phi_spd_memory = 0
                
            dendrite_object.synaptic_inputs[synapse_key]._st_ind_last = _st_ind
                    
        dendrite_object.phi_r[time_index+1] += dendrite_object.synaptic_inputs[synapse_key].phi_spd[time_index+1]
        
    if np.abs(dendrite_object.phi_r[time_index+1]) > .5:
        # print('\nWarning: absolute value of flux drive to dendrite {} exceeded 1 on time step {} (phi_r = {:5.3f})'.format(dendrite_object.name,time_index+1,dendrite_object.phi_r[time_index+1]))
        dendrite_object.rollover+=1
        if np.abs(dendrite_object.phi_r[time_index+1]) > 1:
            dendrite_object.valleyedout+=1
            if np.abs(dendrite_object.phi_r[time_index+1]) > 1.5:
                dendrite_object.doubleroll+=1
    #         print('phi_r = {:5.3f}? Calm the fuck down, bro.'.format(dendrite_object.phi_r[time_index+1]))
    #     print('\n')
    # print("*")
    # find relevant entry in r_fq__array
    _ind__phi_r = ( np.abs( dendrite_object.phi_r__vec[:] - dendrite_object.phi_r[time_index+1] ) ).argmin()
    i_di__vec = np.asarray(dendrite_object.i_di__subarray[_ind__phi_r])
    _ind__s = ( np.abs( i_di__vec[:] - dendrite_object.s[time_index] ) ).argmin()
    r_fq = dendrite_object.r_fq__subarray[_ind__phi_r][_ind__s]
        
    # get alpha
    if hasattr(dendrite_object,'alpha_list'):
        _ind = np.where(dendrite_object.s_list > dendrite_object.s[time_index])
        alpha = dendrite_object.alpha_list[_ind[0][0]]
    else:
        alpha = dendrite_object.alpha    
    
    ### vvv SIGNAL UPDATE vvv ###

    if update == True:
        dendrite_object.s[time_index+1] = dendrite_object.s[time_index] * ( 1 - d_tau*alpha/dendrite_object.beta) + (d_tau/dendrite_object.beta) * r_fq

    ### ^^^ SIGNAL UPDATE ^^^ ###

    for dendrite in dendrite_object.dendritic_inputs:
        recursive_dendrite_updater(dendrite_object.dendritic_inputs[dendrite],time_index,present_time,d_tau)
        
    return

def output_synapse_updater(neuron_object,time_index,present_time):
    
    for synapse_key in neuron_object.synaptic_outputs:
        
        # find most recent spike time for this synapse
        _st_ind = np.where( present_time > neuron_object.synaptic_outputs[synapse_key].spike_times_converted[:] )[0]
        
        if len(_st_ind) > 0:
            
            _st_ind = int(_st_ind[-1])
            if ( neuron_object.synaptic_outputs[synapse_key].spike_times_converted[_st_ind] <= present_time 
                and (present_time - neuron_object.synaptic_outputs[synapse_key].spike_times_converted[_st_ind]) < neuron_object.synaptic_outputs[synapse_key].spd_duration_converted ): # the case that counts    
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
                    neuron_object.synaptic_outputs[synapse_key].phi_spd[time_index+1] = _phi_spd # * neuron_object.synaptic_connection_strengths[synapse_key]
                    neuron_object.synaptic_outputs[synapse_key]._phi_spd_memory = 0
                
            neuron_object.synaptic_outputs[synapse_key]._st_ind_last = _st_ind
                        
    return

def network_time_stepper(network_object,tau_vec,d_tau):
               
    # set neuron threshold flags
    for neuron_key in network_object.neurons:
        network_object.neurons[neuron_key].dend__nr_ni.threshold_flag = False
    
    # step through time
    _t0 = time.time_ns()
    for ii in range(len(tau_vec)-1):
        # print(ii)
        # step through neurons
        for neuron_key in network_object.neurons:
        
            # update all input synapses and dendrites
            recursive_dendrite_updater(network_object.neurons[neuron_key].dend__nr_ni,ii,tau_vec[ii+1],d_tau)
            
            # update all output synapses
            output_synapse_updater(network_object.neurons[neuron_key],ii,tau_vec[ii+1])
            
            # check if ni loop has increased above threshold
            if network_object.neurons[neuron_key].dend__nr_ni.s[ii+1] >= network_object.neurons[neuron_key].integrated_current_threshold:
                
                network_object.neurons[neuron_key].dend__nr_ni.threshold_flag = True
                network_object.neurons[neuron_key].dend__nr_ni.spike_times.append(tau_vec[ii+1])
                network_object.neurons[neuron_key].spike_times.append(tau_vec[ii+1])
                network_object.neurons[neuron_key].spike_indices.append(ii+1)
                
                # add spike to refractory dendrite
                network_object.neurons[neuron_key].dend__ref.synaptic_inputs['{}__syn_refraction'.format(network_object.neurons[neuron_key].name)].spike_times_converted = np.append(network_object.neurons[neuron_key].dend__ref.synaptic_inputs['{}__syn_refraction'.format(network_object.neurons[neuron_key].name)].spike_times_converted,tau_vec[ii+1])

                if network_object.neurons[neuron_key].second_ref == True:
                    network_object.neurons[neuron_key].dend__ref_2.synaptic_inputs['{}__syn_refraction'.format(network_object.neurons[neuron_key].name)].spike_times_converted = np.append(network_object.neurons[neuron_key].dend__ref_2.synaptic_inputs['{}__syn_refraction'.format(network_object.neurons[neuron_key].name)].spike_times_converted,tau_vec[ii+1])


                # add spike to output synapses
                if network_object.neurons[neuron_key].source_type == 'qd' or network_object.neurons[neuron_key].source_type == 'ec':
    
                    num_samples = network_object.neurons[neuron_key].num_photons_out_factor*len(network_object.neurons[neuron_key].synaptic_outputs)
                    random_numbers = rng.random(size = num_samples)
                    
                    photon_delay_tau_vec = np.zeros([num_samples])
                    for qq in range(num_samples):
                        photon_delay_tau_vec[qq] = network_object.neurons[neuron_key].time_params['tau_vec__electroluminescence'][ ( np.abs( network_object.neurons[neuron_key].electroluminescence_cumulative_vec[:] - random_numbers[qq] ) ).argmin() ]
                                           
                    # assign photons to synapses
                    for synapse_name in network_object.neurons[neuron_key].synaptic_outputs:
                        network_object.neurons[neuron_key].synaptic_outputs[synapse_name].photon_delay_times__temp = []
                        
                    while len(photon_delay_tau_vec) > 0:
                        
                        for synapse_name in network_object.neurons[neuron_key].synaptic_outputs:
                            network_object.neurons[neuron_key].synaptic_outputs[synapse_name].photon_delay_times__temp.append(photon_delay_tau_vec[0])
                            photon_delay_tau_vec = np.delete(photon_delay_tau_vec, 0)
                           
                    for synapse_name in network_object.neurons[neuron_key].synaptic_outputs:
                        _ind = ( np.abs( tau_vec[:] - ( tau_vec[ii+1] + np.min(network_object.neurons[neuron_key].synaptic_outputs[synapse_name].photon_delay_times__temp) ) ) ).argmin()
                                                
                        if len(network_object.neurons[neuron_key].synaptic_outputs[synapse_name].spike_times_converted) > 0: # a prior spd event has occurred at this synapse
                            if tau_vec[_ind] - network_object.neurons[neuron_key].synaptic_outputs[synapse_name].spike_times_converted[-1] >= network_object.neurons[neuron_key].synaptic_outputs[synapse_name].spd_reset_time_converted: # the spd has had time to recover                                
                                network_object.neurons[neuron_key].synaptic_outputs[synapse_name].spike_times_converted = np.append(network_object.neurons[neuron_key].synaptic_outputs[synapse_name].spike_times_converted,tau_vec[_ind])
                        else: # a prior spd event has not occurred at this synapse
                            network_object.neurons[neuron_key].synaptic_outputs[synapse_name].spike_times_converted = np.append(network_object.neurons[neuron_key].synaptic_outputs[synapse_name].spike_times_converted,tau_vec[_ind])
                                                
                elif network_object.neurons[neuron_key].source_type == 'delay_delta':
                    
                    _ind = ( np.abs( tau_vec[:] - ( tau_vec[ii+1] + network_object.neurons[neuron_key].light_production_delay ) ) ).argmin()
                    
                    for synapse_name in network_object.neurons[neuron_key].synaptic_outputs:
                        
                        network_object.neurons[neuron_key].synaptic_outputs[synapse_name].spike_times_converted = np.append(network_object.neurons[neuron_key].synaptic_outputs[synapse_name].spike_times_converted,tau_vec[_ind])
                # end add spike to output synapses
                       
    _t1 = time.time_ns()
    # print('done running network time stepper. t_sim = {:7.5e}s\n'.format(1e-9*(_t1-_t0))) # {:7.5e}
        
    return network_object



def recursive_dendrite_data_attachment(dendrite_object,neuron_object):
    
    # attach data to this dendrite
    dendrite_object.output_data = {'s': dendrite_object.s, 'phi_r': dendrite_object.phi_r, 'tau_vec': neuron_object.time_params['tau_vec'], 'time_vec': neuron_object.time_params['tau_vec']/neuron_object.time_params['t_tau_conversion']}
    dendrite_object.time_params = {'t_tau_conversion': neuron_object.time_params['t_tau_conversion']}
            
    # iterate recursively through all input dendrites
    for dendrite_key in dendrite_object.dendritic_inputs:
        recursive_dendrite_data_attachment(dendrite_object.dendritic_inputs[dendrite_key],neuron_object)        
        
    return        


def spd_response(phi_peak,tau_rise,tau_fall,hotspot_duration,t):
        
    if t <= hotspot_duration:
        phi = phi_peak * ( 1 - tau_rise/tau_fall ) * ( 1 - np.exp( -t / tau_rise ) )
    elif t > hotspot_duration:
        phi = phi_peak * ( 1 - tau_rise/tau_fall ) * ( 1 - np.exp( -hotspot_duration / tau_rise ) ) * np.exp( -( t - hotspot_duration ) / tau_fall )
    
    return phi

def construct_dendritic_drives(obj):
                
    for dir_sig in obj.external_inputs:
        
        if hasattr(obj.external_inputs[dir_sig],'piecewise_linear'):
            dendritic_drive = dendritic_drive__piecewise_linear(obj.time_params['time_vec'],obj.external_inputs[dir_sig].piecewise_linear)
                        
        if hasattr(obj.external_inputs[dir_sig],'applied_flux'):
            dendritic_drive = obj.external_inputs[dir_sig].applied_flux * np.ones([len(obj.phi_r_external__vec)])

        # plot_dendritic_drive(time_vec, dendritic_drive)
        obj.external_inputs[dir_sig].drive_signal = dendritic_drive

    return obj

def dendritic_drive__piecewise_linear(time_vec,pwl):
    
    input_signal__dd = np.zeros([len(time_vec)])
    for ii in range(len(pwl)-1):
        t1_ind = (np.abs(np.asarray(time_vec)-pwl[ii][0])).argmin()
        t2_ind = (np.abs(np.asarray(time_vec)-pwl[ii+1][0])).argmin()
        slope = (pwl[ii+1][1]-pwl[ii][1])/(pwl[ii+1][0]-pwl[ii][0])
        partial_time_vec = time_vec[t1_ind:t2_ind+1]
        input_signal__dd[t1_ind] = pwl[ii][1]
        for jj in range(len(partial_time_vec)-1):
            input_signal__dd[t1_ind+jj+1] = input_signal__dd[t1_ind+jj]+(partial_time_vec[jj+1]-partial_time_vec[jj])*slope
    input_signal__dd[t2_ind:] = pwl[-1][1]*np.ones([len(time_vec)-t2_ind])
    
    return input_signal__dd

def dendritic_drive__exp_pls_train__LR(time_vec,exp_pls_trn_params):
        
    t_r1_start = exp_pls_trn_params['t_r1_start']
    t_r1_rise = exp_pls_trn_params['t_r1_rise']
    t_r1_pulse = exp_pls_trn_params['t_r1_pulse']
    t_r1_fall = exp_pls_trn_params['t_r1_fall']
    t_r1_period = exp_pls_trn_params['t_r1_period']
    value_r1_off = exp_pls_trn_params['value_r1_off']
    value_r1_on = exp_pls_trn_params['value_r1_on']
    r2 = exp_pls_trn_params['r2']
    L1 = exp_pls_trn_params['L1']
    L2 = exp_pls_trn_params['L2']
    Ib = exp_pls_trn_params['Ib']
    
    # make vector of r1(t)
    sq_pls_trn_params = dict()
    sq_pls_trn_params['t_start'] = t_r1_start
    sq_pls_trn_params['t_rise'] = t_r1_rise
    sq_pls_trn_params['t_pulse'] = t_r1_pulse
    sq_pls_trn_params['t_fall'] = t_r1_fall
    sq_pls_trn_params['t_period'] = t_r1_period
    sq_pls_trn_params['value_off'] = value_r1_off
    sq_pls_trn_params['value_on'] = value_r1_on
    # print('making resistance vec ...')
    r1_vec = dendritic_drive__square_pulse_train(time_vec,sq_pls_trn_params)
    
    input_signal__dd = np.zeros([len(time_vec)])
    # print('time stepping ...')
    for ii in range(len(time_vec)-1):
        # print('ii = {} of {}'.format(ii+1,len(time_vec)-1))
        dt = time_vec[ii+1]-time_vec[ii]
        input_signal__dd[ii+1] = input_signal__dd[ii]*( 1 - dt*(r1_vec[ii]+r2)/(L1+L2) ) + dt*Ib*r1_vec[ii]/(L1+L2)
    
    return input_signal__dd

def dendritic_drive__exponential(time_vec,exp_params):
        
    t_rise = exp_params['t_rise']
    t_fall = exp_params['t_fall']
    tau_rise = exp_params['tau_rise']
    tau_fall = exp_params['tau_fall']
    value_on = exp_params['value_on']
    value_off = exp_params['value_off']
    
    input_signal__dd = np.zeros([len(time_vec)])
    for ii in range(len(time_vec)):
        time = time_vec[ii]
        if time < t_rise:
            input_signal__dd[ii] = value_off
        if time >= t_rise and time < t_fall:
            input_signal__dd[ii] = value_off+(value_on-value_off)*(1-np.exp(-(time-t_rise)/tau_rise))
        if time >= t_fall:
            input_signal__dd[ii] = value_off+(value_on-value_off)*(1-np.exp(-(time-t_rise)/tau_rise))*np.exp(-(time-t_fall)/tau_fall)
    
    return input_signal__dd

def dendritic_drive__square_pulse_train(time_vec,sq_pls_trn_params):
    
    input_signal__dd = np.zeros([len(time_vec)])
    dt = time_vec[1]-time_vec[0]
    t_start = sq_pls_trn_params['t_start']
    t_rise = sq_pls_trn_params['t_rise']
    t_pulse = sq_pls_trn_params['t_pulse']
    t_fall = sq_pls_trn_params['t_fall']
    t_period = sq_pls_trn_params['t_period']
    value_off = sq_pls_trn_params['value_off']
    value_on = sq_pls_trn_params['value_on']
    
    tf_sub = t_rise+t_pulse+t_fall
    time_vec_sub = np.arange(0,tf_sub+dt,dt)
    pwl = [[0,value_off],[t_rise,value_on],[t_rise+t_pulse,value_on],[t_rise+t_pulse+t_fall,value_off]]
    
    pulse = dendritic_drive__piecewise_linear(time_vec_sub,pwl)    
    num_pulses = np.floor((time_vec[-1]-t_start)/t_period).astype(int)        
    ind_start = (np.abs(np.asarray(time_vec)-t_start)).argmin()
    ind_pulse_end = (np.abs(np.asarray(time_vec)-t_start-t_rise-t_pulse-t_fall)).argmin()
    ind_per_end = (np.abs(np.asarray(time_vec)-t_start-t_period)).argmin()
    num_ind_pulse = len(pulse) # ind_pulse_end-ind_start
    num_ind_per = ind_per_end-ind_start
    for ii in range(num_pulses):
        input_signal__dd[ind_start+ii*num_ind_per:ind_start+ii*num_ind_per+num_ind_pulse] = pulse[:]
        
    if t_start+num_pulses*t_period <= time_vec[-1] and t_start+(num_pulses+1)*t_period >= time_vec[-1]:
        ind_final = (np.abs(np.asarray(time_vec)-t_start-num_pulses*t_period)).argmin()
        ind_end = (np.abs(np.asarray(time_vec)-t_start-num_pulses*t_period-t_rise-t_pulse-t_fall)).argmin()
        num_ind = ind_end-ind_final
        input_signal__dd[ind_final:ind_end] = pulse[0:num_ind]
        
    return input_signal__dd

# =============================================================================
# saving and loading functions 
# =============================================================================

def load_neuron_data(load_string):
        
    with open('data/'+load_string, 'rb') as data_file:         
        neuron_imported = pickle.load(data_file)
    
    return neuron_imported
    
def save_session_data(data_array = [],save_string = 'soen_sim',include_time = True):
    
    if include_time == True:
        tt = time.time()     
        s_str = save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.dat'
    if include_time == False:
        s_str = save_string
    with open('soen_sim_data/'+s_str, 'wb') as data_file:
            pickle.dump(data_array, data_file)
            
    return

def load_session_data(load_string):
        
    with open('soen_sim_data/'+load_string, 'rb') as data_file:         
        data_array_imported = pickle.load(data_file)

    return data_array_imported


# =============================================================================
# chi squareds
# =============================================================================
def chi_squared_error(target_data,actual_data):
    
    print('\ncalculating chi^2 ...')
    
    target_data__interpolated = np.interp( actual_data[0,:] , target_data[0,:] , target_data[1,:] )
    
    error = 0
    for ii in range(len(actual_data[0,:])-1):        
        error += np.abs( target_data__interpolated[ii+1]-actual_data[1,ii+1] )**2 * ( actual_data[0,ii+1] - actual_data[0,ii] )
        
    norm = 0
    for ii in range(len(target_data[0,:])-1):
        norm += np.abs( target_data[1,ii+1] )**2 * ( target_data[0,ii+1] - target_data[0,ii] ) 
  
    print('done calculating chi^2.\n')
    
    return error/norm


def phi_thresholds(neuron_object):
    if neuron_object.loops_present == 'ri':
        ib__list__ri, phi_r__array__ri, i_di__array__ri, r_fq__array__ri, phi_th_plus__vec__ri, phi_th_minus__vec__ri, s_max_plus__vec__ri, s_max_minus__vec__ri, s_max_plus__array__ri, s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')
        _ind_ib = ( np.abs( np.array(ib__list__ri[:]) - neuron_object.dend__nr_ni.ib ) ).argmin()
        return [phi_th_minus__vec__ri[_ind_ib],phi_th_plus__vec__ri[_ind_ib]]
    elif neuron_object.loops_present == 'rtti':
        ib__list__rtti, phi_r__array__rtti, i_di__array__rtti, r_fq__array__rtti, phi_th_plus__vec__rtti, phi_th_minus__vec__rtti, s_max_plus__vec__rtti, s_max_minus__vec__rtti, s_max_plus__array__rtti, s_max_minus__array__rtti = dend_load_arrays_thresholds_saturations('default_rtti')
        _ind_ib = ( np.abs( np.array(ib__list__rtti[:]) - neuron_object.dend__nr_ni.ib ) ).argmin()
        return [phi_th_minus__vec__rtti[_ind_ib],phi_th_plus__vec__rtti[_ind_ib]]




# %%
