#%%
import numpy as np

from soen_sim import input_signal, dendrite, synapse, neuron
from _util__soen import dend_load_arrays_thresholds_saturations
from soen_sim_lib__util import arg_helper
from soen_sim__parameters import common_parameters

common_params = common_parameters()
ib__list, phi_r__array, i_di__array, r_fq__array, phi_th_plus__vec, phi_th_minus__vec, s_max_plus__vec, s_max_minus__vec, s_max_plus__array, s_max_minus__array = dend_load_arrays_thresholds_saturations('default')

#%% common components

# =============================================================================
# common synapse
# =============================================================================
def common_synapse(name):
    
    syn = synapse(name = name, tau_rise = common_params['syn_tau_rise'], tau_fall = common_params['syn_tau_fall'], 
                  hotspot_duration = common_params['syn_hotspot_duration'], spd_duration = common_params['syn_spd_duration'], 
                  phi_peak = common_params['syn_spd_phi_peak'], spd_reset_time = common_params['syn_spd_reset_time'])
    
    return syn

# =============================================================================
# common dendrites
# =============================================================================
def common_dendrite(name, loops_present, beta_di, tau_di, ib, offset_flux = 0):
    
    dend = dendrite(name = name, loops_present = loops_present, circuit_betas = [2*np.pi*1/4, 2*np.pi*1/4, beta_di], junction_critical_current = common_params['Ic'], junction_beta_c = common_params['beta_c'],
                          bias_current = ib, integration_loop_time_constant = tau_di, normalize_input_connection_strengths = False, total_input_connection_strength = 1, offset_flux = offset_flux)
    
    return dend

# =============================================================================
# monosynaptic neuron
# =============================================================================
def monosynaptic_neuron(name,beta_di,tau_di,ib_dendrite,beta_ni,tau_ni,ib_neuron,i_th):
    
    beta_refractory = 2*np.pi*1e2
    ib_refractory = ib__list[7]
    tau_refractory = 50
    
    gate_params = {'ib': ib_dendrite, 'beta_di': beta_di, 'tau_di': tau_di}

    # establish synapse and dendrite
    syn_in, dend_in = one_syn_one_dend_filter(gate_params)
        
    # establish neuron     
    neu = neuron(name = 'ne', 
                      
                      # neuron receiving/integrating dendrite
                      loops_present = 'ri', circuit_betas = [2*np.pi*1/4, 2*np.pi*1/4, beta_ni], junction_critical_current = common_params['Ic'], junction_beta_c = common_params['beta_c'],
                      bias_current = ib_neuron, integration_loop_time_constant = tau_ni, normalize_input_connection_strengths = False, total_input_connection_strength = 1,
                      
                      # neuron refractory dendrite
                      loops_present__refraction = 'ri', circuit_betas__refraction = [2*np.pi*1/4, 2*np.pi*1/4, beta_refractory], junction_critical_current__refraction = common_params['Ic'], junction_beta_c__refraction = common_params['beta_c'],
                      bias_current__refraction = ib_refractory, integration_loop_time_constant__refraction = tau_refractory, refractory_dendrite_connection_strength = 'auto',
                      
                      # synapse to refractory dendrite
                      tau_rise__refraction = common_params['syn_tau_rise'], tau_fall__refraction = common_params['syn_tau_fall'], hotspot_duration__refraction = common_params['syn_hotspot_duration'], spd_duration__refraction = common_params['syn_spd_duration'], phi_peak__refraction = common_params['syn_spd_phi_peak'],
                      
                      # transmitter
                      integrated_current_threshold = i_th, source_type = 'delay_delta', light_production_delay = 2) # 'delay_delta'
                      # integrated_current_threshold = 0.2, source_type = 'ec', num_photons_out_factor = 10) # 'qd' 'ec'
    
    # add input dendrite to neuron
    neu.add_input(dend_in, connection_strength = 1)
    
    # create output synapse
    syn_out = common_synapse(name = 'syn_out')
    
    # add output synapse to neuron
    neu.add_output(syn_out)   
        
    return syn_in, dend_in, neu, syn_out

# =============================================================================
# common neuron
# =============================================================================
def common_neuron(name, loops_present, beta_ni, tau_ni, ib, s_th, beta_ref, tau_ref, ib_ref, offset_flux = 0):
        
    neuron_1 = neuron(name = name,
                      
                  # neuron receiving/integrating dendrite
                  loops_present = loops_present, circuit_betas = [2*np.pi*1/4, 2*np.pi*1/4, beta_ni], junction_critical_current = common_params['Ic'], junction_beta_c = common_params['beta_c'],
                  bias_current = ib, integration_loop_time_constant = tau_ni, absolute_refractory_period = common_params['absolute_refractory_period'], 
                  normalize_input_connection_strengths = False, total_input_connection_strength = 1, offset_flux = offset_flux,
                  
                  # neuron refractory dendrite
                  loops_present__refraction = 'ri', circuit_betas__refraction = [2*np.pi*1/4, 2*np.pi*1/4, beta_ref], junction_critical_current__refraction = 100, junction_beta_c__refraction = 0.3,
                  bias_current__refraction = ib_ref, integration_loop_time_constant__refraction = tau_ref, refractory_dendrite_connection_strength = 'auto',
                  
                  # synapse to refractory dendrite
                  tau_rise__refraction = common_params['syn_tau_rise'], tau_fall__refraction = common_params['syn_tau_fall'], hotspot_duration__refraction = common_params['syn_hotspot_duration'], spd_duration__refraction = common_params['syn_spd_duration'], phi_peak__refraction = common_params['syn_spd_phi_peak'],
                                    
                  # transmitter
                  integrated_current_threshold = s_th, light_production_delay = 2, source_type = 'qd')
    
    return neuron_1

#%%

# =============================================================================
# synapses with dendrites
# =============================================================================

def one_syn_one_dend_filter(params = dict()):
    
    loops_present = arg_helper(params,'loops_present','ri')
    ib = arg_helper(params,'ib',1.7)
    beta_di = arg_helper(params,'beta_di',2*np.pi*1e3)    
    tau_di = arg_helper(params,'tau_di',250)
    
    print('generating one synapse, one dendrite filter\nbeta_di = {:4.2e}, tau_di = {:4.2e}ns'.format(beta_di,tau_di))
    
    # establish synapse
    syn = common_synapse('syn_1')
    # synapse(name = 'syn_1', tau_rise = common_params['syn_tau_rise'], tau_fall = common_params['syn_tau_fall'], hotspot_duration = common_params['syn_hotspot_duration'], spd_duration = common_params['syn_spd_duration'], phi_peak = common_params['syn_spd_phi_peak'])
    
    # establish dendrite     
    dend = common_dendrite('dendrite_filter', loops_present, beta_di, tau_di, ib)
    
    # add synapse to dendrite
    dend.add_input(syn, connection_strength = 1)
    
    return syn, dend

def two_syn_one_dend_logic(params = dict()):
    
    gate = arg_helper(params,'gate','AND')
    # loops_present = arg_helper(params,'loops_present','ri')
    ib = arg_helper(params,'ib',1.7)
    beta_di = arg_helper(params,'beta_di',2*np.pi*1e3)    
    tau_di = arg_helper(params,'tau_di',250)
    connection_strengths = arg_helper(params,'connection_strengths',[])
    
    print('generating two synapse, one dendrite {} gate'.format(gate))
        
    # establish synapses
    syn_1 = common_synapse('syn_1')
    syn_2 = common_synapse('syn_2')
    
    # establish dendrite   
    if gate == 'AND' or gate == 'AND-NOT' or gate == 'XOR':
        loops_present = 'ri'
    elif gate =='OR':
        loops_present = 'rtti'
    dend = common_dendrite('logic_gate', loops_present, beta_di, tau_di, ib)
    
    # add synapses to dendrite
    if len(connection_strengths) == 0:
        if gate == 'AND':
            connection_strengths = [0.5,0.5]
        elif gate =='OR':
            connection_strengths = [0.6,0.6]
        elif gate == 'AND-NOT':
            connection_strengths = [1,-0.5]
        elif gate == 'XOR':
            connection_strengths = [1,-1]
        
    dend.add_input(syn_1, connection_strength = connection_strengths[0])
    dend.add_input(syn_2, connection_strength = connection_strengths[1])     
    
    return syn_1, syn_2, dend

def one_syn_two_dend_one_dend_band_pass(params = dict()):
    
    ib_1a = arg_helper(params,'ib_1a',1.7)
    ib_1b = arg_helper(params,'ib_1b',1.7)
    ib_2 = arg_helper(params,'ib_2',1.7)
    beta_di_1a = arg_helper(params,'beta_di_1a',2*np.pi*1e3)    
    beta_di_1b = arg_helper(params,'beta_di_1b',2*np.pi*1e3)
    beta_di_2 = arg_helper(params,'beta_di_2',2*np.pi*1e3)        
    tau_di_1a = arg_helper(params,'tau_di_1a',250)
    tau_di_1b = arg_helper(params,'tau_di_1b',250)
    tau_di_2 = arg_helper(params,'tau_di_2',250)
    
    # establish synapse
    syn = common_synapse('syn_1')
    
    # establish dendrites
    dend_1a = common_dendrite('logic_gate', beta_di_1a, tau_di_1a, ib_1a)
    dend_1b = common_dendrite('logic_gate', beta_di_1b, tau_di_1b, ib_1b)
    dend_2 = common_dendrite('logic_gate', beta_di_2, tau_di_2, ib_2)
    
    # connect synapse to first two dendrites
    dend_1a.add_input(syn, connection_strength = 1)
    dend_1b.add_input(syn, connection_strength = 1)
    
    # connect dendrites at first stage to second stage
    dend_2.add_input(dend_1a, connection_strength = 1)
    dend_2.add_input(dend_1b, connection_strength = -1)
    
    return syn, dend_1a, dend_1b, dend_2