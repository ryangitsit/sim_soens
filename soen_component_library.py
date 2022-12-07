#%%
import numpy as np

from soen_sim import input_signal, dendrite, synapse, neuron
# from soen_sim_lib__util import arg_helper
# from soen_sim__parameters import common_parameters
from soen_utilities import dend_load_arrays_thresholds_saturations, color_dictionary

# common_params = common_parameters()
ib__list, phi_r__array, i_di__array, r_fq__array, phi_th_plus__vec, phi_th_minus__vec, s_max_plus__vec, s_max_minus__vec, s_max_plus__array, s_max_minus__array = dend_load_arrays_thresholds_saturations('default')

#%% common components

# =============================================================================
# soen_sim_lib__util
# =============================================================================

def arg_helper(params,parameter_name,default_value):
    
    if parameter_name in params.keys(): 
        value = params[parameter_name] 
    else: 
        value = default_value
    
    return value

# =============================================================================
# end soen_sim_lib__util
# =============================================================================

# =============================================================================
# soen_sim__parameters
# =============================================================================

def common_parameters():
    
    jj_params = {'Ic': 100, # uA
                 'beta_c': 0.3, # dimensionless
                 }
    
    syn_params = {'syn_tau_rise': 0.02, # ns
                  'syn_tau_fall': 50, # ns
                  'syn_hotspot_duration': 3, # number of tau rise time constants
                  'syn_spd_duration': 8, # number of tau fall time constants
                  'syn_spd_phi_peak': 0.5, # Phi_0
                  'syn_spd_reset_time': 50, # time that must elapse between spd detection events
                  }
    
    neu_params = neu_params = {'absolute_refractory_period': 20}
    
    params = {**jj_params, **syn_params, **neu_params}
    
    return params

# =============================================================================
# end soen_sim__parameters
# =============================================================================

common_params = common_parameters()

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
def common_dendrite(name, loops_present, beta_di, tau_di, ib, offset_flux = 0, self_feedback_coupling_strength = 0, 
                    normalize_input_connection_strengths = False, total_excitatory_input_connection_strength = 1, total_inhibitory_input_connection_strength = -0.5):
    
    dend = dendrite(name = name, loops_present = loops_present, circuit_betas = [2*np.pi*1/4, 2*np.pi*1/4, beta_di], junction_critical_current = common_params['Ic'], junction_beta_c = common_params['beta_c'],
                          bias_current = ib, integration_loop_time_constant = tau_di, 
                          normalize_input_connection_strengths = normalize_input_connection_strengths, 
                          total_excitatory_input_connection_strength = total_excitatory_input_connection_strength, 
                          total_inhibitory_input_connection_strength = total_inhibitory_input_connection_strength,
                          offset_flux = offset_flux, self_feedback_coupling_strength = self_feedback_coupling_strength)
    
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
bias_current = ib_neuron, integration_loop_time_constant = tau_ni, normalize_input_connection_strengths = True, total_excitatory_input_connection_strength = 1, total_inhibitory_input_connection_strength = -0.5,
                      
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
def common_neuron(name, loops_present, beta_ni, tau_ni, ib, s_th, beta_ref, tau_ref, ib_ref, offset_flux = 0, self_feedback_coupling_strength = 0, refractory_dendrite_connection_strength = 'auto',
                  normalize_input_connection_strengths = False, total_excitatory_input_connection_strength = 1, total_inhibitory_input_connection_strength = -0.5):
        
    neuron_1 = neuron(name = name,
                      
                  # neuron receiving/integrating dendrite
                  loops_present = loops_present, circuit_betas = [2*np.pi*1/4, 2*np.pi*1/4, beta_ni], junction_critical_current = common_params['Ic'], junction_beta_c = common_params['beta_c'],
                  bias_current = ib, integration_loop_time_constant = tau_ni, absolute_refractory_period = common_params['absolute_refractory_period'], 
                  normalize_input_connection_strengths = normalize_input_connection_strengths, 
                  total_excitatory_input_connection_strength = total_excitatory_input_connection_strength, 
                  total_inhibitory_input_connection_strength = total_inhibitory_input_connection_strength, 
                  offset_flux = offset_flux, self_feedback_coupling_strength = 0,
                  
                  # neuron refractory dendrite
                  loops_present__refraction = 'ri', circuit_betas__refraction = [2*np.pi*1/4, 2*np.pi*1/4, beta_ref], junction_critical_current__refraction = 100, junction_beta_c__refraction = 0.3,
                  bias_current__refraction = ib_ref, integration_loop_time_constant__refraction = tau_ref, refractory_dendrite_connection_strength = refractory_dendrite_connection_strength,
                  
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

#%%
# =============================================================================
# nine pixel helpers
# =============================================================================

import copy
import matplotlib as mp
from matplotlib import pyplot as plt
fig_size = plt.rcParams['figure.figsize']
colors = color_dictionary()

def nine_pixel_classifier_drive(plot_drive_array = False):
    
    _z = [1,1,0,
          0,1,0,
          0,1,1]
    
    _v = [1,0,1,
          1,0,1,
          0,1,0]
    
    _n = [0,1,0,
          1,0,1,
          1,0,1]
            
    mat_list = [_z,_v,_n]

    drive_dict = dict()
    str_list = ['z','v','n']
    for ii in range(len(mat_list)):
        _str = str_list[ii]
        _mat = mat_list[ii]
        drive_dict['{}_{:d}'.format(_str,0)] = _mat
        for jj in range(len(_mat)):
            _mat_mod = copy.deepcopy(_mat)
            _mat_mod[jj] = ( _mat_mod[jj] + 1 ) % 2
            drive_dict['{}_{:d}'.format(_str,jj+1)] = _mat_mod            
        
    if plot_drive_array:
    
        color_map = mp.colors.ListedColormap([colors['grey1'],colors['black']])
        fig, ax = plt.subplots(nrows = 3, ncols = 10, sharex = False, sharey = False, figsize = ( fig_size[0] , fig_size[1] ) ) 
        for ii in range(len(ax[:,1])):
            for jj in range(len(ax[1,:])):
                
                _temp_mat = np.zeros([3,3])
                for kk in range(3):
                    _temp_mat[:,kk] = drive_dict['{}_{}'.format(str_list[ii],jj)][kk*3:(kk+1)*3]
                                
                ax[ii,jj].imshow(np.transpose(_temp_mat[:,:]), cmap = color_map, interpolation='none', extent = [0.5,3.5,0.5,3.5], aspect = 'equal', origin = 'upper') # np.transpose(_temp_mat[:,:]) # np.fliplr(np.transpose(_temp_mat[:,:]))
                
                # major ticks
                ax[ii,jj].set_xticks(np.asarray([1,2,3]))
                ax[ii,jj].set_yticks(np.asarray([3,2,1]))
    
                # labels for major ticks
                ax[ii,jj].set_xticklabels(np.asarray([1,2,3]))
                ax[ii,jj].set_yticklabels(np.asarray([1,2,3]))
                
                # minor ticks
                ax[ii,jj].set_xticks(np.asarray([1.5,2.5]), minor=True)
                ax[ii,jj].set_yticks(np.asarray([1.5,2.5]), minor=True)
                
                # gridlines based on minor ticks
                ax[ii,jj].grid(which='minor', color = colors['greengrey3'], linestyle='-', linewidth=1)
                    
        fig.suptitle('Nine-pixel classifier drive matrices')  
        plt.show()
    
    return drive_dict # drive_array, 

from soen_component_library import common_synapse, common_dendrite
from soen_sim import input_signal

def nine_pixel_synapses():
    syn_1 = common_synapse('syn_1')
    syn_2 = common_synapse('syn_2')
    syn_3 = common_synapse('syn_3')
    syn_4 = common_synapse('syn_4')
    syn_5 = common_synapse('syn_5')
    syn_6 = common_synapse('syn_6')
    syn_7 = common_synapse('syn_7')
    syn_8 = common_synapse('syn_8')
    syn_9 = common_synapse('syn_9')
    syn_out = common_synapse('syn_out')
    return syn_1, syn_2, syn_3, syn_4, syn_5, syn_6, syn_7, syn_8, syn_9, syn_out

def nine_pixel_stage_1_dendrites(OR_beta,AND_beta,tau,OR_bias,AND_bias):
    dend_2or5 = common_dendrite('dend_2or5', 'rtti', OR_beta, tau, OR_bias)
    dend_4and6 = common_dendrite('dend_4and6', 'ri', AND_beta, tau, AND_bias)
    dend_5or8 = common_dendrite('dend_5or8', 'rtti', OR_beta, tau, OR_bias)
    dend_1or3 = common_dendrite('dend_1or3', 'rtti', OR_beta, tau, OR_bias)
    dend_7and9 = common_dendrite('dend_7and9', 'ri', AND_beta, tau, AND_bias)
    dend_4or6 = common_dendrite('dend_4or6', 'rtti', OR_beta, tau, OR_bias)
    dend_2and5 = common_dendrite('dend_2and5', 'ri', AND_beta, tau, AND_bias)
    dend_7or9 = common_dendrite('dend_7or9', 'rtti', OR_beta, tau, OR_bias)
    dend_1and3 = common_dendrite('dend_1and3', 'ri', AND_beta, tau, AND_bias)
    dend_5and8 = common_dendrite('dend_5and8', 'ri', AND_beta, tau, AND_bias)
    return dend_2or5, dend_4and6, dend_5or8, dend_1or3, dend_7and9, dend_4or6, dend_2and5, dend_7or9, dend_1and3, dend_5and8

def nine_pixel_stage_2_dendrites(ANDNOT_beta,tau,ANDNOT_bias):
    dend_2or5_andnot_4and6 = common_dendrite('dend_2or5_andnot_4and6', 'ri', ANDNOT_beta, tau, ANDNOT_bias)
    dend_5or8_andnot_4and6 = common_dendrite('dend_5or8_andnot_4and6', 'ri', ANDNOT_beta, tau, ANDNOT_bias)
    dend_1or3_andnot_7and9 = common_dendrite('dend_1or3_andnot_7and9', 'ri', ANDNOT_beta, tau, ANDNOT_bias)
    dend_4or6_andnot_2and5 = common_dendrite('dend_4or6_andnot_2and5', 'ri', ANDNOT_beta, tau, ANDNOT_bias)
    dend_7or9_andnot_1and3 = common_dendrite('dend_7or9_andnot_1and3', 'ri', ANDNOT_beta, tau, ANDNOT_bias)
    dend_4or6_andnot_5and8 = common_dendrite('dend_4or6_andnot_5and8', 'ri', ANDNOT_beta, tau, ANDNOT_bias)
    return dend_2or5_andnot_4and6, dend_5or8_andnot_4and6, dend_1or3_andnot_7and9, dend_4or6_andnot_2and5, dend_7or9_andnot_1and3, dend_4or6_andnot_5and8 

def nine_pixel_stage_3_dendrites(AND_beta,tau_z,tau_v,tau_n,AND_bias):
    dend_z = common_dendrite('dend_z', 'ri', AND_beta, tau_z, AND_bias)
    dend_v = common_dendrite('dend_v', 'ri', AND_beta, tau_v, AND_bias)
    dend_n = common_dendrite('dend_n', 'ri', AND_beta, tau_n, AND_bias)
    return dend_z, dend_v, dend_n

def nine_pixel_stage_3_dendrites__logic_level_restoration(beta,tau_z,tau_v,tau_n,bias):
    dend_z2 = common_dendrite('dend_z2', 'rtti', beta, tau_z, bias)
    dend_v2 = common_dendrite('dend_v2', 'rtti', beta, tau_v, bias)
    dend_n2 = common_dendrite('dend_n2', 'rtti', beta, tau_n, bias)
    return dend_z2, dend_v2, dend_n2

def nine_pixel_add_inputs_to_synapses(syn_1, syn_2, syn_3, syn_4, syn_5, syn_6, syn_7, syn_8, syn_9, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
    syn_1.add_input(in_1)
    syn_2.add_input(in_2)
    syn_3.add_input(in_3)
    syn_4.add_input(in_4)
    syn_5.add_input(in_5)
    syn_6.add_input(in_6)
    syn_7.add_input(in_7)
    syn_8.add_input(in_8)
    syn_9.add_input(in_9)
    return syn_1, syn_2, syn_3, syn_4, syn_5, syn_6, syn_7, syn_8, syn_9

def nine_pixel_add_synapses_to_stage_one_dendrites(syn_1, syn_2, syn_3, syn_4, syn_5, syn_6, syn_7, syn_8, syn_9, dend_2or5, dend_4and6, dend_5or8, dend_1or3, dend_7and9, dend_4or6, dend_2and5, dend_7or9, dend_1and3, dend_5and8, OR_connection_strengths, AND_connection_strengths):    
    dend_2or5.add_input(syn_2, connection_strength = OR_connection_strengths[0])
    dend_2or5.add_input(syn_5, connection_strength = OR_connection_strengths[1])        
    dend_4and6.add_input(syn_4, connection_strength = AND_connection_strengths[0])
    dend_4and6.add_input(syn_6, connection_strength = AND_connection_strengths[1])    
    dend_5or8.add_input(syn_5, connection_strength = OR_connection_strengths[0])
    dend_5or8.add_input(syn_8, connection_strength = OR_connection_strengths[1])    
    dend_1or3.add_input(syn_1, connection_strength = OR_connection_strengths[0])
    dend_1or3.add_input(syn_3, connection_strength = OR_connection_strengths[1])    
    dend_7and9.add_input(syn_7, connection_strength = AND_connection_strengths[0])
    dend_7and9.add_input(syn_9, connection_strength = AND_connection_strengths[1])    
    dend_4or6.add_input(syn_4, connection_strength = OR_connection_strengths[0])
    dend_4or6.add_input(syn_6, connection_strength = OR_connection_strengths[1])    
    dend_2and5.add_input(syn_2, connection_strength = AND_connection_strengths[0])
    dend_2and5.add_input(syn_5, connection_strength = AND_connection_strengths[1])    
    dend_7or9.add_input(syn_7, connection_strength = OR_connection_strengths[0])
    dend_7or9.add_input(syn_9, connection_strength = OR_connection_strengths[1])    
    dend_1and3.add_input(syn_1, connection_strength = AND_connection_strengths[0])
    dend_1and3.add_input(syn_3, connection_strength = AND_connection_strengths[1])    
    dend_5and8.add_input(syn_5, connection_strength = AND_connection_strengths[0])
    dend_5and8.add_input(syn_8, connection_strength = AND_connection_strengths[1])
    return dend_2or5, dend_4and6, dend_5or8, dend_1or3, dend_7and9, dend_4or6, dend_2and5, dend_7or9, dend_1and3, dend_5and8

def nine_pixel_add_stage_one_dendrites_to_stage_two_dendrites(dend_2or5,dend_4and6,dend_5or8,dend_1or3,dend_7and9,dend_4or6,dend_2and5,dend_7or9,dend_1and3,dend_5and8,dend_2or5_andnot_4and6,dend_5or8_andnot_4and6,dend_1or3_andnot_7and9,dend_4or6_andnot_2and5,dend_7or9_andnot_1and3,dend_4or6_andnot_5and8,ANDNOT_connection_strengths):
    dend_2or5_andnot_4and6.add_input(dend_2or5, connection_strength = ANDNOT_connection_strengths[0])
    dend_2or5_andnot_4and6.add_input(dend_4and6, connection_strength = ANDNOT_connection_strengths[1])
    dend_5or8_andnot_4and6.add_input(dend_5or8, connection_strength = ANDNOT_connection_strengths[0])
    dend_5or8_andnot_4and6.add_input(dend_4and6, connection_strength = ANDNOT_connection_strengths[1])
    dend_1or3_andnot_7and9.add_input(dend_1or3, connection_strength = ANDNOT_connection_strengths[0])
    dend_1or3_andnot_7and9.add_input(dend_7and9, connection_strength = ANDNOT_connection_strengths[1])
    dend_4or6_andnot_2and5.add_input(dend_4or6, connection_strength = ANDNOT_connection_strengths[0])
    dend_4or6_andnot_2and5.add_input(dend_2and5, connection_strength = ANDNOT_connection_strengths[1])
    dend_7or9_andnot_1and3.add_input(dend_7or9, connection_strength = ANDNOT_connection_strengths[0])
    dend_7or9_andnot_1and3.add_input(dend_1and3, connection_strength = ANDNOT_connection_strengths[1])
    dend_4or6_andnot_5and8.add_input(dend_4or6, connection_strength = ANDNOT_connection_strengths[0])
    dend_4or6_andnot_5and8.add_input(dend_5and8, connection_strength = ANDNOT_connection_strengths[1])
    return dend_2or5_andnot_4and6,dend_5or8_andnot_4and6,dend_1or3_andnot_7and9,dend_4or6_andnot_2and5,dend_7or9_andnot_1and3,dend_4or6_andnot_5and8

def nine_pixel_add_stage_two_dendrites_to_stage_three_dendrites(dend_2or5_andnot_4and6,dend_5or8_andnot_4and6,dend_1or3_andnot_7and9,dend_4or6_andnot_2and5,dend_7or9_andnot_1and3,dend_4or6_andnot_5and8,dend_z,dend_v,dend_n,AND_connection_strengths):
    dend_z.add_input(dend_2or5_andnot_4and6, connection_strength = AND_connection_strengths[0])
    dend_z.add_input(dend_5or8_andnot_4and6, connection_strength = AND_connection_strengths[1])
    dend_v.add_input(dend_1or3_andnot_7and9, connection_strength = AND_connection_strengths[0])
    dend_v.add_input(dend_4or6_andnot_2and5, connection_strength = AND_connection_strengths[1])
    dend_n.add_input(dend_7or9_andnot_1and3, connection_strength = AND_connection_strengths[0])
    dend_n.add_input(dend_4or6_andnot_5and8, connection_strength = AND_connection_strengths[1])
    return dend_z,dend_v,dend_n

def nine_pixel_add_stage_three_dendrites_and_output_synapse_to_neuron(dend_z2,dend_v2,dend_n2,syn_out,neuron_1,connection_strength):
    neuron_1.add_input(dend_z2, connection_strength = connection_strength)
    neuron_1.add_input(dend_v2, connection_strength = connection_strength)
    neuron_1.add_input(dend_n2, connection_strength = connection_strength)
    neuron_1.add_output(syn_out)
    return neuron_1

def nine_pixel_add_stage_three_dendrites_to_logic_level_restoration(dend_z,dend_v,dend_n,dend_z2,dend_v2,dend_n2,connections_strengths__logic_level_restoration):
    dend_z2.add_input(dend_z, connection_strength = connections_strengths__logic_level_restoration)
    dend_v2.add_input(dend_v, connection_strength = connections_strengths__logic_level_restoration)
    dend_n2.add_input(dend_n, connection_strength = connections_strengths__logic_level_restoration)
    return dend_z2,dend_v2,dend_n2

def nine_pixel_generate_inputs_and_add_to_synapses(syn_1, syn_2, syn_3, syn_4, syn_5, syn_6, syn_7, syn_8, syn_9, drive_dict, input_strings, t_on, Dt, with_jitter = False, num_photons_per_spike = 1):
    
    spike_times__array = []
    for ii in range(9):
        spike_times__array.append([])
        
        for jj in range(len(input_strings)):
            
            drive_vec = drive_dict[input_strings[jj]]
        
            if drive_vec[ii] == 1:
                spike_times__array[ii].append(t_on+jj*Dt)
            # elif drive_vec[ii] == 0:
                # spike_times__array[ii].append()
            
    if with_jitter:
        in_1 = input_signal(name = 'in_1', input_temporal_form = 'arbitrary_spike_train_with_jitter', spike_times = np.asarray(spike_times__array[0]), source_type = 'qd', num_photons_per_spike = num_photons_per_spike)
        in_2 = input_signal(name = 'in_2', input_temporal_form = 'arbitrary_spike_train_with_jitter', spike_times = np.asarray(spike_times__array[1]), source_type = 'qd', num_photons_per_spike = num_photons_per_spike)
        in_3 = input_signal(name = 'in_3', input_temporal_form = 'arbitrary_spike_train_with_jitter', spike_times = np.asarray(spike_times__array[2]), source_type = 'qd', num_photons_per_spike = num_photons_per_spike)
        in_4 = input_signal(name = 'in_4', input_temporal_form = 'arbitrary_spike_train_with_jitter', spike_times = np.asarray(spike_times__array[3]), source_type = 'qd', num_photons_per_spike = num_photons_per_spike)
        in_5 = input_signal(name = 'in_5', input_temporal_form = 'arbitrary_spike_train_with_jitter', spike_times = np.asarray(spike_times__array[4]), source_type = 'qd', num_photons_per_spike = num_photons_per_spike)
        in_6 = input_signal(name = 'in_6', input_temporal_form = 'arbitrary_spike_train_with_jitter', spike_times = np.asarray(spike_times__array[5]), source_type = 'qd', num_photons_per_spike = num_photons_per_spike)
        in_7 = input_signal(name = 'in_7', input_temporal_form = 'arbitrary_spike_train_with_jitter', spike_times = np.asarray(spike_times__array[6]), source_type = 'qd', num_photons_per_spike = num_photons_per_spike)
        in_8 = input_signal(name = 'in_8', input_temporal_form = 'arbitrary_spike_train_with_jitter', spike_times = np.asarray(spike_times__array[7]), source_type = 'qd', num_photons_per_spike = num_photons_per_spike)
        in_9 = input_signal(name = 'in_9', input_temporal_form = 'arbitrary_spike_train_with_jitter', spike_times = np.asarray(spike_times__array[8]), source_type = 'qd', num_photons_per_spike = num_photons_per_spike)
    else:
        in_1 = input_signal(name = 'in_1', input_temporal_form = 'arbitrary_spike_train', spike_times = np.asarray(spike_times__array[0]))
        in_2 = input_signal(name = 'in_2', input_temporal_form = 'arbitrary_spike_train', spike_times = np.asarray(spike_times__array[1]))
        in_3 = input_signal(name = 'in_3', input_temporal_form = 'arbitrary_spike_train', spike_times = np.asarray(spike_times__array[2]))
        in_4 = input_signal(name = 'in_4', input_temporal_form = 'arbitrary_spike_train', spike_times = np.asarray(spike_times__array[3]))
        in_5 = input_signal(name = 'in_5', input_temporal_form = 'arbitrary_spike_train', spike_times = np.asarray(spike_times__array[4]))
        in_6 = input_signal(name = 'in_6', input_temporal_form = 'arbitrary_spike_train', spike_times = np.asarray(spike_times__array[5]))
        in_7 = input_signal(name = 'in_7', input_temporal_form = 'arbitrary_spike_train', spike_times = np.asarray(spike_times__array[6]))
        in_8 = input_signal(name = 'in_8', input_temporal_form = 'arbitrary_spike_train', spike_times = np.asarray(spike_times__array[7]))
        in_9 = input_signal(name = 'in_9', input_temporal_form = 'arbitrary_spike_train', spike_times = np.asarray(spike_times__array[8]))
    
    nine_pixel_add_inputs_to_synapses(syn_1, syn_2, syn_3, syn_4, syn_5, syn_6, syn_7, syn_8, syn_9, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9)
    
    return syn_1, syn_2, syn_3, syn_4, syn_5, syn_6, syn_7, syn_8, syn_9, spike_times__array

def min_max_finder(vec):
    _max = np.max(vec)
    _min = np.min(vec[np.where(vec > 1e-3)])
    return _min, _max

def and_not_coupling(phi_th_minus,phi_th_plus,s_or_min,s_or_max,c_or,s_and_min,s_and_max):
    c_max = phi_th_plus/s_and_max # ( phi_th_plus - s_or_max*c_or ) / s_and_min
    c_min = phi_th_minus/s_and_max # ( phi_th_minus - s_or_min*c_or ) / s_and_max
    # print('c_4and6__min = {}, c_4and6__max = {}'.format(c_4and6__min,c_4and6__max))
    c = c_min + (c_max-c_min)/20 # np.average([c_max,c_min])
    return c, c_min, c_max

def and_coupling(s_a,s_b):
    
    c_a = 1/(4*s_a)
    c_b = s_a*c_a/s_b
    
    return c_a, c_b

# =============================================================================
# end nine pixel helpers
# =============================================================================


