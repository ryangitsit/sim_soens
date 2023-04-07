import numpy as np
import pickle
import time

from numpy.random import default_rng
rng = default_rng()

from sim_soens.soen_utilities import dend_load_arrays_thresholds_saturations

'''
Supporting simulation-related functions
'''

d_params_ri = dend_load_arrays_thresholds_saturations('default_ri')
d_params_rtti = dend_load_arrays_thresholds_saturations('default_rtti')
ib__vec__ri = np.asarray(d_params_ri['ib__list'][:])
ib__vec__rtti = np.asarray(d_params_rtti['ib__list'][:])

def spd_response(phi_peak,tau_rise,tau_fall,hotspot_duration,t):
    term_time_ind = phi_peak * ( 1 - tau_rise/tau_fall )
    if t <= hotspot_duration:
        phi = term_time_ind*( 1 - np.exp( -t / tau_rise ) )
    elif t > hotspot_duration:
        phi = term_time_ind*(
            1-np.exp(-hotspot_duration/tau_rise)
            )*np.exp(-(t-hotspot_duration)/tau_fall)  
    return phi

def construct_dendritic_drives(obj):
                
    for dir_sig in obj.external_inputs:
        
        if hasattr(obj.external_inputs[dir_sig],'piecewise_linear'):
            dendritic_drive = dendritic_drive__piecewise_linear(
                obj.time_params['time_vec'],
                obj.external_inputs[dir_sig].piecewise_linear
                )
                        
        if hasattr(obj.external_inputs[dir_sig],'applied_flux'):
            dendritic_drive=obj.external_inputs[dir_sig].applied_flux*np.ones(
                [len(obj.phi_r_external__vec)]
                )

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
            input_signal__dd[t1_ind+jj+1]=input_signal__dd[t1_ind+jj]+(
                partial_time_vec[jj+1]-partial_time_vec[jj]
                )*slope
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
        input_signal__dd[ii+1]=input_signal__dd[ii]*(
            1-dt*(r1_vec[ii]+r2)/(L1+L2)
            )+dt*Ib*r1_vec[ii]/(L1+L2)
    
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
            input_signal__dd[ii] = value_off+(value_on-value_off)*(
                1-np.exp(-(time-t_rise)/tau_rise))
        if time >= t_fall:
            input_signal__dd[ii] = value_off+(value_on-value_off)*(
                1-np.exp(-(time-t_rise)/tau_rise)
                )*np.exp(-(time-t_fall)/tau_fall)
    
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
    pwl = [
        [0,value_off],
        [t_rise,value_on],
        [t_rise+t_pulse,value_on],
        [t_rise+t_pulse+t_fall,value_off]
        ]
    
    pulse = dendritic_drive__piecewise_linear(time_vec_sub,pwl)    
    num_pulses = np.floor((time_vec[-1]-t_start)/t_period).astype(int)        
    ind_start = (np.abs(np.asarray(time_vec)-t_start)).argmin()
    ind_pulse_end = (
        np.abs(np.asarray(time_vec)-t_start-t_rise-t_pulse-t_fall)
        ).argmin()
    ind_per_end = (np.abs(np.asarray(time_vec)-t_start-t_period)).argmin()
    num_ind_pulse = len(pulse) # ind_pulse_end-ind_start
    num_ind_per = ind_per_end-ind_start
    for ii in range(num_pulses):
        start = ind_start+ii*num_ind_per
        stop = ind_start+ii*num_ind_per+num_ind_pulse
        input_signal__dd[start:stop] = pulse[:]
        
    if t_start+num_pulses*t_period<=time_vec[-1] and t_start+(num_pulses+1)*t_period>=time_vec[-1]:
        ind_final = (
            np.abs(np.asarray(time_vec)-t_start-num_pulses*t_period)
            ).argmin()
        ind_end = (
            np.abs(
            np.asarray(time_vec)-t_start-num_pulses*t_period-t_rise-t_pulse-t_fall
            )).argmin()
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
    
def save_session_data(data_array=[],save_string='soen_sim',include_time=True):
    
    if include_time == True:
        tt = time.time()     
        s_str = save_string+'__'+time.strftime(
            '%Y-%m-%d_%H-%M-%S', 
            time.localtime(tt)
            )+'.dat'
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
    
    target_data__interpolated = np.interp(
        actual_data[0,:],
        target_data[0,:],
        target_data[1,:]
        )
    
    error = 0
    for ii in range(len(actual_data[0,:])-1):        
        error += np.abs(
            target_data__interpolated[ii+1]-actual_data[1,ii+1]
            )**2*( actual_data[0,ii+1]-actual_data[0,ii])
        
    norm = 0
    for ii in range(len(target_data[0,:])-1):
        norm += np.abs(
            target_data[1,ii+1]
            )**2*( target_data[0,ii+1]-target_data[0,ii]) 
  
    print('done calculating chi^2.\n')
    
    return error/norm


def phi_thresholds(neuron_object):

    if neuron_object.loops_present == 'ri':
        d_params_ri = dend_load_arrays_thresholds_saturations('default_ri')
        _ind_ib = (
            np.abs(
            np.array(d_params_ri["ib__list"][:]) - neuron_object.dend_soma.ib
            )).argmin()
        return [
            d_params_ri["phi_th_minus__vec"][_ind_ib],
            d_params_ri["phi_th_plus__vec"][_ind_ib]
            ]
    
    elif neuron_object.loops_present == 'rtti':
        d_params_rtti = dend_load_arrays_thresholds_saturations('default_rtti')
        _ind_ib = (
            np.abs(
            np.array(d_params_rtti["ib__list"][:]) - neuron_object.dend_soma.ib
            )).argmin()
        return [
            d_params_rtti["phi_th_minus__vec"][_ind_ib],
            d_params_rtti["phi_th_plus__vec"][_ind_ib]
            ]


