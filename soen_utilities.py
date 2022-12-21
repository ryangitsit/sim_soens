#%%
import numpy as np
from matplotlib import pyplot as plt
import pickle
import time
from scipy.optimize import fsolve
import sys
import copy
import matplotlib as mp
from soen_sim_data import *

# from soen_utilities import physical_constants, material_parameters, color_dictionary
# colors = color_dictionary()

# m_p = material_parameters()

fig_size = plt.rcParams['figure.figsize']

#%%

def physical_constants():

    p = dict(h = 6.62606957e-34,#Planck's constant in kg m^2/s
         hbar = 6.62606957e-34/(2*np.pi),
         hBar = 6.62606957e-34/(2*np.pi),
         hbar__eV_fs = 10.616133416243974/(2*np.pi),
         hBar__eV_fs = 10.616133416243974/(2*np.pi),
         c = 299792458,#speed of light in meters per second
         c__um_ns = 299792.458,#speed of light in meters per second
         epsilon0 = 8.854187817e-12,#permittivity of free space in farads per meter
         mu0 = 4*np.pi*1e-7,#permeability of free space in volt seconds per amp meter
         k = 1.3806e-23,#Boltzmann's constant in joules per kelvin
         kB = 1.3806e-23,#Boltzmann's constant in joules per kelvin
         kb = 1.3806e-23,#Boltzmann's constant in joules per kelvin
         kB__eV = 8.61703e-05,#Boltzmann's constant in electron volts per kelvin
         kb__ev = 8.61703e-05,#Boltzmann's constant in electron volts per kelvin
         kB_eV = 8.61703e-05,#Boltzmann's constant in electron volts per kelvin
         kb_ev = 8.61703e-05,#Boltzmann's constant in electron volts per kelvin
         e = 1.60217657e-19,#electron charge in coulombs
         m_e = 9.10938291e-31,#mass of electron in kg
         eV = 1.60217657e-19,#joules per eV
         ev = 1.60217657e-19,#joules per eV
         Ry = 9.10938291e-31*1.60217657e-19**4/(8*8.854187817e-12**2*(6.62606957e-34/2/np.pi)**3*299792458),#13.3*eV;#Rydberg in joules
         a0 = 4*np.pi*8.854187817e-12*(6.62606957e-34/2/np.pi)**2/(9.10938291e-31*1.60217657e-19**2),#estimate of Bohr radius
         Phi0 = 6.62606957e-34/(2*1.60217657e-19),#flux quantum
         Phi0__pH_ns = 6.62606957e3/(2*1.60217657),
         N_mole = 6.02214076e23, # atoms per mole
         golden_ratio = (1+np.sqrt(5))/2,
         gamma_euler = 0.5772  # Euler's constant
         )

    return p 

p = physical_constants()

#%%
def bias_ramp(t,dt_ramp,ii_max):
    
    if t < 0:
        ii = 0
        didt = 0
    elif t >= 0 and t <= dt_ramp:
        ii = ii_max*t/dt_ramp
        didt = ii_max/dt_ramp
    elif t >= dt_ramp:
        ii = ii_max
        didt = 0
    
    return ii, didt

def square_pulse_train(t,t_rise,t_hold,t_fall,value_on,value_off,period):
        
    _i = np.floor(t/period)
    # print('t = {}, t_rise = {}, t_hold = {}, t_fall = {}, value_on = {}, value_off = {}, period = {}, _i = {}'.format(t,t_rise,t_hold,t_fall,value_on,value_off,period,_i))
    if _i >= 0:
        
        # print('here1')
        _t = t-_i*period
        # print('_t = {}, t_rise = {}, t_rise+t_hold = {}, t_rise+t_hold+t_fall = {}'.format(_t,t_rise,t_rise+t_hold,t_rise+t_hold+t_fall))
        if _t <= t_rise:
            # print('here1a')
            s = _t*(value_on-value_off)/t_rise
            s_dot = (value_on-value_off)/t_rise
        elif _t > t_rise and _t <= t_rise+t_hold:
            # print('here1b')
            s = value_on
            s_dot = 0
        elif _t > t_rise+t_hold and _t < t_rise+t_hold+t_fall:
            # print('here1c')
            s = ( _t - (t_rise+t_hold) ) * (value_off-value_on) / t_fall + value_on
            s_dot = (value_off-value_on) / t_rise
        elif _t >= t_rise+t_hold+t_fall:
            # print('here1d')
            s = value_off
            s_dot = 0
            
    else:
        
        # print('here2')
        s = value_off
        s_dot = 0                
        
    return s, s_dot

def sigmoid__rise_and_fall(x_vec,x_on,x_off,width,amplitude,off_level):
    
    y = ( amplitude - off_level ) * ( 1 - ( np.exp( (x_vec-x_on) / width ) + 1 )**(-1) ) * ( ( np.exp( (x_vec-x_off) / width ) + 1 )**(-1) ) + off_level
    dydx = ( amplitude - off_level ) * (1/width)* ( ( ( np.exp((x_vec-x_off)/width) + 1 )**(-1) )  * ( np.exp((x_vec-x_on)/width) / ( np.exp((x_vec-x_on)/width) + 1 )**2) - ( 1 - ( np.exp((x_vec-x_on)/width) + 1 )**(-1) ) * np.exp((x_vec-x_off)/width) / ( np.exp((x_vec-x_off)/width) + 1 )**2 )
    
    return y, dydx

def sigmoid__rise(x_vec,x_on,width,amplitude,off_level):
    
    # y = ( amplitude - off_level ) * ( 1 - ( np.exp( (x_vec-x_on) / width ) + 1 )**(-1) ) + off_level
    y = ( amplitude - off_level ) * ( 1 - ( np.exp( (x_vec-x_on) / width ) + 1 )**(-1) ) + off_level
    
    return y

def line(x_vec,m,b):
    
    return m*x_vec+b

def exponential_pulse_train(t,beta_1,Ic,I_spd,period,tau_0,phi_a_max):
    
    r1 = 10e3
    r2 = 275.919 # 123.75
    L1 = 825e-9
    
    k = 0.5 # mutual inductance coupling factor
    L1_sq = beta_1*p['Phi0']/(2*np.pi*Ic)
    L2 = (1/L1_sq) * ( (p['Phi0']*phi_a_max)/(k*I_spd) )**2
    # L2 = 2.757e-9
    Ltot = L1+L2
    
    tau_plus = (Ltot/(r1+r2))/tau_0
    tau_minus = (Ltot/r2)/tau_0
    t0 = 300e-12/tau_0
        
    M = -k * np.sqrt(L1_sq*L2)
    
    i_spd = I_spd/Ic
    i0 = i_spd*(r1/(r1+r2))*(1-np.exp(-t0/tau_plus))
    
    # print('M = {}'.format(M))
    
    _i = np.floor(t/period)
    if _i >= 0:
        
        _t = t-_i*period
        if _t <= t0:
            i = i_spd*(r1/(r1+r2))*(1-np.exp(-_t/tau_plus))
            idot = (i_spd/tau_plus)*(r1/(r1+r2))*np.exp(-_t/tau_plus)
        elif _t > t0:
            i = i0*np.exp(-_t/tau_minus)
            idot = -(i_spd/tau_minus)*(r1/(r1+r2))*(1-np.exp(-t0/tau_plus))*np.exp(-(_t-t0)/tau_minus)
            
        s = M*i*Ic/p['Phi0']  
        s_dot = M*idot*Ic/p['Phi0']
            
    else:
        
        s = 0
        s_dot = 0                
        
    return s, s_dot

def piecewise_linear(t,times_values_list):
    
    s = 0
    s_dot = 0
    for ii in range(len(times_values_list)-1):
        _t1 = times_values_list[ii][0]
        _t2 = times_values_list[ii+1][0]
        if t >= _t1 and t < _t2:
            _m = ((times_values_list[ii+1][1]-times_values_list[ii][1])/(_t2-_t1))
            _b = times_values_list[ii][1]
            s = _m * (t-_t1) + _b
            s_dot = _m
    
    return s, s_dot


def lorentzian(omega,omega_0,**kwargs):
    
    if 'Q' in kwargs:
        Q = kwargs['Q']
        L = (1/np.pi) * (omega_0/(2*Q)) / ( (omega-omega_0)**2 + (omega_0/(2*Q))**2 )
    if 'tau' in kwargs:
        tau = kwargs['tau']
        L = (1/np.pi) * (1/(2*tau)) / ( (omega-omega_0)**2 + (1/(2*tau))**2 )
        
    if 'Q' not in kwargs and 'tau' not in kwargs:
        raise ValueError('[lorentzian] must specify Q or tau')
    if 'Q' in kwargs and 'tau' in kwargs:
        raise ValueError('[lorentzian] specify either Q or tau, but not both')
    
    return L


def fermi_distribution__eV(E_vec,E_f,T):
    
    return ( np.exp( (E_vec-E_f) / (p['kB__eV']*T) ) + 1 )**(-1)


def omega_LRC(L,R,C):
    
    omega_r = np.sqrt( (L*C)**(-1) - 0.25*(R/L)**(2) )
    omega_i = R/(2*L)
    
    return omega_r, omega_i 

def dend_save_rate_array(params,ib__list,phi_r__array,r_fq__array,i_di__array):

    data_array = dict()
    data_array['ib__list'] = ib__list
    data_array['phi_r__array'] = phi_r__array
    data_array['params'] = params
        
    if params['loops_present'] == 'r':
        save_string = 'rate_array__dend_{}__beta_c_{:06.4f}__beta_1_{:07.4f}__beta_2_{:07.4f}__ib_i_{:06.4f}__ib_f_{:06.4f}__num_ib_{:d}__d_phi_a_{:6.4f}'.format(params['loops_present'],params['beta_c'],params['beta_1'],params['beta_2'],ib__list[0],ib__list[-1],len(ib__list),params['d_phi_a'])

        data_array['R_fq__array'] = r_fq__array
    
    if params['loops_present'] == 'ri':
        tau_di = 1e9*params['tau_di']*params['tau_0']
        if tau_di >= 1e4: 
            save_string = 'ra_dend_{}__beta_c_{:06.4f}__beta_1_{:07.4f}__beta_2_{:07.4f}__beta_di_{:07.5e}__tau_di_long__ib_i_{:06.4f}__ib_f_{:06.4f}__d_phi_r_{:6.4f}'.format(params['loops_present'],params['beta_c'],params['beta_1'],params['beta_2'],params['beta_di'],ib__list[0],ib__list[-1],params['d_phi_r'])
        else:        
            save_string = 'ra_dend_{}__beta_c_{:06.4f}__beta_1_{:07.4f}__beta_2_{:07.4f}__beta_di_{:07.5e}__tau_di_{:07.0f}ns__ib_i_{:06.4f}__ib_f_{:06.4f}_{:5.3f}__d_phi_r_{:6.4f}'.format(params['loops_present'],params['beta_c'],params['beta_1'],params['beta_2'],params['beta_di'],1e9*params['tau_di']*params['tau_0'],ib__list[0],ib__list[-1],params['d_phi_r'])
        data_array['r_fq__array'] = r_fq__array
        data_array['i_di__array'] = i_di__array
    
    if params['loops_present'] == 'rtti':
        tau_di = 1e9*params['tau_di']*params['tau_0']
        if tau_di >= 1e4: 
            save_string = 'ra_dend_{}_beta_c_{:05.3f}_b1_{:05.3f}_b2_{:05.3f}_b3_{:05.3f}_b4_{:05.3f}_b_di_{:05.3e}_tau_di_long_ib1_i_{:05.3f}_ib1_f_{:05.3f}_ib2_{:05.3f}_ib3_{:05.3f}_d_phi_r_{:5.3f}'.format(params['loops_present'],params['beta_c'],params['beta_1'],params['beta_2'],params['beta_3'],params['beta_4'],params['beta_di'],ib__list[0],ib__list[-1],params['ib2'],params['ib3'],params['d_phi_r'])
        else:        
            save_string = 'ra_dend_{}_beta_c_{:05.3f}_b1_{:05.3f}_b2_{:05.3f}_b3_{:05.3f}_b4_{:05.3f}_b_di_{:05.3e}_tau_di_{:07.0f}ns_ib1_i_{:05.3f}_ib1_f_{:05.3f}_ib2_{:05.3f}_ib3_{:05.3f}_d_phi_r_{:5.3f}'.format(params['loops_present'],params['beta_c'],params['beta_1'],params['beta_2'],params['beta_3'],params['beta_4'],params['beta_di'],1e9*params['tau_di']*params['tau_0'],ib__list[0],ib__list[-1],params['ib2'],params['ib3'],params['d_phi_r'])
        data_array['r_fq__array'] = r_fq__array
        data_array['i_di__array'] = i_di__array
        
    print('\n\nsaving session data ...\n\n')    
    _path = pathfinder()
    tt = time.time()             
    # with open('soen_sim_data/{}__{}.soen'.format(save_string,time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))), 'wb') as data_file:
    #     pickle.dump(data_array, data_file) 
    with open('{}{}{}__{}.soen'.format(_path,'/soen_sim_data/',save_string,time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))), 'wb') as data_file:
        pickle.dump(data_array, data_file)

def pathfinder():
    for _str in sys.path:
        dir_index = _str.find("sim_soens")
        if _str[dir_index:dir_index+9] == 'sim_soens':
            path = _str.replace('\\','/')[:dir_index+9] +'/'
            break
    return path

def dend_load_rate_array(load_string):

    _path = pathfinder()

    if load_string == 'default' or load_string == 'default_ri':
        # _load_string = 'ra_dend_ri__beta_c_0.3000__beta_1_01.5708__beta_2_01.5708__beta_di_6.28319e+03__tau_di_long__ib_i_1.3673__ib_f_2.0673__d_ib_0.050__d_phi_r_0.0100__working_master'
        # _load_string = 'ra_dend_ri__beta_c_0.3000__beta_1_01.5708__beta_2_01.5708__beta_di_6.28319e+03__tau_di_long__ib_i_1.3524__ib_f_2.0524__d_ib_0.050__d_phi_r_0.0100__working_master'
        _load_string = 'ra_dend_ri__beta_c_0.3000__beta_1_01.5708__beta_2_01.5708__beta_di_6.28319e+03__tau_di_long__ib_i_1.3524__ib_f_2.0524__d_ib_0.050__d_phi_r_0.0025__working_master'
    elif load_string == 'default_rtti':
        _load_string = 'ra_dend_rtti__beta_c_0.300__b1_1.571_b2_1.571_b3_3.142_b4_3.142_b_di_6.28319e+03_tau_di_long_ib1_i_1.500_ib1_f_2.650_ib2_0.350_ib3_0.700_d_phi_r_0.010__working_master' # 'ra_dend_rtti__beta_c_0.3000__beta_1_01.5708__beta_2_01.5708__beta_3_03.1416__beta_4_03.1416_beta_di_6.28319e+03__tau_di_long__ib_i_1.5000__ib_f_2.6500__d_phi_r_0.0100__working_master'
    else:
        _load_string = load_string
    with open('{}{}{}.soen'.format(_path,'/soen_sim_data/',_load_string), 'rb') as data_file:    
        data_array_imported = pickle.load(data_file)   
    
    if 'params' in data_array_imported:
        params_output = data_array_imported['params']
    else:
        params_output = 'params not available in this data set'
        
    if 'phi_a__array' in data_array_imported:
        if 'phi_r__array' not in 'phi_a__array':
            data_array_imported.update({'phi_r__array': data_array_imported['phi_a__array']})
    
    return data_array_imported['ib__list'], data_array_imported['phi_r__array'], data_array_imported['i_di__array'], data_array_imported['r_fq__array'], params_output, _load_string


def dend_load_thresholds_saturations(load_string):

    _path = pathfinder()

    if load_string == 'default' or load_string == 'default_ri':
        _load_string = 'ra_dend_ri__beta_c_0.3000__beta_1_01.5708__beta_2_01.5708__beta_di_6.28319e+03__tau_di_long__ib_i_1.3524__ib_f_2.0524__d_ib_0.050__d_phi_r_0.0025__thresholds_saturations'
    elif load_string == 'default_rtti':
        _load_string = 'ra_dend_rtti__beta_c_0.3000__beta_1_01.5708__beta_2_01.5708__beta_di_6.28319e+03__tau_di_long__ib_i_1.5000__ib_f_2.6500__d_phi_r_0.0100__thresholds_saturations'
    else:
        _load_string = load_string
         
    with open('{}{}{}.soen'.format(_path,'/soen_sim_data/',_load_string), 'rb') as data_file:    
        data_array_imported = pickle.load(data_file)   
    
    return data_array_imported['phi_th_plus__vec'], data_array_imported['phi_th_minus__vec'], data_array_imported['s_max_plus__vec'], data_array_imported['s_max_minus__vec'], data_array_imported['s_max_plus__array'], data_array_imported['s_max_minus__array']


def dend_load_arrays_thresholds_saturations(load_string):
    
    ib__list, phi_r__array, i_di__array, r_fq__array, _, _ = dend_load_rate_array(load_string)
    phi_th_plus__vec, phi_th_minus__vec, s_max_plus__vec, s_max_minus__vec, s_max_plus__array, s_max_minus__array = dend_load_thresholds_saturations(load_string)
    
    return ib__list, phi_r__array, i_di__array, r_fq__array, phi_th_plus__vec, phi_th_minus__vec, s_max_plus__vec, s_max_minus__vec, s_max_plus__array, s_max_minus__array


#%% determine depth of dendritic tree

def depth_of_dendritic_tree(soen_object):
    
    if type(soen_object).__name__ == 'neuron':
        dendrite = soen_object.dend__nr_ni
        
    def find_synapses(_dendrite,counter):
        
        if len(_dendrite.synaptic_inputs) == 0:
            counter += 1
            for input_dendrite in _dendrite.dendritic_inputs:
                _inner_counter = find_synapses(_dendrite.dendritic_inputs[input_dendrite],0)
            counter += _inner_counter
                
        return counter
    
    depth_of_tree = find_synapses(dendrite,0)
    
    return depth_of_tree + 1

#%% JJs

def get_jj_params(Ic,beta_c):
    
    gamma = 1.5e-9 # 5e-15/1e-6 # 1.5e-9 is latest value from David # proportionality between capacitance and Ic (units of farads per amp)
    c_j = gamma*Ic # JJ capacitance
    r_j = np.sqrt( (beta_c*p['Phi0']) /  (2*np.pi*c_j*Ic) )
    tau_0 = p['Phi0']/(2*np.pi*Ic*r_j)
    V_j = Ic*r_j
    omega_c = 2*np.pi*Ic*r_j/p['Phi0']
    omega_p = np.sqrt(2*np.pi*Ic/(p['Phi0']*c_j))
    
    return {'c_j': c_j, 'r_j': r_j, 'tau_0': tau_0, 'Ic': Ic, 'beta_c': beta_c, 'gamma': gamma, 'V_j': V_j, 'omega_c': omega_c, 'omega_p': omega_p}

def Ljj(critical_current,current):
    
    norm_current = np.max([np.min([current/critical_current,1]),1e-9])
    L = (3.2910596281416393e-16/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

def Ljj_dimensionless(normalized_current):
    
    # norm_current = np.max([np.min([normalized_current,1]),1e-9])
    
    return np.arcsin(normalized_current)/(normalized_current)

def Ljj_pH(critical_current,current):
    
    norm_current = np.max([np.min([current/critical_current,1]),1e-9])
    L = (3.2910596281416393e2/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

def Ljj_pH__vec(critical_current,current):
    
    norm_current = current/critical_current
    L = (3.2910596281416393e2/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

def Ljj__vec(critical_current,current):
    
    norm_current = current/critical_current
    L = (3.2910596281416393e-16/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

#%% MOSFETs

def mos_c_i(epsilon_i,d_i): # capacitance per unit area (epsilon_i = permittivity of gate insulator, d_i = thickness of gate insulator)
    
    return epsilon_i/d_i

def mos_V_fb(gate_contact_material = 'aluminum'): # flat-band voltage

    if gate_contact_material == 'aluminum':
        Phi_m = (4.26+4.06)/2 # work function of aluminum in eV (range from 4.06-4.26, https://en.wikipedia.org/wiki/Work_function#Work_functions_of_elements)
    
    Phi_s = (4.85+4.60)/2 # work function of silicon in eV (range from 4.6-4.85, https://en.wikipedia.org/wiki/Work_function#Work_functions_of_elements)
    V_fb = (Phi_m - Phi_s) # mosfet doc eq 1.1 (omitting q because work functions are in eV)

    return V_fb

def nmos_V_t(T,N_a,c_i,epsilon_s):
    
    n_i = m_p['n_i__si']
    phi_b = (p['kB']*T/p['e']) *np.log(N_a/n_i) # grimoire eq 105, mosfet doc eq 1.2
    V_t = mos_V_fb() + 2*phi_b + np.sqrt(4*epsilon_s*p['e']*N_a*phi_b)/c_i
    
    return V_t

def pmos_V_t(T,N_d,c_i,epsilon_s):
    
    Vsb = 0
    n_i = mp['n_i__si']
    phi_b = (p['kB']*T/p['e']) *np.log(N_d/n_i) # grimoire eq 105, mosfet doc eq 1.2
    V_t = -mos_V_fb() - 2*phi_b - np.sqrt(2*epsilon_s*p['e']*N_d*(2*phi_b-Vsb))/c_i
    
    return V_t

def nmos_ivv__charge_control(T,W,L,mu_n,c_i,N_a,V_ds,V_gs):
    
    V_t = nmos_V_t(T,N_a,c_i,m_p['epsilon_si'])
    V_gt = V_gs - V_t
    
    if V_gt > 0:
        
        V_sat = V_gt
        # print('V_t = {:5.2f}\nV_sat = {:5.2f}\n'.format(V_t,V_sat))
        _pf = ((W*mu_n*c_i)/(L)) # prefactor
        if V_ds <= V_sat:
            I_ds = _pf * (V_gt - V_ds/2) * V_ds
        elif V_ds > V_sat:
            I_ds = _pf * (V_gt**2)/2
        else:
            I_ds = 0
            
    else:
        
        I_ds = 0
            
    return I_ds

def pmos_ivv__charge_control(T,W,L,mu_p,c_i,N_d,V_ds,V_gs):
    
    # search for pmos.pdf
    V_t = pmos_V_t(T,N_d,c_i,mp['epsilon_si'])    
    V_gt = V_gs - V_t
    
    if V_gt < 0:
        
        V_sat = V_gt
        _pf = ((W*mu_p*c_i)/(L)) # prefactor
        if V_ds >= V_sat:
            I_ds = _pf * (V_gt - V_ds/2) * V_ds
        elif V_ds < V_sat:
            I_ds = _pf * (V_gt**2)/2
            
    else:
        
        I_ds = 0
            
    return -I_ds

def nmos_ddt_ivv__charge_control(T,W,L,mu_n,c_i,N_a,V_ds,dV_ds_dt,V_gs,dV_gs_dt):
    
    V_t = nmos_V_t(T,N_a,c_i,mp['epsilon_si'])
    
    V_gt = V_gs - V_t
    
    if V_gt > 0:
        
        V_sat = V_gt
        # print('V_t = {:5.2f}\nV_sat = {:5.2f}\n'.format(V_t,V_sat))
        _pf = ((W*mu_n*c_i)/(L)) # prefactor
        if V_ds <= V_sat:
            dI_ds_dt = _pf * ( V_ds*dV_gs_dt + V_gt*dV_ds_dt - V_ds*dV_ds_dt )
        elif V_ds > V_sat:
            dI_ds_dt = _pf * (V_gt*dV_gs_dt)
            
    else:
        
        dI_ds_dt = 0
            
    return dI_ds_dt

def nmos_Ids_sat(T,W,L,mu_n,c_i,N_a,V_gs):
    
    V_t = nmos_V_t(T,N_a,c_i,mp['epsilon_si'])
    V_gt = V_gs - V_t
    if V_gt > 0:
        V_ds_sat = V_gt
        I_ds_sat = ((W*mu_n*c_i)/(L)) * (V_gt**2)/2
    else:
        V_ds_sat = 0
        I_ds_sat = 0
    return I_ds_sat, V_ds_sat

def nmos_inverter_IV(N_a__vec,ri__vec,V_in__vec, params):

    V_out__array = np.zeros([len(N_a__vec),len(ri__vec),len(V_in__vec)])
    
    for mm in range(len(N_a__vec)):
        N_a = N_a__vec[mm]
        
        for nn in range(len(ri__vec)):
            ri = ri__vec[nn]
    
            for ii in range(len(V_in__vec)):
                V_in = V_in__vec[ii]
                args = (params['T'],params['W'],params['L'],params['mu_n'],params['c_i'],N_a,ri,params['Vb'],V_in)
                V_t = nmos_V_t(params['T'],N_a,params['c_i'],mp['epsilon_si'])
                if V_in >= V_t:
                    V_out_guess = params['Vb']
                else:
                    V_out_guess = 0
                V_out = fsolve(nmos_inverter_def, V_out_guess, args)
                V_out__array[mm,nn,ii] = V_out[0]
        
    # to plot, see _plotting__mosfet.py
    # plot_nmos_inverter_IV(N_a__vec,ri__vec,V_in__vec,V_out__array,params)
        
    return V_out__array

def nmos_inverter_def(Vout_guess,T,W,L,mu_n,c_i,N_a,ri,V_ds,V_g):
    
    return Vout_guess + ri*nmos_ivv__charge_control(T,W,L,mu_n,c_i,N_a,Vout_guess,V_g) - V_ds

#%% diodes / LEDs / transmitter

def LED_diode_iv(T,W,L,N_a,N_d,n_i,V):

    A = W*L # diode area
    
    tau_np = 40e-9 # lifetime of electrons on p side
    tau_pn = 40e-9 # lifetime of holes on n side
    
    mu_pn = 100e-4 # %450e-4;%mobility of holes on n side
    mu_np = 250e-4 # %700e-4;%mobility of electrons on p side
    
    Dp = (p['kB']*T/p['e'])*mu_pn
    Dn = (p['kB']*T/p['e'])*mu_np
    
    Lp = np.sqrt(Dp*tau_pn)
    Ln = np.sqrt(Dn*tau_np)
    
    I0 = p['e']*A*( (Dp/Lp)*(n_i**2/N_d) + (Dn/Ln)*(n_i**2/N_a) )

    I_pn = I0 * ( np.exp(p['e']*V/(p['kB']*T)) - 1 );

    return I_pn

def transmitter_save_data(data_dict):
    
    params = data_dict['params']
    if params['diode'] == 'IV':
        _str = 'um3'
        _factor = 1e-18
    elif params['diode'] == 'III-V':
        _str = 'um2'
        _factor = 1e-12
    save_string = 'transmit__{}_process__group_{}_emitter__c_a_{:3.1e}fF_per_um2__rho_qd_{:3.1e}per_{}__W_led_{:03.1f}um__L_led_{:06.2f}um__N_qd_{:3.1e}'.format(params['process'],params['diode'],params['c_a']*1e3,_factor*params['rho_qd'],_str,1e6*params['W_led'],1e6*params['L_led'],params['N_qd'])
    data_array = {'rho_qd': params['rho_qd'], 'W_led': params['W_led'], 'L_led': params['L_led'], 'C_led': params['C_led'], 'c_a': params['c_a'],
                  'N_qd': params['N_qd'], 'N_e': data_dict['N_e_led'], 'process': params['process'], 'diode': params['diode'],
                  'I_led': data_dict['I_led'], 'I_cap': data_dict['I11'],'time_vec': data_dict['time_vec'], 't_on_tron': data_dict['t_on_tron']}
    print('\n\nsaving session data ...\n\n')
    tt = time.time()             
    with open('data/{}__{}.soen'.format(save_string,time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))), 'wb') as data_file:
        pickle.dump(data_array, data_file) 
    
    return

def transmitter_load_data(load_string):
    
    with open('{}{}.soen'.format('transmitter_data/',load_string), 'rb') as data_file:         
        data_array_imported = pickle.load(data_file)   
    
    N_e = data_array_imported['N_e']
    N_qd = data_array_imported['N_qd']
    C_led = data_array_imported['C_led']
    I_cap_vec = data_array_imported['I_cap']
    I_led_vec = data_array_imported['I_led']
    time_vec = data_array_imported['time_vec']
    t_on_tron = data_array_imported['t_on_tron']
    
    return time_vec, I_led_vec, I_cap_vec, C_led, N_qd, N_e, t_on_tron


#%% meander dimensions

def meander_dimensions__inductance(material,inductance,gap, **kwargs):
    
    if material == 'MoSi':
        inductance_per_square = mp['Lsq__MoSi']
        alpha = mp['alpha__MoSi']
    num_squares = inductance/inductance_per_square
    if 'current' in kwargs:
        w_wire = kwargs['current']*alpha
    
    # ao = 2*(w_wire+gap)
    # N1 = -1+np.sqrt(1-(4-num_squares)/(2*ao)*w_wire-3*gap/(2*ao))
    # w_array = (2*w_wire+gap)+N1*ao #%% see pg 43 in notebook started 20160425
    # print('inductance_per_square = {:5.1f}pH\ninductance = {:5.1f}nH\nnum_squares = {:f}\nw_wire = {:f} nm\ngap = {:f} nm\nw_meander = {:f} um\narea_meander = {:f} um^2\n\n'.format(inductance_per_square*1e12,inductance*1e9,num_squares,w_wire*1e9,gap*1e9,w_array*1e6,w_array**2*1e12))
    
    # simpler expression
    w_array = np.sqrt( inductance * w_wire * (w_wire+gap) / inductance_per_square )
    # print('inductance_per_square = {:5.1f}pH\ninductance = {:5.1f}nH\nnum_squares = {:f}\nw_wire = {:f} nm\ngap = {:f} nm\nw_meander = {:f} um\narea_meander = {:f} um^2\n\n'.format(inductance_per_square*1e12,inductance*1e9,num_squares,w_wire*1e9,gap*1e9,w_array*1e6,w_array**2*1e12))
    
    data_dict = {'w_array': w_array, 'w_wire': w_wire, 'num_squares': num_squares}
    
    return data_dict

def meander_dimensions__resistance(material,resistance,gap, **kwargs):
    
    if material == 'MoSi':
        resistance_per_square = mp['rsq__MoSi']
        alpha = mp['alpha__MoSi']
    num_squares = resistance/resistance_per_square
    if 'current' in kwargs:
        w_wire = kwargs['current']*alpha
    
    # ao = 2*(w_wire+gap)
    # N1 = -1+np.sqrt(1-(4-num_squares)/(2*ao)*w_wire-3*gap/(2*ao))
    # w_array = (2*w_wire+gap)+N1*ao #%% see pg 43 in notebook started 20160425
    # print('inductance_per_square = {:5.1f}pH\ninductance = {:5.1f}nH\nnum_squares = {:f}\nw_wire = {:f} nm\ngap = {:f} nm\nw_meander = {:f} um\narea_meander = {:f} um^2\n\n'.format(inductance_per_square*1e12,inductance*1e9,num_squares,w_wire*1e9,gap*1e9,w_array*1e6,w_array**2*1e12))
    
    # simpler expression
    w_array = np.sqrt( resistance * w_wire * (w_wire+gap) / resistance_per_square )
    # print('inductance_per_square = {:5.1f}pH\ninductance = {:5.1f}nH\nnum_squares = {:f}\nw_wire = {:f} nm\ngap = {:f} nm\nw_meander = {:f} um\narea_meander = {:f} um^2\n\n'.format(inductance_per_square*1e12,inductance*1e9,num_squares,w_wire*1e9,gap*1e9,w_array*1e6,w_array**2*1e12))
    
    data_dict = {'w_array': w_array, 'w_wire': w_wire, 'num_squares': num_squares}
    
    return data_dict


#%% misc helpers

def exp_fitter(x,y,index1,index2, rise_or_fall = 'rise'): 
    
    # fit function y(x) to an exponential on the interval from index1 to index2
    # rise: y(x) = A * ( 1 - exp( -(x-x0)/tau) )
    # fall: y(x) = A * exp( -(x-x0)/tau)
    
    A = np.max(y)
    if rise_or_fall == 'rise':
        
        func1 = np.log(1-y[index1:index2]/A)
        x_part = x[index1:index2]
        e_fit = np.polyfit(x_part,func1,1)
        tau = -1/e_fit[0]
        x0_over_tau = e_fit[1]
        y_fit = A * ( 1 - np.exp(-x_part/tau + x0_over_tau) )
        
    elif rise_or_fall == 'fall':
        
        func1 = np.log(y[index1:index2]/A)
        x_part = x[index1:index2]
        e_fit = np.polyfit(x_part,func1,1)
        tau = -1/e_fit[0]
        x0_over_tau = e_fit[1]
        y_fit = A * np.exp(-x_part/tau + x0_over_tau)
                
    return x_part, y_fit, tau


# =============================================================================
# network average path length
# =============================================================================

def k_of_N_and_L(N,L):
    return np.exp( ( np.log(N) - p['gamma_euler'] ) / ( L - 1/2 ) )

# =============================================================================
# end network average path length
# =============================================================================


# =============================================================================
# distributions
# =============================================================================

def gaussian(sigma,mu,x):
    
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp( -(1/2) * ( (x-mu)/sigma )**2 )

def poisson(lam,k):
    if k > 30:
        denominator = np.sqrt(2*np.pi*k)*(k/np.exp(1))**k
    else:
        denominator = np.math.factorial(k)
    if denominator > 1e16:
        distribution = 0
    else:
        distribution = (lam**k)*np.exp(-lam)/denominator
    return distribution

def power_law(alpha,n_min,n_max,n):
    A = (1-alpha)/(n_max**(1-alpha)-n_min**(1-alpha))
    distribution = A*n**(-alpha)
    return distribution, A

def log_normal(sigma,mu,x):
    distribution = (1/(x*sigma*np.sqrt(2*np.pi))) * np.exp( - ( (np.log(x)-mu)**2 / (2*sigma**2) ) )
    mean = np.exp(mu+(sigma**2)/2)
    variance = (np.exp(sigma**2)-1)*(np.exp(2*mu+sigma**2))
    return distribution, mean, variance

def coth(x):
    return 1/np.tanh(x)

# =============================================================================
# end distributions
# =============================================================================


# =============================================================================
# superconducting wires
# =============================================================================

# inductance per unit length of wire above ground plane
def L_per_length(K,lam1,b1,lam2,b2,d,w):
    
    # K is fringe factor
    # lam1 is london penetration depth of strip, lam2 is penetration depth of ground plane
    # b1 is thickness of strip, b2 is thickness of ground plane
    # d is separation between strip and ground plane
    # w is width of wire
    
    return (p['mu0']*d/(K*w)) * ( 1 + (lam1/d)*coth(b1/lam1) + (lam2/d)*coth(b2/lam2) )

# inductance per square of wire above ground plane
def L_per_square(K,lam1,b1,lam2,b2,d):
    
    # K is fringe factor
    # lam1 is london penetration depth of strip, lam2 is penetration depth of ground plane
    # b1 is thickness of strip, b2 is thickness of ground plane
    # d is separation between strip and ground plane
    
    return (p['mu0']*d/K) * ( 1 + (lam1/d)*coth(b1/lam1) + (lam2/d)*coth(b2/lam2) )

def C_per_length(eps_r,w,d,C0):
    
    return np.max([eps_r*p['epsilon0']*w/d, C0]) # eps_r*p['epsilon0']*w/d + C0

# =============================================================================
# end superconducting wires
# =============================================================================

# =============================================================================
# materials
# =============================================================================

def material_parameters():
    
    p = physical_constants()
    
    mp = dict(epsilon_sio2 = (1.46**2)*p['epsilon0'], # dc permittivity of silicon dioxide
              epsilon_si = (3.48**2)*p['epsilon0'], # dc permittivity of silicon
              epsilon_gaas = 12.85*p['epsilon0'], # dc permittivity of GaAs
              epsilon_algaas = 12.9*p['epsilon0'], # dc permittivity of AlGaAs
              n_i__si = 1e16, # electrons per meter cubed in silicon at 300K, grimoire table 5
              n_i__gaas = 2.25e12, # electrons per meter cubed in GaAs at 300K, grimoire table 6
              Lsq__MoSi = 160e-12, # kinetic inductance per square for MoSi
              Lsq__WSi = 400e-12, # kinetic inductance per square for WSi
              rsq__MoSi = 500, # resistance per square for MoSi,
              alpha__MoSi = 2e-2, # w_wire = alpha Idi_sat
              Eg__gaas = 2.275e-19 # GaAs band gap in joules
              )
    
    return mp

# =============================================================================
# end materials
# =============================================================================


# =============================================================================
# colors
# =============================================================================

def color_dictionary():

    colors = dict()    

    ## define colors
    #blues  lightest to darkest
    blueVec1 = np.array([145,184,219]); colors['blue1'] = blueVec1/256;
    blueVec2 = np.array([96,161,219]); colors['blue2'] = blueVec2/256;
    blueVec3 = np.array([24,90,149]); colors['blue3'] = blueVec3/256;
    blueVec4 = np.array([44,73,100]); colors['blue4'] = blueVec4/256;
    blueVec5 = np.array([4,44,80]); colors['blue5'] = blueVec5/256;
    colors['blue1.5'] = (blueVec1+blueVec2)/(512);
    colors['blue2.5'] = (blueVec2+blueVec3)/(512);
    colors['blue3.5'] = (blueVec3+blueVec4)/(512);
    colors['blue4.5'] = (blueVec4+blueVec5)/(512);
    #reds  lightest to darkest
    redVec1 = np.array([246,177,156]); colors['red1'] = redVec1/256;
    redVec2 = np.array([246,131,98]); colors['red2'] = redVec2/256;
    redVec3 = np.array([230,69,23]); colors['red3'] = redVec3/256;
    redVec4 = np.array([154,82,61]); colors['red4'] = redVec4/256;
    redVec5 = np.array([123,31,4]); colors['red5'] = redVec5/256;
    colors['red1.5'] = (redVec1+redVec2)/(512);
    colors['red2.5'] = (redVec2+redVec3)/(512);
    colors['red3.5'] = (redVec3+redVec4)/(512);
    colors['red4.5'] = (redVec4+redVec5)/(512);
    #greens  lightest to darkest
    greenVec1 = np.array([142,223,180]); colors['green1'] = greenVec1/256;
    greenVec2 = np.array([89,223,151]); colors['green2'] = greenVec2/256;
    greenVec3 = np.array([16,162,84]); colors['green3'] = greenVec3/256;
    greenVec4 = np.array([43,109,74]); colors['green4'] = greenVec4/256;
    greenVec5 = np.array([3,87,42]); colors['green5'] = greenVec5/256;
    colors['green1.5'] = (greenVec1+greenVec2)/(512);
    colors['green2.5'] = (greenVec2+greenVec3)/(512);
    colors['green3.5'] = (greenVec3+greenVec4)/(512);
    colors['green4.5'] = (greenVec4+greenVec5)/(512);
    #yellows  lightest to darkest
    yellowVec1 = np.array([246,204,156]); colors['yellow1'] = yellowVec1/256;
    yellowVec2 = np.array([246,185,98]); colors['yellow2'] = yellowVec2/256;
    yellowVec3 = np.array([230,144,23]); colors['yellow3'] = yellowVec3/256;
    yellowVec4 = np.array([154,115,61]); colors['yellow4'] = yellowVec4/256;
    yellowVec5 = np.array([123,74,4]); colors['yellow5'] = yellowVec5/256;
    colors['yellow1.5'] = (yellowVec1+yellowVec2)/(512);
    colors['yellow2.5'] = (yellowVec2+yellowVec3)/(512);
    colors['yellow3.5'] = (yellowVec3+yellowVec4)/(512);
    colors['yellow4.5'] = (yellowVec4+yellowVec5)/(512);
    
    #blue grays
    gBlueVec1 = np.array([197,199,202]); colors['bluegrey1'] = gBlueVec1/256;
    gBlueVec2 = np.array([195,198,202]); colors['bluegrey2'] = gBlueVec2/256;
    gBlueVec3 = np.array([142,145,149]); colors['bluegrey3'] = gBlueVec3/256;
    gBlueVec4 = np.array([108,110,111]); colors['bluegrey4'] = gBlueVec4/256;
    gBlueVec5 = np.array([46,73,97]); colors['bluegrey5'] = gBlueVec5/256;
    colors['bluegrey1.5'] = (gBlueVec1+gBlueVec2)/(512);
    colors['bluegrey2.5'] = (gBlueVec2+gBlueVec3)/(512);
    colors['bluegrey3.5'] = (gBlueVec3+gBlueVec4)/(512);
    colors['bluegrey4.5'] = (gBlueVec4+gBlueVec5)/(512);
    #red grays
    gRedVec1 = np.array([242,237,236]); colors['redgrey1'] = gRedVec1/256;
    gRedVec2 = np.array([242,235,233]); colors['redgrey2'] = gRedVec2/256;
    gRedVec3 = np.array([230,231,218]); colors['redgrey3'] = gRedVec3/256;
    gRedVec4 = np.array([172,167,166]); colors['redgrey4'] = gRedVec4/256;
    gRedVec5 = np.array([149,88,71]); colors['redgrey5'] = gRedVec5/256;
    colors['redgrey1.5'] = (gRedVec1+gRedVec2)/(512);
    colors['redgrey2.5'] = (gRedVec2+gRedVec3)/(512);
    colors['redgrey3.5'] = (gRedVec3+gRedVec4)/(512);
    colors['redgrey4.5'] = (gRedVec4+gRedVec5)/(512);
    #green grays
    gGreenVec1 = np.array([203,209,206]); colors['greengrey1'] = gGreenVec1/256;
    gGreenVec2 = np.array([201,209,204]); colors['greengrey2'] = gGreenVec2/256;
    gGreenVec3 = np.array([154,162,158]); colors['greengrey3'] = gGreenVec3/256;
    gGreenVec4 = np.array([117,122,119]); colors['greengrey4'] = gGreenVec4/256;
    gGreenVec5 = np.array([50,105,76]); colors['greengrey5'] = gGreenVec5/256;
    colors['greengrey1.5'] = (gGreenVec1+gGreenVec2)/(512);
    colors['greengrey2.5'] = (gGreenVec2+gGreenVec3)/(512);
    colors['greengrey3.5'] = (gGreenVec3+gGreenVec4)/(512);
    colors['greengrey4.5'] = (gGreenVec4+gGreenVec5)/(512);
    #yellow grays
    gYellowVec1 = np.array([242,240,236]); colors['yellowgrey1'] = gYellowVec1/256;
    gYellowVec2 = np.array([242,239,233]); colors['yellowgrey2'] = gYellowVec2/256;
    gYellowVec3 = np.array([230,225,218]); colors['yellowgrey3'] = gYellowVec3/256;
    gYellowVec4 = np.array([172,169,166]); colors['yellowgrey4'] = gYellowVec4/256;
    gYellowVec5 =np.array( [149,117,71]); colors['yellowgrey5'] = gYellowVec5/256;
    colors['yellowgrey1.5'] = (gYellowVec1+gYellowVec2)/(512);
    colors['yellowgrey2.5'] = (gYellowVec2+gYellowVec3)/(512);
    colors['yellowgrey3.5'] = (gYellowVec3+gYellowVec4)/(512);
    colors['yellowgrey4.5'] = (gYellowVec4+gYellowVec5)/(512);
    
    #pure grays (white to black)
    gVec1 = np.array([256,256,256]); colors['grey1'] = gVec1/256;
    colors['white'] = colors['grey1']
    gVec2 = np.array([242,242,242]); colors['grey2'] = gVec2/256;
    gVec3 = np.array([230,230,230]); colors['grey3'] = gVec3/256;
    gVec4 = np.array([204,204,204]); colors['grey4'] = gVec4/256;
    gVec5 = np.array([179,179,179]); colors['grey5'] = gVec5/256;
    gVec6 = np.array([153,153,153]); colors['grey6'] = gVec6/256;
    gVec7 = np.array([128,128,128]); colors['grey7'] = gVec7/256;
    gVec8 = np.array([102,102,102]); colors['grey8'] = gVec8/256;
    gVec9 = np.array([77,77,77]); colors['grey9'] = gVec9/256;
    gVec10 = np.array([51,51,51]); colors['grey10'] = gVec10/256;
    gVec11 = np.array([26,26,26]); colors['grey11'] = gVec11/256;
    gVec12 = np.array([0,0,0]); colors['grey12'] = gVec12/256;
    colors['black'] = np.array([0,0,0]);
    
    # alt blue, green, red, yellow
    alt_blue_light = np.array([162,188,200]); colors['alt_blue_light'] = alt_blue_light/256;
    alt_blue_dark = np.array([134,154,175]); colors['alt_blue_dark'] = alt_blue_dark/256;
    alt_green_light = np.array([163,185,169]); colors['alt_green_light'] = alt_green_light/256;
    alt_green_dark = np.array([117,135,133]); colors['alt_green_dark'] = alt_green_dark/256;
    alt_red_light = np.array([175,165,175]); colors['alt_red_light'] = alt_red_light/256;
    alt_red_dark = np.array([145,119,123]); colors['alt_red_dark'] = alt_red_dark/256;
    alt_yellow_light = np.array([175,165,175]); colors['alt_yellow_light'] = alt_yellow_light/256;
    alt_yellow_dark = np.array([145,119,123]); colors['alt_yellow_dark'] = alt_yellow_dark/256;
    
    return colors


    # gist_earth
def colors_gist(x): # x can be scalar or vector, index of color between 0 (white) and 1 (black)
    
    return plt.cm.gist_earth_r(x)

# =============================================================================
# end colors
# =============================================================================


# =============================================================================
# the most important function
# =============================================================================

def index_finder(var_1,var_2):
    
    if type(var_1).__name__ == 'float' or type(var_1).__name__ == 'float64' or type(var_1).__name__ == 'int' or type(var_1).__name__ == 'uint8':
        value = var_1
        array = np.asarray(var_2)
    elif type(var_2).__name__ == 'float' or type(var_2).__name__ == 'float64' or type(var_2).__name__ == 'int' or type(var_2).__name__ == 'uint8':
        value = var_2
        array = np.asarray(var_1)
    else:
        raise ValueError('index_finder: it doesn\'t seem like either input is an integer or float. type(var_1) = {}, type(var_2) = {}'.format(type(var_1).__name__,type(var_2).__name__))
        
    if len(np.shape(array)) == 2:
        _idx_1d = ( np.abs( array[:] - value ) ).argmin()
        _inds = np.unravel_index(_idx_1d,np.shape(array))
    elif len(np.shape(array)) == 1:
        _inds = ( np.abs( array[:] - value ) ).argmin()
    
    return _inds

# =============================================================================
# end the most important function
# =============================================================================
