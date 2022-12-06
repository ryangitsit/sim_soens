import numpy as np
import pickle
import sys
from numpy.random import default_rng

from _util import physical_constants
from _util__soen import get_jj_params, dend_load_arrays_thresholds_saturations
p = physical_constants()

from _functions__soen import run_soen_sim
from _plotting__soen import plot_dendrite, plot_synapse, plot_neuron, plot_neuron_simple, plot_network

class input_signal():
    
    _next_uid = 0
    input_signals = dict()
    
    def __init__(self, **kwargs):
        
        #make new input signal
        self.uid = input_signal._next_uid
        input_signal._next_uid += 1
        self.unique_label = 'in{}'.format(self.uid)
        
        # name the input signal
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'unnamed_input_signal__{}'.format(self.unique_label)
        # end name 
        
        if 'input_temporal_form' in kwargs: 
            if (kwargs['input_temporal_form'] == 'constant' or
                kwargs['input_temporal_form'] == 'constant_rate' or 
                kwargs['input_temporal_form'] == 'arbitrary_spike_train' or
                kwargs['input_temporal_form'] == 'arbitrary_spike_train_with_jitter' or
                kwargs['input_temporal_form'] == 'analog_dendritic_drive'):
                _temporal_form = kwargs['input_temporal_form']
            else:
                raise ValueError('[soen_sim] Tried to assign an invalid input signal temporal form to input %s (unique_label = %s)\nThe allowed values of input_temporal_form are ''single_spike'', ''constant_rate'', ''arbitrary_spike_train'', and ''analog_dendritic_drive''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'single_spike'
        self.input_temporal_form =  _temporal_form #'single_spike' by default
        
        if self.input_temporal_form == 'constant':
            if 'applied_flux' in kwargs:
                self.applied_flux = kwargs['applied_flux']
            else:
                raise ValueError('[soen_sim] If the input temporal form is constant, applied_flux is required as a keyword argument.')
        
        if self.input_temporal_form == 'arbitrary_spike_train' or self.input_temporal_form == 'arbitrary_spike_train_with_jitter':
            if 'spike_times' in kwargs:
                self.spike_times = kwargs['spike_times'] # spike times entered as a list of length >= 1 with entries having units of ns
            else:
                raise ValueError('[soen_sim] arbitrary spike train requires spike_times as input')
                
        # =============================================================================
        #         arbitrary spike train with jitter
        # =============================================================================
        if self.input_temporal_form == 'arbitrary_spike_train_with_jitter': 
            
            if 'source_type' in kwargs:
                self.source_type = kwargs['source_type']
            else:
                self.source_type = 'qd'
                
            if 'num_photons_per_spike' in kwargs:
                self.num_photons_per_spike = kwargs['num_photons_per_spike'] # this many photons generated per spike time
            else:
                self.num_photons_per_spike = 1                
    
            for _str in sys.path:
                if _str[-8:] == 'soen_sim':
                    _path = _str.replace('\\','/')
                    break
            
            if self.source_type == 'qd':
                load_string = 'source_qd_Nph_1.0e+04'
            elif self.source_type == 'ec':
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
            # print('t_vec__el[0] = {:5.3f}ns, t_vec__el[-1] = {:5.3f}ns'.format(t_vec__el[0],t_vec__el[-1]))
            dt_vec = np.diff(t_vec__el)
            el_vec = el_vec[_ind_on:_ind_off]
        
            # form probability distribution
            el_cumulative_vec = np.cumsum(el_vec[:-1]*dt_vec[:])
            el_cumulative_vec = el_cumulative_vec/np.max(el_cumulative_vec)    
            
            # draw samples
            num_samples = self.num_photons_per_spike
            rng = default_rng()
            
            photon_delay_time_vec = np.zeros([len(self.spike_times),num_samples])
            for pp in range(len(self.spike_times)):
                for qq in range(num_samples):
                    random_numbers = rng.random(size = num_samples)
                    photon_delay_time_vec[pp,qq] = t_vec__el[ ( np.abs( el_cumulative_vec[:] - random_numbers[qq] ) ).argmin() ]
                
            # adjust spike times
            self.spike_times__in = self.spike_times
            for ii in range(len(self.spike_times)):
                self.spike_times = self.spike_times + np.min( photon_delay_time_vec[ii,:] )                
        # =============================================================================
        #         end arbitrary spike train with jitter
        # =============================================================================
            
        if self.input_temporal_form == 'constant_rate': # in this case, spike_times has the form [t_first_spike,rate] with rate in MHz
            if 't_first_spike' in kwargs:
                self.t_first_spike = kwargs['t_first_spike']
            else:
                self.t_first_spike = 50 # default time of first spike is 50ns
            if 'rate' in kwargs:
                self.rate = kwargs['rate']
            else:
                self.rate = 1 # default rate is 1MHz
                
        if self.input_temporal_form == 'analog_dendritic_drive':
            if 'piecewise_linear' in kwargs:
                self.piecewise_linear = kwargs['piecewise_linear']    
            
        input_signal.input_signals[self.name] = self

class dendrite():
    
    _next_uid = 0
    dendrites = dict()

    def __init__(self, **kwargs):
        
        # make new dendrite
        self.uid = dendrite._next_uid
        self.unique_label = 'd{}'.format(self.uid)
        dendrite._next_uid += 1
        
        # name the dendrite
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'unnamed_dendrite__{}'.format(self.unique_label)
        # end name 
        
        if 'loops_present' in kwargs:
            if kwargs['loops_present'] == 'ri' or kwargs['loops_present'] == 'pri' or kwargs['loops_present'] == 'rtti' or kwargs['loops_present'] == 'prtti':
                self.loops_present = kwargs['loops_present']
            else:
                raise ValueError('[soen_sim] loops_present must be \'ri\', \'pri\', \'rtti\', \'prtti\'. ')
        else:
            self.loops_present = 'ri'
                    
        if 'circuit_betas' in kwargs:
            if self.loops_present == 'ri':
                if type(kwargs['circuit_betas']) == list and len(kwargs['circuit_betas']) == 3:
                    self.circuit_betas = kwargs['circuit_betas']
                else:
                    raise ValueError('[soen_sim] circuit_betas for an ri dendrite is a list of three real numbers greater than zero with dimensionless units. The first element represents the self inductance of the left branch of the DR loop. The second element represents the right branch of the DR loop. The third element represents the total inductance of the DI loop, including the integrating kinetic inductor and the output inductance.')
            elif self.loops_present == 'rtti':
                if type(kwargs['circuit_betas']) == list and len(kwargs['circuit_betas']) == 3:
                    self.circuit_betas = kwargs['circuit_betas']
                else:
                    raise ValueError('[soen_sim] circuit_betas for an rtti dendrite is a list of three real numbers greater than zero with dimensionless units. The first element represents the self inductance of the left branch of the DR loop. The second element represents the right branch of the DR loop. The third element represents the inductor to the right of the DR loop that goes to the JTL. The fourth element represents the inductor in the JTL. The fifth element represents the total inductance of the DI loop, including the integrating kinetic inductor and the output inductance.')
        else:
            if self.loops_present == 'ri':
                self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, 2*np.pi*1e2] # 2*np.pi* 1/4 to match SQUID handbook defintion with beta_L_sq = 1           
            if self.loops_present == 'rtti':
                self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, 2*np.pi*1e2] # 2*np.pi* 1/4 to match SQUID handbook defintion with beta_L_sq = 1

        if 'junction_critical_current' in kwargs:
            self.junction_critical_current =  kwargs['junction_critical_current']
        else:
            self.junction_critical_current =  100 # default Ic = 100 uA
        self.Ic = self.junction_critical_current
        
        if 'junction_beta_c' in kwargs:
            self.junction_beta_c =  kwargs['junction_beta_c']
        else:
            self.junction_beta_c =  0.3 # default beta_c = 0.3
        self.beta_c = self.junction_beta_c
        
        jj_params = get_jj_params(self.Ic*1e-6,self.beta_c)
            
        if 'bias_current' in kwargs:
            self.bias_current = kwargs['bias_current']
        else:
            if self.loops_present == 'ri':
                self.bias_current = 1.7 # dimensionless bias current        
            elif self.loops_present == 'rtti':
                self.bias_current = 2.0 # dimensionless bias current
        self.ib = self.bias_current
        
        if 'integration_loop_time_constant' in kwargs: # constant with units of ns or list of tau-s pairs; for list of pairs the form is [[tau_1,s_1],[tau_2,s_2],...] where the temporal decay will have time constant tau_1 from s = 0 to s = s_1, will switch to tau_2 from s_1 to s_2, etc 
            self.integration_loop_time_constant = kwargs['integration_loop_time_constant']
        else:
            self.integration_loop_time_constant = 250 # default time constant units of ns
        if type(self.integration_loop_time_constant).__name__ == 'list': 
            tau_vs_current = self.integration_loop_time_constant
            self.tau_list, self.s_list = [], []
            for tau_s in tau_vs_current:
                self.tau_list.append(tau_s[0])
                self.s_list.append(tau_s[1])
            self.tau_list.append(self.tau_list[-1]) # add one more entry with very large s
            self.s_list.append(1e6) # add one more entry with very large s
            self.tau_di = self.tau_list[0] # this is just here to give it an initial value. in time stepper it's broken down by s
        else:
            self.tau_di = self.integration_loop_time_constant  

        if 'normalize_input_connection_strengths' in kwargs:
            if kwargs['normalize_input_connection_strengths'] == True:
                self.normalize_input_connection_strengths = kwargs['normalize_input_connection_strengths']
                if 'total_excitatory_input_connection_strength' in kwargs:
                    self.total_excitatory_input_connection_strength = kwargs['total_excitatory_input_connection_strength']
                else:
                    self.total_excitatory_input_connection_strength = 1                     
                if 'total_inhibitory_input_connection_strength' in kwargs:
                    self.total_inhibitory_input_connection_strength = kwargs['total_inhibitory_input_connection_strength']
                else:
                    self.total_inhibitory_input_connection_strength = -0.5
            else:
                self.normalize_input_connection_strengths = False
        else:
            self.normalize_input_connection_strengths = False
                
        if 'offset_flux' in kwargs: # units of Phi0
            self.offset_flux = kwargs['offset_flux']
        else:
            self.offset_flux = 0
            
        if 'self_feedback_coupling_strength' in kwargs: # J_ii, units of phi/s (normalized flux divided by normalized current in DI loop)
            self.self_feedback_coupling_strength = kwargs['self_feedback_coupling_strength']
        else:
            self.self_feedback_coupling_strength = 0
            
        tau_di = self.tau_di * 1e-9
        beta_di = self.circuit_betas[-1]
        Ic = self.Ic * 1e-6
        Ldi = p['Phi0']*beta_di/(2*np.pi*Ic)
        rdi = Ldi/tau_di
        self.alpha = rdi/jj_params['r_j']
        if hasattr(self,'tau_list'):
            rdi_list = Ldi/(np.asarray(self.tau_list) * 1e-9)
            self.alpha_list = rdi_list/jj_params['r_j']
        self.jj_params = jj_params

        # prepare dendrite to have connections
        self.external_inputs = dict()
        self.external_connection_strengths = dict()
        self.synaptic_inputs = dict()
        self.synaptic_connection_strengths = dict()
        self.dendritic_inputs = dict()
        self.dendritic_connection_strengths = dict()
                    
        dendrite.dendrites[self.name] = self
            
        return 
    
    def add_input(self, connection_object, connection_strength = 1):
        
        if type(connection_object).__name__ == 'input_signal':
            self.external_inputs[connection_object.name] = input_signal.input_signals[connection_object.name]
            self.external_connection_strengths[connection_object.name] = connection_strength

        if type(connection_object).__name__ == 'synapse':
            self.synaptic_inputs[connection_object.name] = synapse.synapses[connection_object.name]
            self.synaptic_connection_strengths[connection_object.name] = connection_strength
            
        if type(connection_object).__name__ == 'dendrite':
            self.dendritic_inputs[connection_object.name] = dendrite.dendrites[connection_object.name]            
            self.dendritic_connection_strengths[connection_object.name] = connection_strength
            
        if type(connection_object).__name__ == 'neuron':
            self.dendritic_inputs[connection_object.name] = neuron.neurons[connection_object.name]            
            self.dendritic_connection_strengths[connection_object.name] = connection_strength
        
        return self
    
    def run_sim(self, **kwargs):
        self = run_soen_sim(self, **kwargs)
        return self
    
    def plot(self):
        plot_dendrite(self)
        return

    def __del__(self):
        # print('dendrite deleted')
        return
    

class synapse():    

    _next_uid = 0
    synapses = dict()
    
    def __init__(self, **kwargs):

        # make new synapse
        # self._instances.add(weakref.ref(self))
        self.uid = synapse._next_uid
        synapse._next_uid += 1
        self.unique_label = 's{}'.format(self.uid)

        # name the synapse
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'unnamed_synapse__{}'.format(self.unique_label)
        # end name  
        
        # synaptic receiver specification
        if 'tau_rise' in kwargs:
            self.tau_rise = kwargs['tau_rise'] # units of ns
        else:
            self.tau_rise = 0.02 # 20ps is default rise time (L_tot/r_ph = 100nH/5kOhm)
        # print('tau_rise = {}'.format(self.tau_rise))
        
        if 'tau_fall' in kwargs:
            self.tau_fall = kwargs['tau_fall'] # units of ns
        else:
            self.tau_fall = 50 # 50ns is default fall time for SPD recovery
        # print('tau_fall = {}'.format(self.tau_fall))
            
        if 'hotspot_duration' in kwargs: # specified in units of number of tau_rise time constants
            self.hotspot_duration = kwargs['hotspot_duration'] * self.tau_rise # units of ns
        else:
            self.hotspot_duration =  2 * self.tau_rise # two time constants is default
        # print('hotspot_duration = {}'.format(self.hotspot_duration))
            
        if 'spd_duration' in kwargs: # how long to observe spd after input spike, specified in units of number of tau_fall time constants
            self.spd_duration = kwargs['spd_duration'] * self.tau_fall # units of ns
        else:
            self.spd_duration = 8 * self.tau_fall # eight time constants is default
        
        if 'phi_peak' in kwargs:
            self.phi_peak = kwargs['phi_peak'] # units of Phi0
        else:
            self.phi_peak = 0.5 # default peak flux is Phi0/2
            
        if 'spd_reset_time' in kwargs: # this duration must elapse before the spd can detect another photon
            self.spd_reset_time = kwargs['spd_reset_time']            
        else:
            self.spd_reset_time = self.tau_fall
        # end synaptic receiver spd circuit specification
        
        synapse.synapses[self.name] = self
        
        return
    
    def add_input(self, input_object):
        
        self.synaptic_input = input_object.name
        self.input_signal = input_object
        
        return self
    
    def run_sim(self, **kwargs):
        self = run_soen_sim(self, **kwargs)
        return self
    
    def plot(self):
        plot_synapse(self)
        return

    def __del__(self):
        # print('synapse deleted')
        return
                
    
class neuron():    

    _next_uid = 0
    neurons = dict()
    
    def __init__(self, **kwargs):

        # make new neuron
        self.uid = neuron._next_uid
        neuron._next_uid += 1
        self.unique_label = 'n{}'.format(self.uid)

        # name the neuron
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'unnamed_neuron__{}'.format(self.unique_label)
        # end name 
        
        # =============================================================================
        #         receiving and integration dendrite
        # =============================================================================
        
        if 'loops_present' in kwargs:
            if kwargs['loops_present'] == 'ri' or kwargs['loops_present'] == 'pri' or kwargs['loops_present'] == 'rtti' or kwargs['loops_present'] == 'prtti':
                self.loops_present = kwargs['loops_present']
            else:
                raise ValueError('[soen_sim] loops_present must be \'ri\', \'pri\', \'rtti\', \'prtti\'. ')
        else:
            self.loops_present = 'ri'
                    
        if 'circuit_betas' in kwargs:
            if self.loops_present == 'ri':
                if type(kwargs['circuit_betas']) == list and len(kwargs['circuit_betas']) == 3:
                    self.circuit_betas = kwargs['circuit_betas']
                else:
                    raise ValueError('[soen_sim] circuit_betas for an ri dendrite is a list of three real numbers greater than zero with dimensionless units. The first element represents the self inductance of the left branch of the DR loop. The second element represents the right branch of the DR loop. The third element represents the total inductance of the DI loop, including the integrating kinetic inductor and the output inductance.')
            elif self.loops_present == 'rtti':
                if type(kwargs['circuit_betas']) == list and len(kwargs['circuit_betas']) == 3:
                    self.circuit_betas = kwargs['circuit_betas']
                else:
                    raise ValueError('[soen_sim] circuit_betas for an ri dendrite is a list of three real numbers greater than zero with dimensionless units. The first element represents the self inductance of the left branch of the DR loop. The second element represents the right branch of the DR loop. The third element represents the inductor to the right of the DR loop that goes to the JTL. The fourth element represents the inductor in the JTL. The fifth element represents the total inductance of the DI loop, including the integrating kinetic inductor and the output inductance.')
        else:
            if self.loops_present == 'ri':
                self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, 2*np.pi*1e2] # 2*np.pi* 1/4 to match SQUID handbook defintion with beta_L_sq = 1           
            if self.loops_present == 'rtti':
                self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, 2*np.pi*1e2] # 2*np.pi* 1/4 to match SQUID handbook defintion with beta_L_sq = 1

        if 'junction_critical_current' in kwargs:
            self.junction_critical_current =  kwargs['junction_critical_current'] # units of microamps
        else:
            self.junction_critical_current =  100 # default Ic = 100 uA
        self.Ic = self.junction_critical_current
        
        if 'junction_beta_c' in kwargs:
            self.junction_beta_c =  kwargs['junction_beta_c']
        else:
            self.junction_beta_c =  0.3 # default beta_c = 0.3
        self.beta_c = self.junction_beta_c
        
        jj_params = get_jj_params(self.Ic*1e-6,self.beta_c)
            
        if 'bias_current' in kwargs:
            self.bias_current = kwargs['bias_current']
        else:
            if self.loops_present == 'ri':
                self.bias_current = 1.7 # dimensionless bias current        
            elif self.loops_present == 'rtti':
                self.bias_current = 2.0 # dimensionless bias current
        
        if 'integration_loop_time_constant' in kwargs: # units of ns
            self.integration_loop_time_constant = kwargs['integration_loop_time_constant']
            self.tau_ni = self.integration_loop_time_constant            
        else:
            self.integration_loop_time_constant = 250 # default time constant units of ns
            
        if 'absolute_refractory_period' in kwargs: # units of ns
            self.absolute_refractory_period = kwargs['absolute_refractory_period']
        else:
            self.absolute_refractory_period = 10
            
        if 'normalize_input_connection_strengths' in kwargs:
            self.normalize_input_connection_strengths = kwargs['normalize_input_connection_strengths']
        else:
            self.normalize_input_connection_strengths = False
            
        if 'total_excitatory_input_connection_strength' in kwargs:
            self.total_excitatory_input_connection_strength = kwargs['total_excitatory_input_connection_strength']
        else:
            self.total_excitatory_input_connection_strength = 1 
            
        if 'total_inhibitory_input_connection_strength' in kwargs:
            self.total_inhibitory_input_connection_strength = kwargs['total_inhibitory_input_connection_strength']
        else:
            self.total_inhibitory_input_connection_strength = -0.5
                
        if 'offset_flux' in kwargs: # units of Phi0
            self.offset_flux = kwargs['offset_flux']
        else:
            self.offset_flux = 0                 
            
        if 'self_feedback_coupling_strength' in kwargs: # J_ii, units of phi/s (normalized flux divided by normalized current in DI loop)
            self.self_feedback_coupling_strength = kwargs['self_feedback_coupling_strength']
        else:
            self.self_feedback_coupling_strength = 0
            
        tau_ni = self.tau_ni * 1e-9
        beta_ni = self.circuit_betas[-1]
        Ic = self.Ic * 1e-6
        Lni = p['Phi0']*beta_ni/(2*np.pi*Ic)
        rni = Lni/tau_ni
        self.alpha = rni/jj_params['r_j']
        self.jj_params = jj_params
        
        # create dendrite for neuronal receiving and integration loop
        neuron_dendrite = dendrite(name = '{}__{}'.format(self.name,'nr_ni'), loops_present = self.loops_present, 
                      circuit_betas = self.circuit_betas, junction_critical_current = self.junction_critical_current, junction_beta_c = self.junction_beta_c,
                      bias_current = self.bias_current, 
                      integration_loop_time_constant = self.integration_loop_time_constant, 
                      normalize_input_connection_strengths = self.normalize_input_connection_strengths, 
                      total_excitatory_input_connection_strength = self.total_excitatory_input_connection_strength, 
                      total_inhibitory_input_connection_strength = self.total_inhibitory_input_connection_strength, 
                      offset_flux = self.offset_flux,
                      self_feedback_coupling_strength = self.self_feedback_coupling_strength)
        neuron_dendrite.is_soma = True    
        
        self.dend__nr_ni = neuron_dendrite
        self.dend__nr_ni.absolute_refractory_period = self.absolute_refractory_period
        
        # =============================================================================
        #         end receiving and integration dendrite
        # =============================================================================
        
        # =============================================================================
        #         refractory dendrite
        # =============================================================================
        
        if 'loops_present__refraction' in kwargs:
            if kwargs['loops_present__refraction'] == 'ri' or kwargs['loops_present__refraction'] == 'pri' or kwargs['loops_present__refraction'] == 'rtti' or kwargs['loops_present__refraction'] == 'prtti':
                self.loops_present__refraction = kwargs['loops_present__refraction']
            else:
                raise ValueError('[soen_sim] loops_present must be \'ri\', \'pri\', \'rtti\', \'prtti\'. ')
        else:
            self.loops_present__refraction = 'ri'
                    
        if 'circuit_betas__refraction' in kwargs:
            if self.loops_present__refraction == 'ri':
                if type(kwargs['circuit_betas__refraction']) == list and len(kwargs['circuit_betas__refraction']) == 3:
                    self.circuit_betas__refraction = kwargs['circuit_betas__refraction']
                else:
                    raise ValueError('[soen_sim] circuit_betas for an ri dendrite is a list of three real numbers greater than zero with dimensionless units. The first element represents the self inductance of the left branch of the DR loop. The second element represents the right branch of the DR loop. The third element represents the total inductance of the DI loop, including the integrating kinetic inductor and the output inductance.')
            elif self.loops_present__refraction == 'rtti':
                if type(kwargs['circuit_betas__refraction']) == list and len(kwargs['circuit_betas__refraction']) == 3:
                    self.circuit_betas__refraction = kwargs['circuit_betas__refraction']
                else:
                    raise ValueError('[soen_sim] circuit_betas for an rtti dendrite is a list of three real numbers greater than zero with dimensionless units. The first element represents the self inductance of the left branch of the DR loop. The second element represents the right branch of the DR loop. The third element represents the total inductance of the DI loop, including the integrating kinetic inductor and the output inductance.')
        else:
            if self.loops_present__refraction == 'ri':
                self.circuit_betas__refraction = [2*np.pi* 1/4, 2*np.pi* 1/4, 2*np.pi*1e2] # 2*np.pi* 1/4 to match SQUID handbook defintion with beta_L_sq = 1           
            if self.loops_present__refraction == 'rtti':
                self.circuit_betas__refraction = [2*np.pi* 1/4, 2*np.pi* 1/4, 2*np.pi*0.5, 2*np.pi*0.5, 2*np.pi*1e2] # 2*np.pi* 1/4 to match SQUID handbook defintion with beta_L_sq = 1

        if 'junction_critical_current__refraction' in kwargs:
            self.junction_critical_current__refraction =  kwargs['junction_critical_current__refraction'] # units of microamps
        else:
            self.junction_critical_current__refraction =  100 # default Ic = 100 uA
        self.Ic__refraction = self.junction_critical_current__refraction
        
        if 'junction_beta_c__refraction' in kwargs:
            self.junction_beta_c__refraction =  kwargs['junction_beta_c__refraction']
        else:
            self.junction_beta_c__refraction =  0.3 # default beta_c = 0.3
        self.beta_c__refraction = self.junction_beta_c__refraction
        
        jj_params__refraction = get_jj_params(self.Ic__refraction*1e-6,self.beta_c__refraction)
            
        if 'bias_current__refraction' in kwargs:
            self.bias_current__refraction = kwargs['bias_current__refraction']
        else:
            if self.loops_present__refraction == 'ri':
                self.bias_current__refraction = 1.7 # dimensionless bias current        
            elif self.loops_present__refraction == 'rtti':
                self.bias_current__refraction = 3.1 # dimensionless bias current
        
        if 'integration_loop_time_constant__refraction' in kwargs: # units of ns
            self.integration_loop_time_constant__refraction = kwargs['integration_loop_time_constant__refraction']
        else:
            self.integration_loop_time_constant__refraction = 50 # default time constant units of ns
        self.tau_ref = self.integration_loop_time_constant__refraction    
            
        tau_ref = self.tau_ref * 1e-9
        beta_nr = self.circuit_betas__refraction[-1]
        Ic = self.Ic__refraction * 1e-6
        Lnr = p['Phi0']*beta_nr/(2*np.pi*Ic)
        r_ref = Lnr/tau_ref
        self.alpha__refraction = r_ref/jj_params['r_j']
        self.jj_params__refraction = jj_params__refraction
        
        # create dendrite for neuronal refraction
        neuron_refractory_dendrite = dendrite(name = '{}__dend_{}'.format(self.name,'refraction'), loops_present = self.loops_present__refraction, circuit_betas = self.circuit_betas__refraction, 
                                              junction_critical_current = self.junction_critical_current__refraction, junction_beta_c = self.junction_beta_c__refraction,
                                              bias_current = self.bias_current__refraction, integration_loop_time_constant = self.integration_loop_time_constant__refraction)

        self.dend__ref = neuron_refractory_dendrite
        
        if 'refractory_dendrite_connection_strength' in kwargs:
            if type(kwargs['refractory_dendrite_connection_strength']).__name__ == 'float' or type(kwargs['refractory_dendrite_connection_strength']).__name__ == 'int':
                self.refractory_dendrite_connection_strength = kwargs['refractory_dendrite_connection_strength']
            elif kwargs['refractory_dendrite_connection_strength'] == 'auto':
                ib__list__ri, phi_r__array__ri, i_di__array__ri, r_fq__array__ri, phi_th_plus__vec__ri, phi_th_minus__vec__ri, s_max_plus__vec__ri, s_max_minus__vec__ri, s_max_plus__array__ri, s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')
                ib__list__rtti, phi_r__array__rtti, i_di__array__rtti, r_fq__array__rtti, phi_th_plus__vec__rtti, phi_th_minus__vec__rtti, s_max_plus__vec__rtti, s_max_minus__vec__rtti, s_max_plus__array__rtti, s_max_minus__array__rtti = dend_load_arrays_thresholds_saturations('default_rtti')
                if self.loops_present == 'ri':
                    ib_list = ib__list__ri
                    phi_th_minus_vec = phi_th_minus__vec__ri
                    phi_th_plus_vec = phi_th_plus__vec__ri
                elif self.loops_present == 'rtti':
                    ib_list = ib__list__rtti
                    phi_th_minus_vec = phi_th_minus__vec__rtti
                    phi_th_plus_vec = phi_th_plus__vec__rtti
                if self.loops_present__refraction == 'ri':
                    ib_list_r = ib__list__ri
                    s_max_plus_vec = s_max_plus__vec__ri
                elif self.loops_present__refraction == 'rtti':
                    ib_list_r = ib__list__rtti
                    s_max_plus_vec = s_max_plus__vec__rtti
                _ind_ib = ( np.abs( ib_list[:] - self.dend__nr_ni.ib ) ).argmin()
                phi_th_plus = phi_th_plus_vec[_ind_ib]
                phi_th_minus = phi_th_minus_vec[_ind_ib]
                delta = phi_th_plus - phi_th_minus
                _ind_ib_r = ( np.abs( ib_list_r[:] - self.dend__ref.ib ) ).argmin()
                s_max = s_max_plus_vec[_ind_ib_r]
                self.refractory_dendrite_connection_strength = -delta/s_max # ( phi_th_minus + delta/100 ) / s_max
        else:
            self.refractory_dendrite_connection_strength = -0.7 # default; when in doubt, use 'auto'
            
        self.dend__nr_ni.add_input(self.dend__ref, connection_strength = self.refractory_dendrite_connection_strength)
        
        # =============================================================================
        #         end refractory dendrite
        # =============================================================================
        
        # =============================================================================
        #         synapse to refractory dendrite
        # =============================================================================
        
        if 'tau_rise__refraction' in kwargs:
            tau_rise__refraction = kwargs['tau_rise__refraction'] # units of ns
        else:
            tau_rise__refraction = 0.02 # 20ps is default rise time (L_tot/r_ph = 100nH/5kOhm)
        
        if 'tau_fall__refraction' in kwargs:
            tau_fall__refraction = kwargs['tau_fall__refraction'] # units of ns
        else:
            tau_fall__refraction = 50 # 50ns is default fall time for SPD recovery
            
        if 'hotspot_duration__refraction' in kwargs: # specified in units of number of tau_rise time constants
            hotspot_duration__refraction = kwargs['hotspot_duration__refraction']
        else:
            hotspot_duration__refraction =  2
            
        if 'spd_duration__refraction' in kwargs: # specified in units of number of tau_fall time constants
            spd_duration__refraction = kwargs['spd_duration__refraction']
        else:
            spd_duration__refraction = 8 # eight time constants is default
        
        if 'phi_peak__refraction' in kwargs:
            phi_peak__refraction = kwargs['phi_peak__refraction'] # units of Phi0
        else:
            phi_peak__refraction = 0.5 # default peak flux is Phi0/2
            
        # create synapse for neuronal refraction
        synapse__ref = synapse(name = '{}__syn_{}'.format(self.name,'refraction'), tau_rise = tau_rise__refraction, tau_fall = tau_fall__refraction, hotspot_duration = hotspot_duration__refraction, spd_duration = spd_duration__refraction, phi_peak = phi_peak__refraction)
        
        # add neuronal output as synaptic input
        synapse__ref.add_input(self)
        
        # add synaptic output as input to refractory dendrite
        self.dend__ref.add_input(synapse__ref, connection_strength = 1)
        
        # =============================================================================
        #         end synapse to refractory dendrite
        # =============================================================================
        
        # =============================================================================
        #         transmitter
        # =============================================================================
        
        if 'integrated_current_threshold' in kwargs:
            self.integrated_current_threshold = kwargs['integrated_current_threshold']
        else:
            self.integrated_current_threshold = 0.5 # units of Ic
            
        if 'source_type' in kwargs: 
            if kwargs['source_type'] == 'ec' or kwargs['source_type'] == 'qd' or kwargs['source_type'] == 'delay_delta':
                self.source_type = kwargs['source_type']
            else:
                raise ValueError('[soen_sim] sources presently defined are ''qd'', ''ec'', or ''delay_delta''')
        else:
            self.source_type = 'qd' # 'qd', 'ec', or 'delay_delta'
            
        if self.source_type == 'delay_delta':
            if 'light_production_delay' in kwargs: # only necessary if source_type = delta_delay
                self.light_production_delay = kwargs['light_production_delay']
            else:
                self.light_production_delay = 2 # ns
            
        if self.source_type == 'qd' or self.source_type == 'ec':
            if 'num_photons_out_factor' in kwargs:
                self.num_photons_out_factor = kwargs['num_photons_out_factor']
            else:
                self.num_photons_out_factor = 10 # produce num_photons = num_photons_out_factor * len(synaptic_outputs) each time the neuron fires
            
        # =============================================================================
        #         end transmitter
        # =============================================================================
        
        # prepare for spikes        
        self.spike_times = []
        self.spike_indices = []
        self.dend__nr_ni.spike_times = []
        
        # prepare for output synapses
        self.synaptic_outputs = dict()

        neuron.neurons[self.name] = self
        
        return    
        
    def add_input(self, connection_object, connection_strength = 1):
        self.dend__nr_ni.add_input(connection_object, connection_strength)
        return
        
    def add_output(self, connection_object):
        
        if type(connection_object).__name__ == 'synapse':
            self.synaptic_outputs[connection_object.name] = synapse.synapses[connection_object.name]
            synapse.synapses[connection_object.name].add_input(self)
        else: 
            raise ValueError('[soen_sim] an output from a neuron must be a synapse')
                
        return
    
    def run_sim(self, **kwargs):
        self = run_soen_sim(self, **kwargs)
        return self
    
    def plot(self):
        if self.plot_simple:
            plot_neuron_simple(self)
        else:
            plot_neuron(self)
        return

    def __del__(self):
        # print('dendrite deleted')
        return


class network():
    
    _next_uid = 0
    network = dict()
    # from scipy.sparse import csr_matrix
    
    def __init__(self, **kwargs):

        # make network
        self.uid = network._next_uid
        network._next_uid += 1
        self.unique_label = 'net{}'.format(self.uid)

        # name the network
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'unnamed_network__{}'.format(self.unique_label)
        # end name 
        
        # JJ params
        dummy_dendrite = dendrite() # make dummy dendrite to obtain default Ic, beta_c
        jj_params = get_jj_params(dummy_dendrite.Ic*1e-6,dummy_dendrite.beta_c)
        self.jj_params = jj_params # add jj_params to network just as stupid hack to construct t_tau_conversion (this should be done somewhere else globally)
        
        # prepare network to have neurons
        self.neurons = dict()
 
    def add_neuron(self, neuron_object):
        self.neurons[neuron_object.name] = neuron_object
        return
    
    def run_sim(self, **kwargs):
        self.dt = kwargs['dt']
        self = run_soen_sim(self, **kwargs)
        return self
    
    def plot(self):
        plot_network(self)
        return

    def get_recordings(self):
        self.t = self.neurons[list(self.neurons.keys())[0]].time_params['time_vec']
        spikes = [ [] for _ in range(2) ]
        print(spikes)
        S = []
        Phi_r = []
        spike_signals = []
        count = 0
        for neuron_key in self.neurons:
            neuron = self.neurons[neuron_key]
            s = neuron.dend__nr_ni.s
            S.append(s)
            phi_r = neuron.dend__nr_ni.phi_r
            Phi_r.append(phi_r)
            spike_t = neuron.spike_times
            spikes[0].append(np.ones(len(spike_t))*count)
            spikes[1].append((spike_t/neuron.time_params['t_tau_conversion']))
            spike_signal = []
            spike_times = spike_t/neuron.time_params['t_tau_conversion']
            for spike in spike_times:
                spike_signal.append(s[int(spike/self.dt)])
                spike_signals.append(spike_signal)
            count+=1
        spikes[0] = np.concatenate(spikes[0])
        spikes[1] = np.concatenate(spikes[1])
        self.spikes = spikes
        self.spike_signals = spike_signals
        self.phi_r = Phi_r
        self.signal = S
        neuron.spike_times = []