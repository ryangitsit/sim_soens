import numpy as np
import pickle
import sys
from numpy.random import default_rng

from soen_utilities import get_jj_params, dend_load_arrays_thresholds_saturations, physical_constants, index_finder
p = physical_constants()

from soen_functions import run_soen_sim
from soen_plotting import plot_dendrite, plot_synapse, plot_neuron, plot_neuron_simple, plot_network

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

    def __init__(self, **params):
        
        # DEFAULT SETTINGS
        self.uid = dendrite._next_uid
        self.unique_label = 'd{}'.format(self.uid)
        dendrite._next_uid += 1
        self.name = 'unnamed_dendrite__{}'.format(self.unique_label)
        if 'loops_present' in params:
            self.loops_present = params['loops_present']
        else:
            self.loops_present = 'ri'
        if self.loops_present == 'ri':
            self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, 2*np.pi*1e2]         
        if self.loops_present == 'rtti':
            self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, 2*np.pi*1e2]
        self.Ic =  100
        self.beta_c =  0.3
        if self.loops_present == 'ri':
            self.ib = 1.7        
        elif self.loops_present == 'rtti':
            self.ib = 2.0 
        self.tau_di= 250
        self.normalize_input_connection_strengths = False
        self.total_excitatory_input_connection_strength = 1
        self.total_inhibitory_input_connection_strength = -0.5
        self.offset_flux = 0
        self.self_feedback_coupling_strength = 0


        # UPDATE TO CUSTOM PARAMS
        self.__dict__.update(params)
        # print(self.type,self.dentype)
        # self.loops_present = self.type
        if hasattr(self, 'dentype'):
            if self.dentype == 'refractory':
                # print("REFRACTORY DENDRITE")
                self.loops_present = self.loops_present__refraction
                self.circuit_betas = self.circuit_betas__refraction 
                self.Ic = self.Ic__refraction
                self.beta_c = self.beta_c__refraction
                self.ib = self.ib_ref
                self.tau_di = self.tau_ref   
                self.name = '{}__dend_{}'.format(self.name,'refraction')  
            elif self.dentype == 'soma':
                self.tau_di = self.tau_ni
                self.ib = self.ib_n
                self.name = '{}__{}'.format(self.name,'nr_ni')
                # print("SOMATIC DENDRITE")
        else:
            self.ib = self.ib
            # print("REGULAR DENDRITE")

        if 'integrated_current_threshold' in params:
            self.s_th = params['integrated_current_threshold']
        
        params = self.__dict__
        self.bias_current = self.ib

        # for k,v in params.items():
        #     print(k," -> ",v)
        # print(" ************************************************************* ")
        # futher adjustments to parameters

        if (self.loops_present != 'ri' 
            and self.loops_present != 'pri' 
            and self.loops_present != 'pri' 
            and self.loops_present != 'rtti' 
            and self.loops_present != 'prtti'):
            raise ValueError('''
            [soen_sim] loops_present must be:
                \'ri\', \'pri\', \'rtti\', \'prtti\'. ''')

        if type(self.circuit_betas) == list and (len(self.circuit_betas) == 3 
                                            or len(self.circuit_betas) == 5):
            if self.loops_present == 'ri':
                self.circuit_betas = self.circuit_betas
            elif self.loops_present == 'rtti':
                self.circuit_betas = self.circuit_betas
        else:
            raise ValueError('''
            [soen_sim] circuit_betas for an ri dendrite is a list of three real 
            numbers greater than zero with dimensionless units. The first 
            element represents the self inductance of the left branch of the DR 
            loop. The second element represents the right branch of the DR loop. 
            The third element represents the total inductance of the DI loop, 
            including the integrating kinetic inductor and the output 
            inductance.''')

        jj_params = get_jj_params(self.Ic*1e-6,self.beta_c)

        # if not hasattr(self, 'ib'):
        #     print('ib switch')
        #     self.ib = self.ib

        if type(self.tau_di).__name__ == 'list': 
            tau_vs_current = self.tau_di
            self.tau_list, self.s_list = [], []
            for tau_s in tau_vs_current:
                self.tau_list.append(tau_s[0])
                self.s_list.append(tau_s[1])
            self.tau_list.append(self.tau_list[-1]) # add one more entry with very large s
            self.s_list.append(1e6) # add one more entry with very large s
            self.tau_di = self.tau_list[0] # this is just here to give it an initial value. in time stepper it's broken down by s 

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

        # print(self.dentype)
        # print(" ************************************************************* ")

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
    
    def __init__(self, **params):

        # DEFAULT SETTINGS
        self.uid = neuron._next_uid
        neuron._next_uid += 1
        self.unique_label = 'n{}'.format(self.uid)
        self.name = 'unnamed_neuron__{}'.format(self.unique_label)

        # receiving and integration dendrite
        if 'loops_present' in params:
            self.loops_present = params['loops_present']
        else:
            self.loops_present = 'ri'
        self.beta_ni = 2*np.pi*1e2
        if self.loops_present == 'ri':
            self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, self.beta_ni]         
        if self.loops_present == 'rtti':
            self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, self.beta_ni]  
        self.Ic =  100
        self.beta_c =  0.3
        if self.loops_present == 'ri':
            self.ib = 1.7 # dimensionless bias current   
            self.ib_n = 1.802395858835221
        elif self.loops_present == 'rtti':
            self.ib = 2.0 # dimensionless bias current
            self.ib_n = 2.19
        self.integration_loop_time_constant = 250
        self.absolute_refractory_period = 10
        self.normalize_input_connection_strengths = False
        self.total_excitatory_input_connection_strength = 1
        self.total_inhibitory_input_connection_strength = -0.5
        self.offset_flux = 0 
        # J_ii, units of phi/s (normalized flux / normalized current in DI loop)
        self.self_feedback_coupling_strength = 0
        self.s_th = 0.5 # units of Ic

        self.tau_ni = 50
        self.beta_di = 2*np.pi*1e2
        self.tau_di = 500
        self.s_th = 0.5
        self.integrated_current_threshold = self.s_th

        # refractory dendrite
        if 'loops_present__refraction' in params:
            self.loops_present__refraction = params['loops_present__refraction']
        else:
            self.loops_present__refraction = 'ri'
        self.beta_ref = 2*np.pi*1e2
        if self.loops_present__refraction == 'ri':
            self.circuit_betas__refraction = [2*np.pi* 1/4, 2*np.pi* 1/4, self.beta_ref]         
        if self.loops_present__refraction  == 'rtti':
            self.circuit_betas__refraction = [2*np.pi* 1/4, 2*np.pi* 1/4, 2*np.pi*0.5, 
                                  2*np.pi*0.5, self.beta_ref]
        self.Ic__refraction =  100
        self.beta_c__refraction =  0.3
        if self.loops_present__refraction == 'ri':
            self.ib_ref = 1.7    
        elif self.loops_present__refraction == 'rtti':
            self.ib_ref = 3.1 
        self.tau_ref= 50
        self.refractory_dendrite_connection_strength = 'auto'
        auto = True
        self.second_ref=False

        ### synapse to receiving dendrite ###
        self.tau_rise__refraction = 0.02
        self.tau_fall__refraction = 50
        self.hotspot_duration__refraction =  2
        self.spd_duration__refraction = 8
        self.phi_peak__refraction = 0.5

        ### transmitter ###
        self.source_type = 'qd'
        self.num_photons_out_factor = 10


        # UPDATE TO CUSTOM PARAMS
        self.__dict__.update(params)
        
        self.integrated_current_threshold = self.s_th
        params = self.__dict__

        # for k,v in params.items():
        #     print(k," -> ",v)
        # print(" ============================================================= ")

        # futher adjustments to parameters

        ### receiving and integration dendrite ###
        # loops
        if (self.loops_present != 'ri' 
            and self.loops_present != 'pri' 
            and self.loops_present != 'pri' 
            and self.loops_present != 'rtti' 
            and self.loops_present != 'prtti'):
            raise ValueError('''
            [soen_sim] loops_present must be:
                \'ri\', \'pri\', \'rtti\', \'prtti\'. ''')

        if (self.loops_present__refraction != 'ri' 
            and self.loops_present__refraction != 'pri' 
            and self.loops_present__refraction != 'pri' 
            and self.loops_present__refraction != 'rtti' 
            and self.loops_present__refraction != 'prtti'):
            raise ValueError('''
            [soen_sim] loops_present_refraction must be:
                \'ri\', \'pri\', \'rtti\', \'prtti\'. ''')

        self.circuit_betas[-1] = self.beta_ni
        # circuit_betas
        if type(self.circuit_betas) == list and len(self.circuit_betas) == 3:
            if self.loops_present == 'ri':
                self.circuit_betas = self.circuit_betas
            elif self.loops_present == 'rtti':
                self.circuit_betas = self.circuit_betas
        else:
            raise ValueError('''
            [soen_sim] circuit_betas for an ri dendrite is a list of three real 
            numbers greater than zero with dimensionless units. The first 
            element represents the self inductance of the left branch of the DR 
            loop. The second element represents the right branch of the DR loop. 
            The third element represents the total inductance of the DI loop, 
            including the integrating kinetic inductor and the output 
            inductance.''')
        
        # misc
        self.Ic = self.Ic
        self.beta_c = self.beta_c
        jj_params = get_jj_params(self.Ic*1e-6,self.beta_c)

        tau_ni = self.tau_ni * 1e-9
        beta_ni = self.circuit_betas[-1]
        Ic = self.Ic * 1e-6
        Lni = p['Phi0']*beta_ni/(2*np.pi*Ic)
        rni = Lni/tau_ni
        self.alpha = rni/jj_params['r_j']
        self.jj_params = jj_params


        ### refractory dendrite ###
        self.circuit_betas__refraction[-1] = self.beta_ref
        if (type(self.circuit_betas__refraction) == list 
            and (len(self.circuit_betas__refraction) == 3 
            or len(self.circuit_betas__refraction) == 5)):

            if self.loops_present__refraction == 'ri':
                self.circuit_betas__refraction = self.circuit_betas__refraction
            elif self.loops_present == 'rtti':
                self.circuit_betas__refraction = self.circuit_betas__refraction
        else:
            raise ValueError('''
            [soen_sim] circuit_betas for an ri dendrite is a list of three real 
            numbers greater than zero with dimensionless units. The first 
            element represents the self inductance of the left branch of the DR 
            loop. The second element represents the right branch of the DR loop. 
            The third element represents the total inductance of the DI loop, 
            including the integrating kinetic inductor and the output 
            inductance.''')

        jj_params__refraction = get_jj_params(self.Ic__refraction*1e-6,
                                              self.beta_c__refraction)

        if hasattr(self, 'tau_ref'):
            self.tau_ref = self.tau_ref
        else:
            self.tau_ref = self.tau_ref 
        
        tau_ref = self.tau_ref * 1e-9
        beta_nr = self.circuit_betas__refraction[-1]
        Ic = self.Ic__refraction * 1e-6
        Lnr = p['Phi0']*beta_nr/(2*np.pi*Ic)
        r_ref = Lnr/tau_ref
        self.alpha__refraction = r_ref/jj_params['r_j']
        self.jj_params__refraction = jj_params__refraction

        if type(self.refractory_dendrite_connection_strength).__name__ != 'str':
            auto = False
        elif self.refractory_dendrite_connection_strength == 'auto':
            auto = True
        elif self.refractory_dendrite_connection_strength == 'match_excitatory':
            self.refractory_dendrite_connection_strength = self.total_excitatory_input_connection_strength
            auto = False


        ### synapse to receiving dendrite ###
        #none

        ### transmitter ###
        # print(self.source_type)
        if (self.source_type != 'ec' 
            and self.source_type != 'qd' 
            and self.source_type != 'delay_delta'):
            raise ValueError('''[soen_sim] sources presently defined are: 
                                    'qd', 'ec', or 'delay_delta' ''')

        if self.source_type == 'delay_delta':
            if hasattr(self, 'light_production_delay'):
                self.light_production_delay = self.light_production_delay
            else:
                self.light_production_delay = 2
        else:
            self.num_photons_out_factor = self.num_photons_out_factor

        # print(self.num_photons_out_factor)
        # print(" ============================================================= ")
                                
        # ======================================================================
        #         receiving and integration dendrite
        # ======================================================================
                
        # create dendrite for neuronal receiving and integration loop
        # neuron_dendrite = dendrite(name = '{}__{}'.format(self.name,'nr_ni'), loops_present = self.loops_present, 
        #               circuit_betas = self.circuit_betas, junction_critical_current = self.Ic, junction_beta_c = self.beta_c,
        #               bias_current = self.bias_current, 
        #               integration_loop_time_constant = self.integration_loop_time_constant, 
        #               normalize_input_connection_strengths = self.normalize_input_connection_strengths, 
        #               total_excitatory_input_connection_strength = self.total_excitatory_input_connection_strength, 
        #               total_inhibitory_input_connection_strength = self.total_inhibitory_input_connection_strength, 
        #               offset_flux = self.offset_flux,
        #               self_feedback_coupling_strength = self.self_feedback_coupling_strength)
        
        neuroden_params = params
        neuroden_params['dentype'] = 'soma'
        # neuroden_params['name'] = '{}__{}'.format(self.name,'nr_ni')
        neuron_dendrite = dendrite(**neuroden_params)
        neuron_dendrite.is_soma = True    
        
        self.dend__nr_ni = neuron_dendrite
        self.dend__nr_ni.absolute_refractory_period = self.absolute_refractory_period
        
        # ======================================================================
        #         end receiving and integration dendrite
        # ======================================================================
        
        # ======================================================================
        #         refractory dendrite
        # ======================================================================
        
        
        # create dendrite for neuronal refraction
        # neuron_refractory_dendrite = dendrite(name = '{}__dend_{}'.format(self.name,'refraction'), loops_present = self.loops_present__refraction, circuit_betas = self.circuit_betas__refraction, 
        #                                       junction_critical_current = self.Ic__refraction, junction_beta_c = self.beta_c__refraction,
        #                                       bias_current = self.ib_ref, integration_loop_time_constant = self.integration_loop_time_constant__refraction)
        neuroref_params = params
        neuroref_params['dentype'] = 'refractory'
        # neuroref_params['name'] = '{}__dend_{}'.format(self.name,'refraction')
        self.dend__ref = dendrite(**neuroref_params)
        # self.dend__ref = neuron_refractory_dendrite

        if self.second_ref==True:
            print("SECOND REF")
            neuroref_params = params
            neuroref_params['dentype'] = 'refractory'
            # neuroref_params['name'] = '{}_2__dend_{}'.format(self.name,'refraction')
            self.dend__ref_2 = dendrite(**neuroref_params)
            self.dend__nr_ni.add_input(self.dend__ref_2, connection_strength = self.refractory_dendrite_connection_strength)
        
        if auto:
            
                ib__list__ri, phi_r__array__ri, i_di__array__ri, r_fq__array__ri, phi_th_plus__vec__ri, phi_th_minus__vec__ri, s_max_plus__vec__ri, s_max_minus__vec__ri, s_max_plus__array__ri, s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')
                ib__list__rtti, phi_r__array__rtti, i_di__array__rtti, r_fq__array__rtti, phi_th_plus__vec__rtti, phi_th_minus__vec__rtti, s_max_plus__vec__rtti, s_max_minus__vec__rtti, s_max_plus__array__rtti, s_max_minus__array__rtti = dend_load_arrays_thresholds_saturations('default_rtti')
                if self.loops_present == 'ri':
                    ib_list = ib__list__ri
                    phi_th_minus_vec = phi_th_minus__vec__ri
                    phi_th_plus_vec = phi_th_plus__vec__ri
                    s_max_plus__array = s_max_plus__array__ri
                elif self.loops_present == 'rtti':
                    ib_list = ib__list__rtti
                    phi_th_minus_vec = phi_th_minus__vec__rtti
                    phi_th_plus_vec = phi_th_plus__vec__rtti
                    s_max_plus__array = s_max_plus__array__rtti
                if self.loops_present__refraction == 'ri':
                    ib_list_r = ib__list__ri
                    s_max_plus_vec__refractory = s_max_plus__vec__ri
                elif self.loops_present__refraction == 'rtti':
                    ib_list_r = ib__list__rtti
                    s_max_plus_vec__refractory = s_max_plus__vec__rtti
                    
                _ind_ib_soma = index_finder(ib_list[:],self.dend__nr_ni.ib) # ( np.abs( ib_list[:] - self.dend__nr_ni.ib ) ).argmin()
                
                # for depricated auto calculation
                # phi_th_plus = phi_th_plus_vec[_ind_ib_soma]
                # phi_th_minus = phi_th_minus_vec[_ind_ib_soma]
                # delta = phi_th_plus - phi_th_minus
                
                phi_th_minus = phi_th_minus_vec[_ind_ib_soma]
                _phi_vec_prelim = np.asarray( phi_r__array__ri[_ind_ib_soma] )
                _phi_vec_prelim = _phi_vec_prelim[ np.where( _phi_vec_prelim >= 0 ) ]
                _ind_phi_max = index_finder(_phi_vec_prelim,0.5)
                
                s_max_plus__vec = s_max_plus__array[_ind_ib_soma][:_ind_phi_max]
                
                _ind_s_th = index_finder(s_max_plus__vec,self.s_th)
                phi_a_s_th = _phi_vec_prelim[_ind_s_th]
                delta = phi_a_s_th - phi_th_minus
                                
                _ind_ib_refractory = index_finder(ib_list_r[:],self.dend__ref.ib) # ( np.abs( ib_list_r[:] - self.dend__ref.ib ) ).argmin()
                _s_max_refractory = s_max_plus_vec__refractory[_ind_ib_refractory]
                                
                self.refractory_dendrite_connection_strength = -delta/_s_max_refractory # ( phi_th_minus + delta/100 ) / s_max
                
        self.dend__nr_ni.add_input(self.dend__ref, connection_strength = self.refractory_dendrite_connection_strength)


        # =============================================================================
        #         end refractory dendrite
        # =============================================================================
        


        # =============================================================================
        #         synapse to refractory dendrite
        # =============================================================================
            
        # create synapse for neuronal refraction
        synapse__ref = synapse(
            name = '{}__syn_{}'.format(self.name,'refraction'), 
            tau_rise = self.tau_rise__refraction, 
            tau_fall = self.tau_fall__refraction, 
            hotspot_duration = self.hotspot_duration__refraction, 
            spd_duration = self.spd_duration__refraction, 
            phi_peak = self.phi_peak__refraction)
        
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
        # print(self.name, "<--", connection_object.name)
        self.dend__nr_ni.add_input(connection_object, connection_strength)
        return
        
    def add_output(self, connection_object):
        if type(connection_object).__name__ == 'synapse':
            # print(self.name, "-->", connection_object.name)
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
        self.sim = False
        # make network
        self.uid = network._next_uid
        network._next_uid += 1
        self.unique_label = 'net{}'.format(self.uid)

        self.__dict__.update(kwargs)

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

        if self.sim==True:
            self.simulate()
 
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
        # print(spikes)
        S = []
        Phi_r = []
        spike_signals = []
        count = 0
        # print(self.neurons)
        for neuron_key in self.neurons:
            neuron = self.neurons[neuron_key]
            s = neuron.dend__nr_ni.s
            S.append(s)
            phi_r = neuron.dend__nr_ni.phi_r
            Phi_r.append(phi_r)
            spike_t = neuron.spike_times/neuron.time_params['t_tau_conversion']
            self.neurons[neuron_key].spike_t = spike_t
            # print(spike_t)
            spikes[0].append(np.ones(len(spike_t))*count)
            spikes[1].append((spike_t))
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
        # print(spike_signal)
        # print(spike_signals)

    def simulate(self):
        # print(self.nodes)
        # net = network(name = 'network_under_test')
        for n in self.nodes:
            self.add_neuron(n.neuron)
        self.run_sim(dt=self.dt, tf=self.tf)
        self.get_recordings()

