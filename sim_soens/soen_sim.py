import numpy as np

from .soen_utilities import (
    get_jj_params, 
    dend_load_arrays_thresholds_saturations, 
    physical_constants, 
    index_finder
)
from .soen_time_stepper import run_soen_sim
from .soen_plotting import (
    plot_dendrite, 
    plot_synapse, 
    plot_neuron, 
    plot_neuron_simple, 
    plot_network
)
p = physical_constants()


class input_signal():
    '''
    Input Signal Object:
     - input_signal(
        name='...',
        input_temporal_form = type of input = 'arbitrary_spike_train'
        spike_times = times of spikes for that input channel (ns) = array
        )
    '''
    _next_uid = 0
    input_signals = dict()
    
    def __init__(self, **params):
        
        #make new input signal
        self.uid = input_signal._next_uid
        input_signal._next_uid += 1
        self.unique_label = 'in{}'.format(self.uid)
        self.name = 'unnamed_input_signal__{}'.format(self.unique_label)
        self.input_temporal_form = 'arbitrary_spike_train'
        self.source_type = 'qd'
        self.num_photons_per_spike = 1 


        # UPDATE TO CUSTOM PARAMS
        self.__dict__.update(params)
        

        if self.input_temporal_form  not in ['constant',
                                             'constant_rate', 
                                             'arbitrary_spike_train', 
                                             'analog_dendritic_drive']:
            raise ValueError(f'''
            [soen_sim] Tried to assign an invalid input signal temporal form 
            to input {self.name} (unique_label = {self.unique_label})\n
            The allowed values of input_temporal_form are:
            ''single_spike'', ''constant_rate'', ''arbitrary_spike_train'', 
            and ''analog_dendritic_drive
            ''')

        # currently not supported
        if self.input_temporal_form == 'constant':
            if not hasattr(self,'applied_flux'):
                raise ValueError('''
                [soen_sim] If the input temporal form is constant, applied_flux 
                is required as a keyword argument.''')
        
        # typical form
        elif self.input_temporal_form in ['arbitrary_spike_train']:
            if not hasattr(self,'spike_times'):
                raise ValueError(
            '''
            [soen_sim] arbitrary spike train requires spike_times as input
            ''')
        
        # not supported (constant rate spike train)
        # easier to use np.arange with arbitrary spike train
        if self.input_temporal_form == 'constant_rate':
            if not hasattr(self,'t_first_spike'):
                self.t_first_spike = 50
            if not hasattr(self,'t_first_spike'):
                self.rate= 1

        # currently not supported 
        # drive signals directly to receiving loop
        if self.input_temporal_form == 'analog_dendritic_drive':
            print('analog_dendritic_drive')
            if not hasattr(self,'piecewise_linear'):
                raise ValueError(
                '''
                [soen_sim] must give a value for provide piecewise_linear to use
                           analog_dendritic_drive
                ''')

        input_signal.input_signals[self.name] = self             

            

class dendrite():
    '''
    Dendrite object class
    - keyword arguments (_ni, _n --> soma, _di --> in-arbor dendrite)
        - ib, ib_n, ib_di = bias current 
        - loops_present = type of dendrite ('ri','rtti','pri')
        - beta_di, beta_ni = inductance parameter (2*np.pi*1e[2,3,4,5,6])
        - tau_ni, tau_di = time constant (defines leak rate)
    - a dendrite has sereral key roles
        - somatic dendrite 'dend_soma' --> cell body, attached to transmitter,
          can fire
        - refractory dendrite 'dend_ref' --> receives flux from soma when soma
          fires.  It then couples this flux back to soma inhibitively with a 
          leak rate
        - in-arbor dendrite layi_groupj_dendk --> a dendrite in a tree that 
          eventually leads to soma.  The outermost layer usually hosts the 
          dendrites that are associated with synapses
    '''
    _next_uid = 0
    dendrites = dict()

    def __init__(self, **params):
        
        # DEFAULT SETTINGS
        self.uid = dendrite._next_uid
        self.unique_label = 'd{}'.format(self.uid)
        self.ib_ramp = False
        dendrite._next_uid += 1
        self.name = 'unnamed_dendrite__{}'.format(self.unique_label)
        self.pri=False
        self.offset_trj = []

        # check ahead of time if loop defined in custom params
        # pull default params for that loop
        if 'loops_present' in params:
            self.loops_present = params['loops_present']
        else:
            self.loops_present = 'ri'

        tp = 2*np.pi
        # loop-specific defaul params
        if self.loops_present == 'ri':
            self.circuit_betas = [tp*1/4, tp*1/4, tp*1e2]
            self.ib = 1.802395858835221
            self.ib_di = 1.802395858835221
            

        elif self.loops_present == 'rtti':
            self.circuit_betas = [tp*1/4, tp*1/4, tp*1e2]
            self.ib = 2.0 
            self.ib_di = 2.0 

        elif self.loops_present == 'pri':
            self.circuit_betas = [tp*1/4, tp*1/4, tp*1/4, tp*1e2]
            self.ib = 2.1
            self.phi_p = .2
        string = self.loops_present
        if string == 'pri':
            string = 'ri'
        d_params = dend_load_arrays_thresholds_saturations(f'default_{string}')
        self.ib_max = d_params['ib__list'][-1]
        # print(self.ib_max,self.name)
        
        # miscelleneous params/settings
        self.tau_di = 250
        self.beta_di = 2*np.pi*1e2
        self.Ic =  100
        self.beta_c =  0.3
        self.normalize_input_connection_strengths = False
        self.total_excitatory_input_connection_strength = 1
        self.total_inhibitory_input_connection_strength = -0.5
        self.offset_flux = 0
        self.self_feedback_coupling_strength = 0
        self.rollover = 0
        self.valleyedout = 0
        self.doubleroll = 0

        # UPDATE TO CUSTOM PARAMS
        self.__dict__.update(params)
        
        if hasattr(self, 'dentype'):
            if self.dentype == 'refractory':
                # print("REFRACTORY DENDRITE")
                self.loops_present = self.loops_present__refraction
                self.circuit_betas = self.circuit_betas__refraction 
                self.Ic = self.Ic__refraction
                self.beta_c = self.beta_c__refraction
                self.ib = self.ib_ref
                self.tau_di = self.tau_ref   
                self.name = f'{self.name}__dend_refraction'
            elif self.dentype == 'soma':
                # print("SOMATIC DENDRITE")
                self.tau_di = self.tau_ni
                self.ib = self.ib_n
                self.beta_di = self.beta_ni
                self.name = f'{self.name}_soma'
        else:
            # print("REGULAR DENDRITE")
            if "ib_di" in self.__dict__.keys():
                self.ib = self.ib_di
            else:
                self.ib = self.ib
            if "dend_name" in self.__dict__.keys():
                self.name = self.dend_name
        self.circuit_betas[-1] = self.beta_di
            
        if 'integrated_current_threshold' in params:
            self.s_th = params['integrated_current_threshold']
        
        # params = self.__dict__
        self.bias_current = self.ib
        self.bias_dynamics = [self.ib]

        # warnings
        if self.loops_present not in ['ri','pri','rtti','prtti']:
            raise ValueError('''
            [soen_sim] loops_present must be:
                \'ri\', \'pri\', \'rtti\', \'prtti\'. ''')

        bs = [3,4,5]
        if type(self.circuit_betas)==list and len(self.circuit_betas) not in bs:
            raise ValueError('''
            [soen_sim] circuit_betas list must equal number of inductors for the
            given circuit.  Note that the value for the DI loop is the total
            inductance.
                ri -> 3
                rtti -> 3
                pri -> 4''')

        jj_params = get_jj_params(self.Ic*1e-6,self.beta_c)

        if type(self.tau_di).__name__ == 'list': 
            tau_vs_current = self.tau_di
            self.tau_list, self.s_list = [], []
            for tau_s in tau_vs_current:
                self.tau_list.append(tau_s[0])
                self.s_list.append(tau_s[1])
            self.tau_list.append(self.tau_list[-1]) # add entry with large s
            self.s_list.append(1e6) # add entry with large s
            self.tau_di = self.tau_list[0]  

        # jj_params setup based on custom params
        tau_di = self.tau_di * 1e-9
        beta_di = self.beta_di
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
        '''
        Add any object with an output as input to this object
         - connection strength kwarg defines weighting of connection
        '''
        name = connection_object.name
        cs = connection_strength
        if type(connection_object).__name__ == 'input_signal':
            self.external_inputs[name] = input_signal.input_signals[name]
            self.external_connection_strengths[name] = cs

        if type(connection_object).__name__ == 'synapse':
            self.synaptic_inputs[name] = synapse.synapses[name]
            self.synaptic_connection_strengths[name] = cs
            
        if type(connection_object).__name__ == 'dendrite':
            self.dendritic_inputs[name] = dendrite.dendrites[name]            
            self.dendritic_connection_strengths[name] = cs
            
        if type(connection_object).__name__ == 'neuron':
            self.dendritic_inputs[name] = neuron.neurons[name]            
            self.dendritic_connection_strengths[name] = cs
        
        return self
    
    def run_sim(self, **kwargs):
        self = run_soen_sim(self, **kwargs)
        return self

    # deprecated    
    # def plot(self):
    #     plot_dendrite(self)
    #     return

    def __del__(self):
        # print(f'dendrite {self.name} deleted')
        return
    


class synapse():  
    '''
    Synapse object class 
     - can be attached to any dendritic receiving loop 
     - default settings are closest to current empirical data
     - best left as is
     - steep rise (tau_rise) followed by slower decay (tau_fall)
     - phi_peak defines amplitude of rise
     - reset time defines the period that no new photons can be received
    '''

    _next_uid = 0
    synapses = dict()
    
    def __init__(self, **params):

        # make new synapse
        # self._instances.add(weakref.ref(self))
        self.uid = synapse._next_uid
        synapse._next_uid += 1
        self.unique_label = 's{}'.format(self.uid)
        self.name = 'unnamed_synapse__{}'.format(self.unique_label)

        # synaptic receiver specification
        self.tau_rise = 0.02 # 20ps is default rise (L_tot/r_ph = 100nH/5kOhm)
        self.tau_fall = 50 # 50ns is default fall time for SPD recovery
        self.hotspot_duration = 3 #3 # two time constants is default
        self.spd_duration = 8
        self.phi_peak = 0.5 # default peak flux is Phi0/2
        self.spd_reset_time = 50 #self.tau_fall
        
        self.__dict__.update(params)

        self.hotspot_duration *=  self.tau_rise
        self.spd_duration *= self.tau_fall
     
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
    '''
    Neuron object class
    - keyword arguments (_ref applies to refractory dendrite)
        - ib, ib_n, ib_ref = bias current 
        - loops_present, loops_present_ref = dendrite type ('ri','rtti','pri')
        - beta_ni, beta_ref = inductance parameter (2*np.pi*1e[2,3,4,5,6])
        - tau_ni, tau_ref = time constant (defines leak rate)
        - s_th = spiking threshold
    - the neuron class automatically constructs soma and refractory dendrites
    '''
    _next_uid = 0
    neurons = dict()
    
    def __init__(self, **params):

        # DEFAULT SETTINGS
        self.uid = neuron._next_uid
        neuron._next_uid += 1
        self.unique_label = 'n{}'.format(self.uid)
        self.name = 'unnamed_neuron__{}'.format(self.unique_label)

        # dendrite parameters
        if 'loops_present' in params:
            self.loops_present = params['loops_present']
        else:
            self.loops_present = 'ri'
        self.beta_ni = 2*np.pi*1e2
        self.Ic =  100
        self.beta_c =  0.3
        if self.loops_present == 'ri':
            self.ib = 1.802395858835221 #1.7 # dimensionless bias current   
            self.ib_n = 1.802395858835221
            # self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, self.beta_ni]
        elif self.loops_present == 'rtti':
            self.ib = 2.19 # dimensionless bias current
            self.ib_n = 2.19
            self.ib_di= 2.19
            # self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, self.beta_ni] 
        self.integration_loop_time_constant = 250
        self.absolute_refractory_period = 10
        self.normalize_input_connection_strengths = False
        self.total_excitatory_input_connection_strength = 1
        self.total_inhibitory_input_connection_strength = -0.5
        self.offset_flux = 0 
        # J_ii, units of phi/s (normalized flux / normalized current in DI loop)
        self.self_feedback_coupling_strength = 0
        self.tau_ni = 50
        self.integrated_current_threshold = self.s_th = 0.5

        # refractory dendrite
        if 'loops_present__refraction' in params:
            self.loops_present__refraction = params['loops_present__refraction']
        else:
            self.loops_present__refraction = 'ri'
        self.beta_ref = 2*np.pi*1e2
        if self.loops_present__refraction == 'ri':
            self.circuit_betas__refraction = [2*np.pi* 1/4, 
                                              2*np.pi* 1/4, 
                                              self.beta_ref]         
        if self.loops_present__refraction  == 'rtti':
            self.circuit_betas__refraction = [2*np.pi* 1/4, 
                                              2*np.pi* 1/4, 
                                              2*np.pi*0.5, 
                                              2*np.pi*0.5, 
                                              self.beta_ref]
        self.Ic__refraction =  100
        self.beta_c__refraction =  0.3
        if self.loops_present__refraction == 'ri':
            self.ib_ref = 1.7    
        elif self.loops_present__refraction == 'rtti':
            self.ib_ref = 3.1 
        self.tau_ref= 50
        dend_ref_cs = 'auto'
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
        self.light_production_delay = 2


        # UPDATE TO CUSTOM PARAMS
        self.__dict__.update(params)
        self.circuit_betas = [2*np.pi* 1/4, 2*np.pi* 1/4, self.beta_ni]
        self.integrated_current_threshold = self.s_th
        params = self.__dict__

        # jj setup
        jj_params = get_jj_params(self.Ic*1e-6,self.beta_c)

        tau_ni = self.tau_ni * 1e-9
        beta_ni = self.beta_ni
        Ic = self.Ic * 1e-6
        Lni = p['Phi0']*beta_ni/(2*np.pi*Ic)
        rni = Lni/tau_ni
        self.alpha = rni/jj_params['r_j']
        self.jj_params = jj_params


        ### refractory dendrite ###
        self.circuit_betas__refraction[-1] = self.beta_ref

        jj_params__refraction = get_jj_params(self.Ic__refraction*1e-6,
                                              self.beta_c__refraction)
        tau_ref = self.tau_ref * 1e-9
        beta_nr = self.beta_ref
        Ic = self.Ic__refraction * 1e-6
        Lnr = p['Phi0']*beta_nr/(2*np.pi*Ic)
        r_ref = Lnr/tau_ref
        self.alpha__refraction = r_ref/jj_params['r_j']
        self.jj_params__refraction = jj_params__refraction

        ref_connect = dend_ref_cs
        if type(ref_connect).__name__ != 'str':
            auto = False
        elif ref_connect == 'auto':
            auto = True
        elif ref_connect == 'match_excitatory':
            ref_connect = self.total_excitatory_input_connection_strength
            auto = False


        ### transmitter ###
        if self.source_type not in ['ec' ,'qd','delay_delta']:
            raise ValueError('''[soen_sim] sources presently defined are: 
                                    'qd', 'ec', or 'delay_delta' ''')

           
        # ======================================================================
        #                               soma
        # ======================================================================
        
        neuroden_params = params
        neuroden_params['dentype'] = 'soma'
        # neuroden_params['name'] = '{}__{}'.format(self.name,'soma')
        neuron_dendrite = dendrite(**neuroden_params)
        neuron_dendrite.is_soma = True    
        
        self.dend_soma = neuron_dendrite
        ref = self.absolute_refractory_period
        self.dend_soma.absolute_refractory_period=ref
        
        
        # ======================================================================
        #                       refractory dendrite
        # ======================================================================

        neuroref_params = params
        neuroref_params['dentype'] = 'refractory'
        self.dend__ref = dendrite(**neuroref_params)
        # self.dend__ref = neuron_refractory_dendrite

        if self.second_ref==True:
            print("SECOND REF")
            neuroref_params = params
            neuroref_params['dentype'] = 'refractory'
            self.dend__ref_2 = dendrite(**neuroref_params)
            self.dend_soma.add_input(
                self.dend__ref_2, 
                connection_strength=dend_ref_cs
                )
        
        # automatically normalizes refractory strength
        if auto:
            d_params_ri = dend_load_arrays_thresholds_saturations('default_ri')
            d_params_rtti = dend_load_arrays_thresholds_saturations('default_rtti')
            if self.loops_present == 'ri':
                ib_list = d_params_ri["ib__list"]
                phi_th_minus_vec = d_params_ri["phi_th_minus__vec"]
                phi_th_plus_vec = d_params_ri["phi_th_plus__vec"]
                s_max_plus__array = d_params_ri["s_max_plus__array"]
            elif self.loops_present == 'rtti':
                ib_list = d_params_rtti["ib__list"]
                phi_th_minus_vec = d_params_rtti["phi_th_minus__vec"]
                phi_th_plus_vec = d_params_rtti["phi_th_plus__vec"]
                s_max_plus__array = d_params_rtti["s_max_plus__array"]

            if self.loops_present__refraction == 'ri':
                ib_list_r = d_params_ri["ib__list"]
                s_max_plus_vec__refractory = d_params_ri["s_max_plus__vec"]
            elif self.loops_present__refraction == 'rtti':
                ib_list_r = d_params_rtti["ib__list"]
                s_max_plus_vec__refractory = d_params_rtti["s_max_plus__vec"]
                
            # ( np.abs( ib_list[:] - self.dend_soma.ib ) ).argmin()
            _ind_ib_soma = index_finder(ib_list[:],self.dend_soma.ib) 
            
            phi_th_minus=phi_th_minus_vec[_ind_ib_soma]
            _phi_vec_prelim=np.asarray(d_params_ri["phi_r__array"][_ind_ib_soma])
            _phi_vec_prelim=_phi_vec_prelim[ np.where( _phi_vec_prelim >= 0 ) ]
            _ind_phi_max=index_finder(_phi_vec_prelim,0.5)
            
            s_max_plus__vec = s_max_plus__array[_ind_ib_soma][:_ind_phi_max]
            
            _ind_s_th = index_finder(s_max_plus__vec,self.s_th)
            phi_a_s_th = _phi_vec_prelim[_ind_s_th]
            delta = phi_a_s_th - phi_th_minus

            # ( np.abs( ib_list_r[:] - self.dend__ref.ib ) ).argmin()           
            _ind_ib_refractory = index_finder(ib_list_r[:],self.dend__ref.ib) 
            _s_max_refractory = s_max_plus_vec__refractory[_ind_ib_refractory]

            # ( phi_th_minus + delta/100 ) / s_max              
            dend_ref_cs = -delta/_s_max_refractory 
                
        self.dend_soma.add_input(
            self.dend__ref,
            connection_strength = dend_ref_cs
            )


        # ======================================================================
        #         end refractory dendrite
        # ======================================================================
        


        # ======================================================================
        #         synapse to refractory dendrite
        # ======================================================================
            
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
        
        # ======================================================================
        #         end synapse to refractory dendrite
        # ======================================================================
        

        
        # prepare for spikes        
        self.spike_times = []
        self.spike_indices = []
        self.dend_soma.spike_times = []
        
        # prepare for output synapses
        self.synaptic_outputs = dict()

        neuron.neurons[self.name] = self
        return    
        
    def add_input(self, connection_object, connection_strength = 1):
        # print(self.name, "<--", connection_object.name)
        self.dend_soma.add_input(connection_object, connection_strength)
        return
        
    def add_output(self, connection_object):
        if type(connection_object).__name__ == 'synapse':
            name = connection_object.name
            # print(self.name, "-->", connection_object.name)
            self.synaptic_outputs[name] = synapse.synapses[name]
            synapse.synapses[name].add_input(self)
        else: 
            raise ValueError('[soen_sim] a neuron can only output to a synapse')     
        return
    
    def run_sim(self, **kwargs):
        self = run_soen_sim(self, **kwargs)
        return self

    ## deprecated --> use node.plot_neuron_activity(net)    
    # def plot(self):
    #     if self.plot_simple:
    #         plot_neuron_simple(self)
    #     else:
    #         plot_neuron(self)
    #     return

    def __del__(self):
        # print('dendrite deleted')
        return
    

class network():
    '''
    Network object class
     - 
    '''
    _next_uid = 0
    network = dict()
    
    def __init__(self, **kwargs):
        self.sim = False
        # make network
        self.uid = network._next_uid
        network._next_uid += 1
        self.unique_label = 'net{}'.format(self.uid)
        self.new_way=False
        self.null_synapses=True
        self.nodes=[]
        self.dt = 0.1
        self.tf = 250
        self.timer=False
        self.backend = 'python'
        self.name = 'unnamed_network__{}'.format(self.unique_label)

        self.__dict__.update(kwargs)
        
        # JJ params
        # make dummy dendrite to obtain default Ic, beta_c
        dummy_dendrite = dendrite(name='dummy') 
        jj_params = get_jj_params(dummy_dendrite.Ic*1e-6,dummy_dendrite.beta_c)
        self.jj_params = jj_params # this should be done somewhere else globally
        
        # prepare network to have neurons
        self.neurons = dict()

        if self.sim==True:
            self.simulate()

    def __copy__(self):
        copy_object = network()
        return copy_object

    def __deepcopy__(self, memodict={}):
        copy_object = network()
        copy_object.time_params= self.time_params
        copy_object.neurons= self.neurons
        copy_object.nodes= self.nodes
        copy_object.t= self.t
        copy_object.spikes= self.spikes
        copy_object.spike_signals= self.spike_signals
        copy_object.phi_r = self.phi_r
        copy_object.signal= self.signal
        copy_object.dt= self.dt
        copy_object.tf= self.tf
        return copy_object
 
    def add_neuron(self, neuron_object):
        self.neurons[neuron_object.name] = neuron_object
        # if self.neurons[neuron_object.name] not in self.nodes:
        #     self.nodes.append(self.neurons[neuron_object.name])
        # return
    
    def run_sim(self, **kwargs):
        self = run_soen_sim(self)
        return self

    ## deprecated, use activitiy_plot(nodes,net)    
    # def plot(self):
    #     plot_network(self)
    #     return

    def get_recordings(self):
        '''
        Collects network data and makes accessible
         - net.spikes = [indices, times] of spikes
         - net.neuron.phi_r = received flux history of that neuron
         - net.neuron.s = soma signal history of that neuron
        '''
        self.t = self.time_params['time_vec'] 
        spikes = [ [] for _ in range(2) ]
        # print(spikes)
        S = []
        Phi_r = []
        spike_signals = []
        count = 0
        # print(self.neurons)
        for neuron_key in self.neurons:
            neuron = self.neurons[neuron_key]
            s = neuron.dend_soma.s
            S.append(s)
            phi_r = neuron.dend_soma.phi_r
            Phi_r.append(phi_r)
            spike_t = neuron.spike_times/neuron.time_params['t_tau_conversion']
            self.neurons[neuron_key].spike_t = spike_t
            spikes[0].append(np.ones(len(spike_t))*count)
            spikes[1].append((spike_t))
            spike_signal = []
            spike_times = spike_t #/neuron.time_params['t_tau_conversion']
            for spike in spike_times:
                spot = int(spike/self.dt)
                spread = int(5/self.dt)
                spike_signal.append(np.max(s[np.max([0,spot-spread]):spot+spread]))
                # spike_signal.append(s[spot])
            spike_signals.append(spike_signal)
            count+=1
        spikes[0] = np.concatenate(spikes[0])
        spikes[1] = np.concatenate(spikes[1])
        self.spikes = spikes
        self.spike_signals = spike_signals
        self.phi_r = Phi_r
        self.signal = S


    def simulate(self):
        '''
        Simulates network
         - adds nodes to net
         - checks if any synapses without input and gives empty array of spikes
         - runs simulation
         - gets recordings
        '''
        for n in self.nodes:
            self.add_neuron(n.neuron)
        if self.null_synapses == True:
            count=0
            for n in self.nodes:
                for syn in n.synapse_list:
                    if "synaptic_input" not in syn.__dict__:
                        syn.add_input(input_signal(
                            name = 'input_synaptic_drive', 
                            input_temporal_form = 'arbitrary_spike_train', 
                            spike_times = []
                            ))
                        count+=1
            # print(f"{count} synapses recieving no input.")
        self.run_sim()
        self.get_recordings()


class HardwareInTheLoop:
    '''
    HardwareInTheLoop object class
     - takes expected spiking value at certain intervals for array of neurons
     - at end of interval, checks spikes with .forward_pass()
     - then propogates error onto trace dendrites with .backward_pass() in the 
       form of received spikes to trace SPDs added for next interval
        - proportionate to size of error
    '''
    def __init__(self, **params):
        self.expect = [[0,50],[None,None],[50,0],[None,None],[None,None]]
        self.interval = 500
        self.phase = 0
        self.error_factor = 10
        # self.traces=None
        self.traces = []
        self.__dict__.update(params)
        
        self.check_time = self.interval*(self.phase+1)
        self.errors = [[] for i in range(len(self.expect))]


    def forward_error(self,neurons):
        '''
        Returns difference between actual and expected spikes for each neuron
            - Only counts spikes for a given interval phase
            - **rewrite this more optimally (probably with list comprehension)
        '''
        counts = [0 for i in range(len(neurons))]
        for i,n in enumerate(neurons):
            # print(type(n.neuron))
            if "spike_times" in n.neuron.__dict__:
                for spk in n.neuron.spike_times:
                    spk = spk/self.conversion
                    if (spk > self.check_time - self.interval 
                        and spk < self.check_time):
                        counts[i]+=1
            else:
                counts[i]=0

        # If no expectation, error will equal zero
        # for i,ex in enumerate(self.expect[self.phase]):
        #     if ex==None:
                # self.expect[self.phase][i] = counts[i]
        print("phase:",self.phase)
        if self.expect[self.phase][0] == None:
            self.expect[self.phase][0] = counts[0]
        if self.expect[self.phase][1] == None:
            self.expect[self.phase][1] = counts[1]
            

        self.errors[self.phase] = np.subtract(counts,self.expect[self.phase])
        # print(counts,self.expect[self.phase])

    def backward_error(self,nodes):
        freq_factor = self.freq_factor
        error = self.errors[self.phase]
        conv = self.conversion
        # print("self ERROR: ", error,"\n")
        for i in range(len(error)):
            for dend in nodes[i].trace_dendrites:
                for name,syn in dend.synaptic_inputs.items():
                    self.traces.append(dend)

                    if error[i] < 0:
                        if 'plus' in syn.name:
                            # print("plus error: ",error[i])
                            freq = np.max([300-np.abs(error[i])*freq_factor,50])
                            # print("plus frequency: ",freq)
                            syn.input_signal.spike_times += list(
                                np.arange(
                                self.check_time,
                                self.check_time+self.interval,
                                freq))
                            st = syn.input_signal.spike_times
                            syn.spike_times_converted = np.asarray(st)*conv

                    elif error[i] > 0:
                        if 'minus' in syn.name:
                            # print("minus error: ",error[i])
                            freq = np.max([300-np.abs(error[i])*freq_factor,50])
                            # print("minus frequency: ",freq)
                            syn.input_signal.spike_times += list(
                                np.arange(
                                self.check_time,
                                self.check_time+self.interval,
                                freq))
                            st = syn.input_signal.spike_times
                            syn.spike_times_converted = np.asarray(st)*conv

                    # print(syn.name,syn.input_signal.spike_times)
                        
            # print("\n")
        self.phase+=1
        self.check_time = self.interval*(self.phase+1)
        self.trace_biases = {}
        for trace in self.traces:
            self.trace_biases[trace.name] = []
            # return net
