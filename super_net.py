import numpy as np

from soen_utilities import index_finder, dend_load_arrays_thresholds_saturations
from soen_sim import input_signal, network
from soen_component_library import common_dendrite, common_synapse, common_neuron
from super_library import NeuralZoo
"""
### THIS FILE TO BE REWRITTEN ###
ToDo:
 - Find way to generate structure only once, for any input
 - Find cleaner way of dealing with parameter adjustments
 
 Proposed input method:

input = Input(Channels=100, type=[random,MNIST,audio,custom])

neuron_pupulation = Neurons(N=100, connectivity=[random,structured,custom], **kwargs)
 - pass in dictionary of parameter settings through kwargs, else defaults
 - Can customize connectivity with an adjacency matrix

monitor - Monitor(neuron_population, ['spikes','phi_r','signal','etc...'])

network = Network(input,neuron_population,monitor)

network.run(simulation_time*ns)
"""

class BaseNet:
    pass

class FractalNet():
    '''
    
    '''
    def __init__(self,**params):
        # default params
        self.N = 4
        self.duration = 100
        self.run=False

        # custom input params
        self.__dict__.update(params)

        # make and potentially run neurons and network
        self.make_neurons(**params)
        self.make_net()
        if self.run == True:
            self.run_network()

    def make_neurons(self,**params):
        '''
        Make required number of neurons with default parameters
         - Store in a list `neurons`
        '''
        self.neurons = []
        W = [
            [[.5,.5,.5]],
            [[.5,.5,.5],[.5,.5,.5],[.5,.5,.5]]
            ]
        
        for i in range(self.N):
            neuron = NeuralZoo(type='custom',weights=W,**params)
            neuron.synaptic_layer()
            self.neurons.append(neuron)
        # print(neurons[0].synapses)
        # for k,v in neurons[0].__dict__.items():
        #     print(k,": ",v,"\n")
        # neurons[0].plot_custom_structure()
        # print(neurons[0].neuron.dend__nr_ni.dendritic_inputs['n0_lay0_branch0_den1'].dendritic_inputs['n0_lay1_branch1_den0'].synaptic_inputs['3'].name)

    def make_net(self):
        self.layer_n = 3
        branches = 3
        for i in range(1,self.layer_n+1):
            # print(self.neurons[i].synapse_list)
            for j in range(branches):
                self.neurons[i].neuron.add_output(self.neurons[0].synapse_list[(j*3)+(i-1)])
                # print(self.neurons[0].neuron.name, i, (j*3)+(i-1), self.neurons[0].synapse_list[(j*3)+(i-1)].name)
        # print(self.neurons[0].neuron.dend__nr_ni.dendritic_inputs['n0_lay0_branch0_den1'].dendritic_inputs['n0_lay1_branch1_den0'].synaptic_inputs['3'].__dict__)
        # print(self.neurons[1].neuron.__dict__)
        # for i in range(self.N):
        #     print(i," - ", self.neurons[i].synapse_list[0].__dict__)
        # for i in range(9):
        #     print(self.neurons[0].synapse_list[i].input_signal.name)
        # print("\n\n")    

    def connect_input(self,inputs):
        count=0
        for i in range(1,self.layer_n+1):
            for j in range(9):
                input = input_signal(name = 'input_synaptic_drive'+str(i)+str(j), 
                                    input_temporal_form = 'arbitrary_spike_train', 
                                    spike_times = inputs.spike_rows[count])
                # print(input.spike_times)
                self.neurons[i].synapse_list[j].add_input(input)
                # print(self.neurons[i].synapse_list[j].input_signal.__dict__)
                count+=1 
        # for i in range(self.N):
        #     print(i," - ", self.neurons[i].synapse_list[0].__dict__)
        #     if i !=0:
        #         print(self.neurons[i].synapse_list[4].input_signal.name)

    def run_network(self):
        self.net = network(dt=0.1,tf=5000,nodes=self.neurons)
        # for n in range(self.N):
        #     self.net.add_neuron(self.neurons[n])
        print("running network")
        self.net.simulate()



class PointReservoir:
    '''
    
    '''
    def __init__(self,**params):
        # default params
        self.N = 72
        self.duration = 1000
        self.run=False
        self.dt = 0.1
        self.tf = 360*9

        # custom input params
        self.__dict__.update(params)

        # make and potentially run neurons and network
        self.make_neurons(**params)
        self.make_net()

    def make_neurons(self,**params):
        '''
        Start with ideal situation
            - Input in
            - Internal connectivity
        '''
        self.neurons = []
        syn_struct = [
            [[[1]]],
            [[[1]]],
            [[[1]]],
            [[[1]]],
            [[[1]]],
            [[[1]]],
            [[[1]]],
            [[[1]]],
            [[[1]]],
            [[[1]]],
        ]
        
        for i in range(self.N):
            neuron = NeuralZoo(type='custom',name=f'res_neuron_{i}',synaptic_structure=syn_struct,seed=self.run*1000+i,**params) # name=f'res_neuron_{i}',
            # neuron.synaptic_layer()
            self.neurons.append(neuron)

    def make_net(self):
        np.random.seed(self.run)
        connections = 0
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    num = np.random.randint(100)
                    # print(num)
                    if num < 10:
                        # print(len(self.neurons[j].synapse_list))
                        for syn in self.neurons[j].synapse_list:
                            if "synaptic_input" not in syn.__dict__:
                                self.neurons[i].neuron.add_output(syn)
                                # syn.connection_strength = np.random.rand()
                                connections+=1
                                break
        # print("Reservoir connections: ", connections)

    def connect_input(self,input):
        connections = 0
        syn_finder = 0
        for repeat in range(10):
            for i,row in enumerate(input.spike_rows):
                # print((len(input.spike_rows)*repeat+i)%72)
                for syn in self.neurons[(len(input.spike_rows)*repeat+i)%72].synapse_list:
                    if "synaptic_input" not in syn.__dict__:
                        array = np.sort(row)
                        array = np.append(array,np.max(array)+.001)
                        syn.add_input(input_signal(name = 'input_synaptic_drive', 
                                            input_temporal_form = 'arbitrary_spike_train', 
                                            spike_times = array) )
                        # print(syn.__dict__.synaptic_input)
                        connections += 1 
                        break
        # print("Input connections: ", connections)


        # connections = 0
        # syn_finder = 0
        # while syn_finder == 0:
        #     for row in input.spike_rows:
        #         syn_found = 0
        #         for n in self.neurons:
        #             for syn in n.synapse_list:
        #                 if "synaptic_input" not in syn.__dict__:
        #                     array = np.sort(row)
        #                     array = np.append(array,np.max(array)+.001)
        #                     syn.add_input(input_signal(name = 'input_synaptic_drive', 
        #                                         input_temporal_form = 'arbitrary_spike_train', 
        #                                         spike_times = array) )
        #                     # print(syn.__dict__.synaptic_input)
        #                     connections += 1 
        #                     syn_found += 1
        #                     print(syn_found,connections)
        #                     break
        #             if syn_found > 0:
        #                 break
        #         if syn_found == 0:
        #             syn_finder += 1
        #         # syn_found = 0
        # print("Input connections: ", connections)
        # self.neurons[i].synapse_list[j].add_input(input)


    def run_network(self,prune_synapses=True):
        self.net = network(dt=self.dt,tf=self.tf,nodes=self.neurons,new_way=False)
        print("running network")
        self.net.simulate(prune_synapses)


class SuperNet:
    '''
    Organizes a system and structure of loop neurons
    '''

    def __init__(self,**entries):
        self.N = 10
        self.duration = 100
        self.name = 'Super_Net'
        self.connectivity = 'random'
        self.in_connect = 'ordered'
        self.dend_type = 'default_ri'
        self.recurrence = None
        # self.__dict__.update(entries['params'])
        self.ib__list__ri, self.phi_r__array__ri, self.i_di__array__ri, self.r_fq__array__ri, self.phi_th_plus__vec__ri, self.phi_th_minus__vec__ri, self.s_max_plus__vec__ri, self.s_max_minus__vec__ri, self.s_max_plus__array__ri, self.s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')
        self.__dict__.update(entries)
        self.param_setup()
        self.make_neurons()


    def param_setup(self):
        np.random.seed(0)
        '''
        Initializes empty list for each neuron specific parameter and then appends N terms
         - This function would likely be circumvented by passing in a parameter matrix (N x p)
            - *Should coordinate on preferred organization for parameter passing
        '''
        N = self.N

        self.BETA_DI = []
        self.TAU_DI = []
        self.IB = []
        self.S_MAX = []
        self.PHI_TH = []
        self.IB_N = []
        self.S_TH_FACTOR_N = []
        self.S_MAX_N = []
        self.PHI_TH_N = []
        # self.beta_ni = []
        # self.tau_ni = []

        self.W_SD = []
        self.W_SID = []
        self.W_DN = []


        for n in range(N):
            # dendrites
            self.BETA_DI.append(self.beta_di)
            self.TAU_DI.append(np.random.randint(self.tau_di[0],self.tau_di[1])) #self.tau_di
            self.IB.append(self.ib__list__ri[np.random.randint(7,10)])
            self.S_MAX.append(self.s_max_plus__vec__ri[index_finder(self.IB[n],self.ib__list__ri[:])])
            self.PHI_TH.append(self.phi_th_plus__vec__ri[index_finder(self.IB[n],self.ib__list__ri[:])])
            # neurons
            self.IB_N.append(self.ib__list__ri[np.random.randint(7,10)])
            self.S_TH_FACTOR_N.append(self.s_th_factor_n)
            self.S_MAX_N.append(self.s_max_plus__vec__ri[index_finder(self.IB_N[n],self.ib__list__ri[:])])
            self.PHI_TH_N.append(self.phi_th_plus__vec__ri[index_finder(self.IB_N[n],self.ib__list__ri[:])])
            
            # weights
            if len(self.w_sid) > 1:
                self.W_SID.append(np.random.uniform(self.w_sid[0],self.w_sid[0])) # 0.9
            else:
                self.W_SID.append(self.w_sid[0]) # 0.9

            if len(self.w_sd) > 1:
                self.W_SD.append(np.random.uniform(self.w_sd[0],self.w_sd[0])/self.norm_sd)  # 0.9
            else:
                self.W_SD.append(self.w_sd[0])  # 0.9

            if len(self.w_dn) > 1:
                self.W_DN.append(((np.random.uniform(self.w_dn[0],self.w_dn[0]))/(2*self.S_MAX[n]) / self.norm_sd))  # 0.5
            else:
                self.W_DN.append(self.w_dn[0]/(2*self.S_MAX[n]))  # 0.5


        self.ib_ref = self.ib__list__ri[self.ib_ref]


    def make_neurons(self):
        '''
        Creates N synapses and dendrites and feeds each synapse with all input given some propability p
            - 
        '''
        np.random.seed(0)
        self.synapses = []
        self.dendrites = []
        for n in range(self.N):
            self.synapses.append(common_synapse(n+1))
            self.dendrites.append(common_dendrite(n+1, 'ri', self.BETA_DI[n], self.TAU_DI[n], self.IB[n]))

        self.neurons = []  
        for n in range(self.N):
            self.dendrites[n].add_input(self.synapses[n], connection_strength = self.W_SD[n])
            self.neurons.append(common_neuron(n, 'ri', self.beta_ni, self.tau_ni, self.IB_N[n], self.S_TH_FACTOR_N[n]*self.S_MAX_N[n], self.beta_ref, self.tau_ref, self.ib_ref))
            self.neurons[n].add_input(self.dendrites[n], connection_strength = self.W_DN[n])
            
        # random topology
        if self.connectivity == "random":
            for i in range(self.N):
                for j in range(self.N):
                    if np.random.rand() < self.reservoir_p:
                        self.neurons[i].add_output(self.synapses[j])

        if self.connectivity == "cascade":
            for n in range(self.N):
                if n < self.N-1:
                    self.neurons[n].add_output(self.synapses[n+1])
                else:
                    self.neurons[n].add_output(self.synapses[0])

    def connect_input(self,input):
        in_spikes = input.spike_rows
        self.inputs = []
        self.synapse_in = []
        count = 0
        for i, inp in enumerate(in_spikes):
            if np.any(inp):
                self.inputs.append(input_signal(name = 'input_synaptic_drive', input_temporal_form = 'arbitrary_spike_train', spike_times = inp)) 
                self.synapse_in.append(common_synapse(10000+i))
                self.synapse_in[count].add_input(self.inputs[count])
                count+=1
        print("input neurons: ", len(self.inputs))
        # print(self.inputs[1].spike_times)
        ### change for more complex input
        if self.in_connect == "random":
            p = self.input_p
            for i in range(len(self.inputs)):
                for j in range(self.N):
                    rnd = np.random.rand() 
                    if rnd < p:
                        # print(i,j)
                        self.dendrites[j].add_input(self.synapse_in[i], connection_strength = self.W_SID[j])

        elif self.in_connect == "ordered":
            for i in range(len(self.synapse_in)):
                if i < self.N:
                    self.dendrites[i].add_input(self.synapse_in[i], connection_strength = self.W_SID[i])
                else:
                    self.dendrites[i-self.N].add_input(self.synapse_in[i], connection_strength = self.W_SID[i-self.N])
        self.make_net()

    def make_net(self):
        # create network
        self.net = network(name = 'network_under_test')

        # add neurons to network
        for n in range(self.N):
            self.net.add_neuron(self.neurons[n])

    def run(self,dt=None):
        # self.net.run_sim(dt = self.dt_soen, tf = self.inputs[0].spike_times[-1] + np.max([self.tau_di] ))
        if dt:
            self.net.run_sim(dt = dt, tf = self.duration + np.max(self.tau_di))
        else:
            self.net.run_sim(dt = self.dt_soen, tf = self.duration + np.max(self.tau_di))

    def record(self,params):
        recordings = {}
        if 'spikes' in params:
            print('spikes')
            spikes = [ [] for _ in range(2) ]
        S = []
        Phi_r = []
        count = 0
        for neuron_key in self.net.neurons:

            s = self.net.neurons[neuron_key].dend__nr_ni.s
            S.append(s)

            phi_r = self.net.neurons[neuron_key].dend__nr_ni.phi_r
            Phi_r.append(phi_r)

            spike_t = self.net.neurons[neuron_key].spike_times
            spikes[0].append(np.ones(len(spike_t))*count)
            spikes[1].append(spike_t/self.neurons[neuron_key].time_params['t_tau_conversion'])
            count+=1
        spikes[0] =np.concatenate(spikes[0])
        spikes[1] = np.concatenate(spikes[1])
        self.spikes = spikes

    def plot_signals():
        pass

    def spks_to_txt(self,spikes,prec,dir,name):
        """
        Convert Brain spikes to txt file
        - Each line is a neuron index
        - Firing times are recorded at at their appropriate neuron row
        """
        import os
        dirName = f"results_/{dir}"
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass

        indices = spikes[0]
        times = spikes[1]
        with open(f'{dirName}/{name}.txt', 'w') as f:
            for row in range(self.N):
                for i in range(len(indices)):
                    if row == indices[i]:
                        if row == 0:
                            f.write(str(np.round(times[i],prec)))
                            f.write(" ")
                        else:
                            f.write(str(np.round(times[i],prec)))
                            f.write(" ")
                f.write('\n')


    