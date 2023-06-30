import numpy as np

from sim_soens.soen_utilities import index_finder, dend_load_arrays_thresholds_saturations
from sim_soens.soen_sim import input_signal, network
from sim_soens.super_library import NeuralZoo
from sim_soens.super_node import SuperNode

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
        # print(neurons[0].neuron.dend_soma.dendritic_inputs['n0_lay0_branch0_den1'].dendritic_inputs['n0_lay1_branch1_den0'].synaptic_inputs['3'].name)

    def make_net(self):
        self.layer_n = 3
        branches = 3
        for i in range(1,self.layer_n+1):
            # print(self.neurons[i].synapse_list)
            for j in range(branches):
                self.neurons[i].neuron.add_output(self.neurons[0].synapse_list[(j*3)+(i-1)])
                # print(self.neurons[0].neuron.name, i, (j*3)+(i-1), self.neurons[0].synapse_list[(j*3)+(i-1)].name)
        # print(self.neurons[0].neuron.dend_soma.dendritic_inputs['n0_lay0_branch0_den1'].dendritic_inputs['n0_lay1_branch1_den0'].synaptic_inputs['3'].__dict__)
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
        w_sd = 1
        syn_struct = [
            [[[w_sd]]],
            [[[w_sd]]],
            [[[w_sd]]],
            [[[w_sd]]],
            [[[w_sd]]],
            [[[w_sd]]],
            [[[w_sd]]],
            [[[w_sd]]],
            [[[w_sd]]],
            [[[w_sd]]],
        ]
        
        for i in range(self.N):
            # neuron = NeuralZoo(type='custom',name=f'res_neuron_{i}',synaptic_structure=syn_struct,seed=self.run*1000+i,**params) # name=f'res_neuron_{i}',
            neuron = SuperNode(name=f'res_neuron_{i}',synaptic_structure=syn_struct,seed=self.run*1000+i,**params)
            # neuron.synaptic_layer()
            self.neurons.append(neuron)

    def make_net(self):
        self.connectivity = []
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
                                self.connectivity.append([i,j])
                                # syn.connection_strength = np.random.rand()
                                connections+=1
                                break
        # print("Reservoir connections: ", connections)

    def connect_input(self,input):
        connections = 0
        syn_finder = 0
        self.input_connectivity = []
        self.input_channels = len(input.spike_rows)
        
        for repeat in range(self.laps):
            for i,row in enumerate(input.spike_rows):
                # print((len(input.spike_rows)*repeat+i)%72)
                # j = (len(input.spike_rows)*repeat+i)%72
                count=0
                for j,syn in enumerate(self.neurons[(len(input.spike_rows)*repeat+i)%self.N].synapse_list):
                    if "synaptic_input" not in syn.__dict__:
                        self.input_connectivity.append([i,j])
                        array = np.sort(row)
                        array = np.append(array,np.max(array)+.001)
                        syn.add_input(input_signal(name = 'input_synaptic_drive', 
                                            input_temporal_form = 'arbitrary_spike_train', 
                                            spike_times = array) )
                        count+=1
                        connections += 1 
                        break

    def graph_input(self):
        import networkx as nx
        from networkx.algorithms import bipartite
        import matplotlib.pyplot as plt
        self.input_connectivity = []
        G = nx.Graph()
        keys_in = np.arange(0,self.input_channels,1)
        keys_res = np.arange(self.input_channels,self.input_channels+self.N,1)
        ki = []
        kr = []
        for i in keys_in:
            ki.append(str(i))
        for j in keys_in:
            kr.append(str(j))
        G.add_nodes_from(keys_in, bipartite=0)
        G.add_nodes_from(keys_res,bipartite=1)

        add_edges = []
        print(self.input_connectivity)
        for ii, connect in self.input_connectivity:
            print(ii)
            add_edges.append( (str(connect[ii][0]),str(connect[ii][1]+self.input_channels)))
            # G.add_edge(connect[0],connect[1]+self.input_channels)
        self.edges=add_edges    
        G.add_edges_from(add_edges)
        bipartite.is_bipartite(G)

        nx.draw_networkx(G, pos = nx.drawing.layout.bipartite_layout(G, keys_in), width = 2)
        plt.show()




    def graph_net(self):

        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        G.add_nodes_from(range(self.N))
        for connect in self.connectivity:
            G.add_edge(connect[0],connect[1],with_labels=True)

        # print(len(G.edges()))
        # print(G.degree())
        # print(max(list(zip(*G.degree()))[1]))
        # print(np.mean(list(zip(*G.degree()))[1]))
        plt.figure(figsize=(14,14))
        nx.draw_circular(G, with_labels=True)
        plt.show()




    def run_network(self,prune_synapses=True,backend='Python'):
        self.net = network(dt=self.dt,tf=self.tf,nodes=self.neurons,new_way=False,backend=backend)
        self.net.null_synapses = prune_synapses
        print("running network")
        self.net.simulate()


class SuperNet:
    def __init__(self,**params):
        self.N  = 10
        self.dt = 0.1
        self.tf = 500
        self.__dict__.update(params)
        self.make_nodes()
        self.connect()
        

    def make_nodes(self):
        self.nodes = []
        count = 0
        for i,quantity in enumerate(self.node_quantities):
            for ii in range(quantity):
                self.nodes.append(SuperNode(net_idx=count,**self.node_params[i]))
                count += 1

    def connect(self):
        np.random.seed(None)
        if hasattr(self,'prob_connect'):
            self.connectivity = []
            for i in range(self.N):
                for j in range(self.N):
                    if np.random.rand() <= self.prob_connect and i!=j:
                        n1 = self.nodes[i]
                        n2 = self.nodes[j]
                        self.rand_recursive_connect(n1,n2)


        elif hasattr(self,'connectivity'):
            for connect in self.connectivity:
                i = connect[0]
                j = connect[1]
                n1 = self.nodes[i]
                n2 = self.nodes[j]
                self.rand_recursive_connect(n1,n2)

        print(f"\nInternal network connections = {len(self.connectivity)}")

    def input_connect(self,input,prob_input=None,in_connectivity=None):

        if prob_input != None:
            self.in_connectivity = []
            for i in range(input.channels):
                for j in range(self.N):
                    if np.random.rand() <= prob_input:
                        n1 = input.signals[i]
                        n1.net_idx = i
                        n2 = self.nodes[j]
                        self.rand_recursive_connect(n1,n2)


        elif in_connectivity != None:
            for connect in self.in_connectivity:
                i = connect[0]
                j = connect[1]
                n1 = input.signals[i]
                n1.net_idx = i
                n2 = self.nodes[j]
                self.rand_recursive_connect(n1,n2)

        print(f"\nInput network connections = {len(self.in_connectivity)}")

                        
    def rand_recursive_connect(self,n1,n2):
        available_syn=0 
        for syn in n2.synapse_list:
            if "synaptic_input" not in syn.__dict__:
                available_syn+=1

        if available_syn > 0:

            rand_int = len(n2.synapse_list)
            syn_idx = np.random.randint(rand_int)
            syn = n2.synapse_list[syn_idx]

            if "synaptic_input" not in syn.__dict__:
                
                if type(n1).__name__ == 'input_signal':
                    syn.add_input(n1)
                    self.in_connectivity.append([n1.net_idx,n2.net_idx])
                else:
                    n1.neuron.add_output(syn)
                    self.connectivity.append([n1.net_idx,n2.net_idx])
            else:
                self.rand_recursive_connect(n1,n2)
        else:
            return

    def graph_net(self):

        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        G.add_nodes_from(range(self.N))
        for connect in self.connectivity:
            G.add_edge(connect[0],connect[1],with_labels=True)

        plt.figure(figsize=(10,10))
        nx.draw_circular(G, with_labels=True)
        plt.show()

    def run_network(self,backend='python'):
        print(f"Running {backend} network")
        self.net = network(dt=self.dt,tf=self.tf,nodes=self.nodes,backend=backend)
        self.net.null_synapses = True
        print("\nrunning network")
        self.net.simulate()

    def raster_plot(self):
        from sim_soens.soen_plotting import raster_plot
        raster_plot(self.net.spikes)




class __SuperNet:
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
        self.input_connectivity = []
        ### change for more complex input
        if self.in_connect == "random":
            p = self.input_p
            for i in range(len(self.inputs)):
                for j in range(self.N):
                    rnd = np.random.rand() 
                    if rnd < p:
                        # print(i,j)
                        self.dendrites[j].add_input(self.synapse_in[i], connection_strength = self.W_SID[j])
                        self.input_connectivity.append([i,j])

        elif self.in_connect == "ordered":
            for i in range(len(self.synapse_in)):
                if i < self.N:
                    self.dendrites[i].add_input(self.synapse_in[i], connection_strength = self.W_SID[i])
                else:
                    self.dendrites[i-self.N].add_input(self.synapse_in[i], connection_strength = self.W_SID[i-self.N])
                self.input_connectivity.append([i,j])
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

            s = self.net.neurons[neuron_key].dend_soma.s
            S.append(s)

            phi_r = self.net.neurons[neuron_key].dend_soma.phi_r
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


    