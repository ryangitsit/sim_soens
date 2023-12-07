import numpy as np

from sim_soens.soen_components import input_signal, network
from sim_soens.neuron_library import NeuralZoo
from sim_soens.super_node import SuperNode

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

    def make_net(self):
        self.layer_n = 3
        branches = 3
        for i in range(1,self.layer_n+1):
            # print(self.neurons[i].synapse_list)
            for j in range(branches):
                self.neurons[i].neuron.add_output(self.neurons[0].synapse_list[(j*3)+(i-1)])


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
        self.w_coeff = 0.6

        # custom input params
        self.__dict__.update(params)

        np.random.seed(self.run)
        
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
        syn_struct = [ [[[np.random.rand()*self.w_coeff]]] for _ in range(10)]
        
        for i in range(self.N):
            neuron = SuperNode(
                name=f'res_neuron_{i}',synaptic_structure=syn_struct,seed=self.run*1000+i,**params
                )
            # neuron.synaptic_layer()
            self.neurons.append(neuron)

    def make_net(self):
        self.connectivity = []
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

    def run_network(self,prune_synapses=True,backend='julia'):
        self.net = network(
            dt=self.dt,tf=self.tf,nodes=self.neurons,new_way=False,backend=backend,jul_threading=4
            )
        self.net.null_synapses = prune_synapses
        print("running network")
        self.net.simulate()

