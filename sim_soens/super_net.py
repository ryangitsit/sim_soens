import numpy as np

from sim_soens.soen_components import network
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


class SuperNet:
    '''
    For generating custom networks both in terms of topology and neuron typology
        - Like `SuperNode`, `SuperNet` is a wrapper class that facilitates custom design (now of networks)
        - Pass in network params and node params to quickly generate a custom network of custom neurons
    '''
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

