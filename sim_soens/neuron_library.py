import numpy as np
from sim_soens.soen_components import neuron, dendrite, synapse
from sim_soens.super_node import SuperNode

'''
Here a class for calling from a 'zoo' of possible neurons is implemented.

Plan:
 - Syntax for custom neuron calls based on dendritic structures and parameters.
 - Library of predefined neurons, both bio-inspired and engineering-specific
 - Should include a testing/plotting paradigm inherent in the class
 - Add more explicit connectivity defintions and corresponding plotting
'''


class MNISTNode(SuperNode):
    '''
    NeuralZoo object class
     - offers a variety of archetypal neurons ready-to-implement
    '''
    def __init__(self,**entries):
        super().__init__()

        self.ib            = 1.8
        self.tau           = 50
        self.beta          = 3.0
        self.s_th          = 0.25
        self.eta           = 0.005
        self.elast         = "None"
        self.valid         = "True"
        self.exp_name      = "test"
        self.inhibit       = -1
        self.backend       = "julia"
        self.run           =  1
        self.digits        = 10
        self.samples       = 50
        self.elasticity    = None
        self.layers        = 6
        self.decay         = "False"
        self.probabilistic = 1
        self.weights       = "preset"
        self.dataset       = "MNIST"
        self.duration      = 2500
        self.low_bound     = -0.5
        self.plotting      = "sparse"
        self.jul_threading = 4
        self.hebbian       = "False"
        self.exin          = None
        self.fixed         = 0.5
        self.rand_flux     = 0.005
        self.inh_counter   = None
        self.norm_fanin    = True
        self.lay_weighting = [1, 1, 1, 4, 8, 10]
        self.fan_coeff     = 1.5
        self.dt            = 1.0
        self.target        = 50
        self.max_offset    = 0.5

        self.__dict__.update(entries)
        self.params = self.__dict__
        print(f"Building {self.name} [>       ]", end="\r")

        if 'loaded_weights' not in list(entries.keys()):
            self.build_arbor()
            print(f"Building {self.name} [=>      ]", end="\r")
        else:
            self.build_arbor()
            self.weights = entries['loaded_weights']
            print(f"Loading W {self.name} [=>      ]", end="\r")
        # create a neuron object given init params

        self.neuron = neuron(**self.params)
        self.neuron.dend_soma.branch=0

        # add somatic dendrite (dend_soma) and refractory dendrite to list
        self.dendrite_list = [self.neuron.dend_soma,self.neuron.dend__ref]

        # check that the structure implied by .weights is compatible with construction method
        self.check_arbor_structor(self.weights)
        print(f"Building {self.name} [==>     ]", end="\r")
        self.make_dendrites()
        self.connect_dendrites()
        print(f"Building {self.name} [===>    ]", end="\r")
        if self.rand_flux is not None:
            print("Random Flux")
            self.random_flux(self.rand_flux)
            print(f"Building {self.name} [====>   ]", end="\r")
            
        if self.inh_counter==True:
            print("Inhibition counter")
            # self.add_inhibition_counts()
            print(f"Building {self.name} [=====>  ]", end="\r")
        if self.norm_fanin:
            print(f"Fanin normalization with coefficient of {self.fan_coeff}")
            self.normalize_fanin(self.fan_coeff)
            print(f"Building {self.name} [======> ]", end="\r")
        elif self.norm_fanin_prime:

            print(f"Symmetric fanin normalization with buffer of {self.fan_buffer} and coeff of {self.fan_coeff}")

            self.normalize_fanin_symmetric(buffer=self.fan_buffer,coeff=self.fan_coeff)


        self.make_and_connect_synapses()
        print(f"Building {self.name} [=======>]", end="\r")

    def build_arbor(self):
        if self.layers == 6:
            lw = np.array(self.lay_weighting)

            # create random weights for each layer
            layer_1 = np.array(
                [self.make_weights(7,self.exin,self.fixed)])*lw[0]*.5
            
            layer_2 = np.array(
                [self.make_weights(7,self.exin,self.fixed)*lw[1] for _ in range(int(49/7))])
            
            layer_3 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[2] for _ in range(int(98/2))])
            
            layer_4 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[3] for _ in range(int(196/2))])
            
            layer_5 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[4] for _ in range(int(392/2))])
            
            layer_6 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[5] for _ in range(int(784/2))])

            # place them in a weight structure (defines structure and weighing of a neuron)
            self.weights = [
                layer_1,
                layer_2,
                layer_3,
                layer_4,
                layer_5,
                layer_6,
            ]

            # for i,w in enumerate(self.weights):
            #     print(f"Layer {i} has shape {w.shape}")

        elif self.layers == 5:
            print("Tiling Technique")
            lw = np.array(self.lay_weighting)

            # create random weights for each layer
            layer_1 = np.array(
                [self.make_weights(7,self.exin,self.fixed)])*lw[0]*.5
            
            layer_2 = np.array(
                [self.make_weights(7,self.exin,self.fixed)*lw[1] for _ in range(int(49/7))])
            
            layer_3 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[2] for _ in range(int(98/2))])
            
            layer_4 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[3] for _ in range(int(196/2))])
            
            layer_5 = np.array(
                [self.make_weights(4,self.exin,self.fixed)*lw[4] for _ in range(int(784/4))])

            # place them in a weight structure (defines structure and weighing of a neuron)
            self.weights = [
                layer_1,
                layer_2,
                layer_3,
                layer_4,
                layer_5,
            ]

            for i,w in enumerate(self.weights):
                print(f"Layer {i} has shape {w.shape}")

        elif self.layers == 9:
            print("Deep Tree")
            lw = np.array(self.lay_weighting)

            # create random weights for each layer
            layer_1 = np.array(
                [self.make_weights(4,self.exin,self.fixed)])*lw[0]*.5

            layer_2 = [
                self.make_weights(2,self.exin,self.fixed)*lw[1] for _ in range(int(6/2))]
            layer_2.append(self.make_weights(1,self.exin,self.fixed)*lw[1])
                    
            layer_3 = [
                self.make_weights(2,self.exin,self.fixed)*lw[1] for _ in range(int(12/2))]
            layer_3.append(self.make_weights(1,self.exin,self.fixed)*lw[1])

            layer_4 = [
                self.make_weights(2,self.exin,self.fixed)*lw[1] for _ in range(int(24/2))]
            layer_4.append(self.make_weights(1,self.exin,self.fixed)*lw[1])

            layer_5 = [
                self.make_weights(2,self.exin,self.fixed)*lw[1] for _ in range(int(48/2))]
            layer_5.append(self.make_weights(1,self.exin,self.fixed)*lw[1])
            
            layer_6 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[2] for _ in range(int(98/2))])
            
            layer_7 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[3] for _ in range(int(196/2))])
            
            layer_8 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[4] for _ in range(int(392/2))])
            
            layer_9 = np.array(
                [self.make_weights(2,self.exin,self.fixed)*lw[4] for _ in range(int(784/2))])

            # place them in a weight structure (defines structure and weighing of a neuron)
            self.weights = [
                layer_1,
                layer_2,
                layer_3,
                layer_4,
                layer_5,
                layer_6,
                layer_7,
                layer_8,
                layer_9
            ]

            for i,w in enumerate(self.weights):
                shp = []
                for group in w:
                    shp.append(len(group))
                # print(f"Layer {i} has shape {len(shp),sum(shp)}")  
                    

class SpecificNode(SuperNode):
    '''
    NeuralZoo object class
     - offers a variety of archetypal neurons ready-to-implement
    '''
    def __init__(self,**entries):
        super().__init__()

        self.ib            = 1.8
        self.tau           = 50
        self.beta          = 3.0
        self.s_th          = 0.25
        self.eta           = 0.005
        self.elast         = "None"
        self.valid         = "True"
        self.exp_name      = "test"
        self.inhibit       = -1
        self.backend       = "julia"
        self.run           =  1
        self.digits        = 10
        self.samples       = 50
        self.elasticity    = None
        self.layers        = 6
        self.decay         = "False"
        self.probabilistic = 1
        self.weights       = "preset"
        self.dataset       = "MNIST"
        self.duration      = 2500
        self.low_bound     = -0.5
        self.plotting      = "sparse"
        self.jul_threading = 4
        self.hebbian       = "False"
        self.exin          = None
        self.fixed         = 0.5
        self.rand_flux     = 0.005
        self.inh_counter   = None
        self.norm_fanin    = True
        self.lay_weighting = [1, 1, 1, 4, 8, 10]
        self.fan_coeff     = 1.5
        self.dt            = 1.0
        self.target        = 50
        self.max_offset    = 0.5

        self.__dict__.update(entries)
        self.params = self.__dict__

        self.build_arbor()

        # create a neuron object given init params
        self.neuron = neuron(**self.params)
        self.neuron.dend_soma.branch=0

        # add somatic dendrite (dend_soma) and refractory dendrite to list
        self.dendrite_list = [self.neuron.dend_soma,self.neuron.dend__ref]

        # check that the structure implied by .weights is compatible with construction method
        self.check_arbor_structor(self.weights)
                        
        self.make_dendrites()
        self.connect_dendrites()

        self.random_flux(self.rand_flux)
        self.add_inhibition_counts()
        print(f"Fanin normalization with coefficient of {self.fan_coeff}")
        self.normalize_fanin(self.fan_coeff)

        self.make_and_connect_synapses()

    def build_arbor(self):
        self.weights = [[[.5,.7]],
                    [[.3,.3],[.4,.5]]]

class NeuralZoo():
    '''
    NeuralZoo object class
     - offers a variety of archetypal neurons ready-to-implement
    '''
    def __init__(self,**entries):
        
        self.random_syn = False
        self.__dict__.update(entries)
        self.params = self.__dict__


        if self.type == 'custom':
            self.custom()

        elif self.type == '3fractal':
            self.fractal_three()

        elif self.type == 'single':
            self.single()

        elif self.type == 'mono_point':
            self.mono_point()

        elif self.type == 'mono_dendrite':
            self.mono_dend()

        elif self.type == 'mono_dend_soma':
            self.mono_dend_soma()

        elif self.type == 'self_feed':
            self.self_feed()

        elif self.type == 'mono_plus_minus':
            self.mono_plus_minus()

        elif self.type == 'double_ref':
            self.double_ref()

        elif self.type == 'point_3ex_1in':
            self.point_3ex_1in()

        elif self.type == 'asym_plus_minus':
                self.asym_plus_minus()

        elif self.type == 'denex3_denin1':
                self.denex3_denin1()

        elif self.type == 'proximal_basal':
                self.proximal_basal()

        elif self.type == 'plastic_neuron':
                self.plastic_neuron()
    
        self.refractory_synapse = self.neuron.dend__ref.synaptic_inputs[f"{self.name}__syn_refraction"]


    def custom(self):
        '''
        LEGACY FUNCTION REPLACED WITH SUPERNODE
        '''    

        # give neuron name if not already assigned
        if "name" not in self.params:
            self.params['name'] = f"rand_neuron_{int(np.random.rand()*100000)}"

        # create a neuron object given init params
        self.neuron = neuron(**self.params)

        # add somatic dendrite (dend_soma) and refractory dendrite to list
        self.dendrite_list = [self.neuron.dend_soma,self.neuron.dend__ref]

        # normalize input to soma to 1 in terms of weighting
        self.neuron.normalize_input_connection_strengths=1

        # default random seed
        np.random.seed(None)

        # for systematic seeding of multi-run experiments
        if hasattr(self, 'seed'):
            np.random.seed(self.seed)
            # print("random seed: ",self.seed)


        # weights defines structure implicitly and defines connection strengths
        if hasattr(self, 'weights'):
            arbor = self.weights
        else:
            arbor = []

        self.check_arbor_structor(arbor)
                        
        # dendrites attribute will have some structure as arbor
        # [layer][group][dendrite]
        # populated with dendrite objects
        dendrites = [ [] for _ in range(len(arbor)) ]
        if len(arbor)>0:
            count=0
            den_count = 0
            for i,layer in enumerate(arbor):
                c=0
                for j,dens in enumerate(layer):
                    sub = []
                    for k,d in enumerate(dens):
                        #** add flags and auto connects for empty connections

                        # parameters for creating current dendrite
                        dend_params = self.params


                        # check for any dendrite-specific parameters
                        # if so, use in dend_parameters
                        # otherwise, one of the following will be used
                        #   - default parameters (defined in dendrite class)
                        #   - general dendrite parameters defined in this node's
                        #     initialization 
                        if hasattr(self, 'betas'):
                            dend_params["beta_di"] =(np.pi*2)*10**self.betas[i][j][k]
                        if hasattr(self, 'biases'):
                            if hasattr(self, 'types'):
                                if self.types[i][j][k] == 'ri':
                                    dend_params["ib"] = self.ib_list_ri[self.biases[i][j][k]]
                                else:
                                    dend_params["ib"] = self.ib_list_rtti[self.biases[i][j][k]]
                            else:
                                dend_params["ib"] = self.ib_list_ri[self.biases[i][j][k]]
                        if hasattr(self, 'taus'):
                            dend_params["tau_di"] = self.taus[i][j][k]
                        if hasattr(self, 'types'):
                            dend_params["loops_present"] = self.types[i][j][k]
                            # print("HERE",self.types[i][j][k])
                        else:
                            dend_params["loops_present"] = 'ri'

                        # self.params = self.__dict__
                        dend_params["dend_name"] = f"{self.neuron.name}_lay{i+1}_branch{j}_den{k}"
                        dend_params["type"] = type

                        # generate a dendrite given parameters
                        dend = dendrite(**dend_params)

                        # add it to group
                        sub.append(dend)

                        # add it to node's dendrite list
                        self.dendrite_list.append(dend)
                        den_count+=1
                        c+=1

                        # keep track of origin branch
                        if i==0:
                            dend.branch=k
                    
                    # add group to layer
                    dendrites[i].append(sub)
        
            # iterate over dendrites and connect them as defined by structure
            for i,l in enumerate(dendrites):
                for j, subgroup in enumerate(l):
                    for k,d in enumerate(subgroup):
                        if i==0:
                            # print(i,j,k, " --> soma")
                            self.neuron.add_input(d, 
                                connection_strength=self.weights[i][j][k])
                            # self.neuron.add_input(d, 
                            #     connection_strength=self.w_dn)
                        else:
                            # print(i,j,k, " --> ", i-1,0,j)
                            receiving_dend = np.concatenate(dendrites[i-1])[j]
                            receiving_dend.add_input(d, 
                                connection_strength=self.weights[i][j][k])
                            d.branch = receiving_dend.branch
                        d.output_connection_strength = self.weights[i][j][k]

        # add the somatic dendrite to the 0th layer of the arboric structure
        dendrites.insert(0,[[self.neuron.dend_soma]])

        # if syns attribute, connect as a function of grouping to final layer
        if hasattr(self, 'syns'):
            self.synapses = [[] for _ in range(len(self.syns))]
            for i,group in enumerate(self.syns):
                for j,s in enumerate(group):
                    self.synapses[i].append(synapse(name=s))
            count=0
            for j, subgroup in enumerate(dendrites[len(dendrites)-1]):
                for k,d in enumerate(subgroup):
                    for s in self.synapses[count]:
                        dendrites[len(dendrites)-1][j][k].add_input(s, 
                            connection_strength = self.syn_w[j][k])
                    count+=1

        # if synaptic_structure, connect synapses to specified dendrites
        # synaptic_sructure has form [synapse][layer][group][denrite]
        # thus, there is an entire arbor-shaped structure for each synapse
        # the value at an given index specifies connection strength
        elif hasattr(self, 'synaptic_structure'):
            
            # for easier access later
            self.synapse_list = []

            # synaptic_structure shaped list of actual synapse objects
            self.synapses = [[] for _ in range(len(self.synaptic_structure))]

            # iterate over each arbor-morphic structure
            for ii,S in enumerate(self.synaptic_structure):

                # make a synapse
                syn = synapse(name=f'{self.neuron.name[-2:]}_syn{ii}')

                # append to easy-access list
                self.synapse_list.append(syn)

                # new arbor-morphic list to be filled with synapses
                syns = [[] for _ in range(len(S))]

                # whereever there is a value in syn_struct, put synapse there
                for i,layer in enumerate(S):
                    syns[i] = [[] for _ in range(len(S[i]))]
                    for j,group in enumerate(layer):
                        for k,s in enumerate(group):
                            if s != 0:
                                # print('synapse')
                                syns[i][j].append(syn)
                            else:
                                # print('no synapse')
                                syns[i][j].append(0)
                self.synapses[ii]=syns
            # print('synapses:', self.synapses)

            # itereate over new synapse-filled list of arbor-structures
            # add synaptic input to given arbor elements
            for ii,S in enumerate(self.synapses):
                for i,layer in enumerate(S):
                    for j, subgroup in enumerate(layer):
                        for k,d in enumerate(subgroup):
                            s=S[i][j][k]
                            if s !=0:
                                if self.random_syn == False:
                                    connect = self.synaptic_structure[ii][i][j][k]
                                elif self.random_syn == True:
                                    connect = np.random.rand()
                                dendrites[i][j][k].add_input(s, 
                                    connection_strength = connect)

        # make dendrites readable through node object
        if dendrites:
            self.dendrites = dendrites

    ############################################################################
    #                           input functions                                #
    ############################################################################  

    def synaptic_layer(self):
        '''
        Simply adds a synapse to all dendrites on the outer layer
        '''
        self.synapse_list = []
        count = 0
        if hasattr(self,'w_sd'):
            w_sd = self.w_sd
        else:
            w_sd = 1
        for g in self.dendrites[len(self.dendrites)-1]:
            for d in g:
                syn = synapse(name=f'{self.neuron.name}_syn{count}')
                self.synapse_list.append(syn)
                count+=1
                d.add_input(syn,connection_strength=w_sd)

    def uniform_input(self,input):
        '''
        Add the same input channel to all available synapses
        '''
        for S in self.synapse_list:
            S.add_input(input.signals[0])

    def custom_input(self,input,connections):
        '''
        Add the same input channel to specific synapses
         - Simply defined as list of indice tuples
        '''
        for connect in connections:
            self.synapses_list[connect].add_input(input.signals[0])
                            
    def multi_channel_input(self,input,connectivity=None):
        '''
        Add specific input channels to specific synapses
        '''
        for connect in connectivity:
            # print(connect[0],connect[1])
            self.synapse_list[connect[0]].add_input(input.signals[connect[1]])


    ############################################################################
    #                           premade neurons                                #
    ############################################################################  

    # def single(self):

    #     self.synapse = synapse(name=1)

    #     self.dendrite = common_dendrite(1, 'ri', self.beta_di, 
    #                                       self.tau_di, self.ib)
                                    
    #     self.dendrite.add_input(self.synapse, connection_strength = self.w_sd)

    #     self.neuron = common_neuron(1, 'ri', self.beta_ni, self.tau_ni, 
    #                                   self.ib_n, self.s_th, 
    #                                   self.beta_ref, self.tau_ref, self.ib_ref)

    #     self.neuron.add_input(self.dendrite, connection_strength = self.w_dn)


    # def fractal_three(self):
    #     H = 3 # depth
    #     n = [3,3] # fanning at each layer, (length = H-1), from soma to synapses

    #     fractal_neuron = common_neuron(1, 'ri', self.beta_ni, self.tau_ni, 
    #                                    self.ib_n, self.s_th, 
    #                                    self.beta_ref, self.tau_ref, self.ib_ref)
    #     # fractal_neuron.name = 'name'
    #     dendrites = [ [] for _ in range(H-1) ]
    #     synapses = []

    #     count=0
    #     count_syn=0
    #     last_layer = 1
    #     # returns dendrites[layer][dendrite] = dendrites[H-1][n_h]
    #     for h in range(H-1): 
    #         for d in range(n[h]*last_layer):
    #             dendrites[h].append(common_dendrite(count, 'ri', self.beta_di, 
    #                                 self.tau_di, self.ib))

    #             if h == H-2:
    #                 synapses.append(synapse(name=d))
    #                 dendrites[h][d].add_input(synapses[d], 
    #                                           connection_strength = self.w_sd)
    #             count+=1
    #         last_layer = n[h]

    #     for i,layer in enumerate(dendrites):
    #         # print("layer:", i)
    #         for j,d in enumerate(layer):
    #             # print("  dendrite", j)
    #             if i < H-2:
    #                 for g in range(n[1]):
    #                     d.add_input(dendrites[i+1][j*n[1]+g], 
    #                                 connection_strength=self.w_dd)
    #                     # print(j,j*n[1]+g)
    #                 fractal_neuron.add_input(d, connection_strength=self.w_dn)
    #     self.dendrites = dendrites
    #     self.synapses = synapses
    #     self.fractal_neuron = fractal_neuron


    def plastic_neuron(self):
        '''
        Plasticity equipped neuron
        '''
        # print(self.weights)
        self.custom()
        self.synaptic_layer()
        input_obj = self.input_obj

        if len(input_obj.signals) == 2:
            # print("Multi-signal")
            self.synapse_list[0].add_input(input_obj.signals[0])
            self.synapse_list[1].add_input(input_obj.signals[1])
        else:
            self.synapse_list[0].add_input(input_obj.signals[0])
            self.synapse_list[1].add_input(input_obj.signals[0])

        exin = ["plus","minus"]
        self.trace_dendrites = []
        for lay in self.dendrites[1:]:
            for group in lay:
                for i,d in enumerate(group):
                    cs = self.weights[0][0][i]*self.trace_factor
                    # print(cs)
                    for ei in exin:
                        trace_dend = dendrite(
                            name=f'n{self.n_count}_d{i}_{ei}',
                            tau_di=self.trace_tau
                            )
                        trace_dend.add_input(d,connection_strength=cs)
                        rnd = int(np.random.rand()*100000)
                        syn = synapse(
                            name=f'{d.name}_tracesyn_{trace_dend.name}_{rnd}'
                            )
                        trace_dend.add_input(
                            syn,
                            connection_strength=self.trace_syn_factor
                            )
                        self.trace_dendrites.append(trace_dend)
                        self.dendrite_list.append(trace_dend)
                        self.synapse_list.append(syn)
        # self.neuron = p_neuron
        return self

    def mono_point(self):
        '''
        Monosynaptic Point Neuron
        '''
        self.synaptic_structure = [[[[1]]]]
        mono_p = self.custom()
        return mono_p

    def mono_dend(self,w=0.5):
        '''
        Monosynaptic Neuron with intermediate dendrite
        '''
        if hasattr(self, 'w_dn'):
            w = self.w_dn
        self.synaptic_structure = [[[[0]],[[1]]]]
        self.weights = [[[w]]]
        mono_d = self.custom()
        
        return mono_d

    def mono_dend_soma(self):
        '''
        Monosynaptic Neuron with intermediate dendrite and skip connection to
        soma
        '''
        self.synaptic_structure = [[[[1]],[[1]]]]
        self.weights = [[[.1]]]
        mono_dend_soma = self.custom()
        return mono_dend_soma

    def self_feed(self):
        '''
        Monosynaptic Neuron with intermediate self-feeding dendrite
        '''
        self.synaptic_structure = [[[[0]],[[1]]]]
        self.weights = [[[.5]]]
        mono_dend_soma = self.custom()
        self.dendrites[1][0][0].self_feedback_coupling_strength = .75
        return mono_dend_soma

    def mono_plus_minus(self):
        '''
        Monosynaptic Neuron where synapse feeds one excitatory dendrite, one 
        inhibitory, and the soma directly
        '''
        self.synaptic_structure = [[[[1]],[[.8,.9]]]]
        self.weights = [[[-.4,.2]]]
        neuron = self.custom()
        return neuron

    def double_ref(self):
        '''
        Monosynaptic Neuron with intermediate dendrite and extra refractory dend
        **Not working, need to edit time stepper
        '''
        self.synaptic_structure = [[[[0]],[[1]]]]
        self.weights = [[[.5]]]
        # self.second_ref=True
        neuron = self.custom()
        # ref2 = dendrite(**params)
        # self.neuron.add_input(ref2, connection_strength=-.67)
        # ref2.add_input(self.neuron.dend_soma, connection_strength=1)
        # self.second_ref = ref2
        return neuron

    def point_3ex_1in(self):
        '''
        Three excitatory synapses and one inhibitory synapse, all feeding soma
        '''
        self.synaptic_structure = [[[[1]]],[[[1]]],[[[1]]],[[[-1]]]]
        neuron = self.custom()
        return neuron


    def asym_plus_minus(self):
        '''
        Three excitatory synapses and one inhibitory synapse, all feeding one
        intermediate dendrite.  Another dendrite is fed with one inhibitory
        synapse
        '''
        self.synaptic_structure = [
            [
                [[0]],
                [[1,0]]
            ],
            [
                [[0]],
                [[1,0]]
            ],
            [
                [[0]],
                [[1,0]]
            ],
            [
                [[0]],
                [[-1,0]]
            ],
            [
                [[0]],
                [[0,-1]]
            ],
        ]
        self.weights = [[[.8,-.4]]]
        neuron = self.custom()
        return neuron

    def denex3_denin1(self):
        '''
        x3 on E/I dens above
        '''
        self.synaptic_structure = [
            [
                [[0]],
                [[1,0,0,0]]
            ],
            [
                [[0]],
                [[1,0,0,0]]
            ],
            [
                [[0]],
                [[1,0,0,0]]
            ],
            [
                [[0]],
                [[-1,0,0,0]]
            ],
            [
                [[0]],
                [[0,1,0,0]]
            ],
            [
                [[0]],
                [[0,1,0,0]]
            ],
            [
                [[0]],
                [[0,1,0,0]]
            ],
            [
                [[0]],
                [[0,-1,0,0]]
            ],
            [
                [[0]],
                [[0,0,1,0]]
            ],
            [
                [[0]],
                [[0,0,1,0]]
            ],
            [
                [[0]],
                [[0,0,1]]
            ],
            [
                [[0]],
                [[0,0,-1,0]]
            ],
            [
                [[0]],
                [[0,0,0,-1]]
            ],
        ]
        self.weights = [[[.2,.25,.3,-.25]]]
        neuron = self.custom()
        return neuron

    def proximal_basal(self):
        '''
        One 3/1-E/I dendrite feeds another 3/1-E/I dendrite and soma.  Latter 
        dendrite only feeds soma. Third denrite has inihibitory synapse only.
        '''
        self.synaptic_structure = [
            [
                [[0]],
                [[1,0,0]]
            ],
            [
                [[0]],
                [[1,0,0]]
            ],
            [
                [[0]],
                [[1,0,0]]
            ],
            [
                [[0]],
                [[-1,0,0]]
            ],
            [
                [[0]],
                [[0,1,0]]
            ],
            [
                [[0]],
                [[0,1,0]]
            ],
            [
                [[0]],
                [[0,1,0]]
            ],
            [
                [[0]],
                [[0,-1,0]]
            ],
            [
                [[0]],
                [[0,0,-1]]
            ],
        ]
        self.weights = [[[.5,.5,-.25]]]
        neuron = self.custom()
        self.dendrites[1][0][1].add_input(
            self.dendrites[1][0][0], 
            connection_strength=.5
            )
        return neuron



    ############################################################################
    #                           helper functions                               #
    ############################################################################  

    def parameter_print(self):
        print("\nSOMA:")
        print(f" ib = {self.neuron.ib}")
        print(f" ib_n = {self.neuron.ib_n}")
        print(f" tau_ni = {self.neuron.tau_ni}")
        print(f" beta_ni = {self.neuron.beta_ni}")
        # print(f" tau = {self.neuron.tau}")
        print(f" loops_present = {self.neuron.loops_present}")
        print(f" s_th = {self.neuron.s_th}")
        print("\n")
        # print(f" ib_di = {self.neuron.ib_di}")
        # print(f" tau_di = {self.neuron.tau_di}")
        # print(f" beta_di = {self.neuron.beta_di}")

        print("\nREFRACTION:")
        print(f" ib_ref = {self.neuron.ib_ref}")
        print(f" tau_ref = {self.neuron.tau_ref}")
        print(f" beta_ref = {self.neuron.beta_ref}")

        print("\nDENDRITES:")
        for dend in self.dendrite_list:
            name = "arbor"
            if "ref" in dend.name:
                name = 'refractory'
            elif "soma" in dend.name:
                name = "soma"
            print(f" {name}", dend.name)
            print(f"   ib_di = {dend.ib}")
            print(f"   tau_di = {dend.tau_di}")
            print(f"   beta_di = {dend.beta}")
            print(f"   loops_present = {dend.loops_present}")
        print("\n\n")

        print("\nCONNECTIVITY:")

    def check_arbor_structor(self,arbor):
        for i,layer in enumerate(arbor):
            for j,dens in enumerate(layer):
                for k,d in enumerate(dens):
                    if i == 0:
                        if len(layer) != 1:
                            print('''
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            ARBOR ERROR: First layer may only have one group.
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            ''')
                    else:
                        if len(layer) != len(np.concatenate(arbor[i-1])):
                            print(f'''
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            ARBOR ERROR: Groups in layer {i} must be equal to total dendrites in layer {i-1}
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            ''')
                            return

    ############################################################################
    #                  plotting (to be moved --> soen_plotting)                #
    ############################################################################  
    def plot_neuron_activity(self,net,**kwargs):
        '''
        Plots signal activity for a given neuron
         - syntax     -> NeuralZoo.plot_neuron_activity(net,spikes=True,input=input)
         - kwargs:
            - net        -> network within which neurons were simulated
            - phir       -> plot phi_r of soma and phi_r thresholds
            - dend       -> plot dendritic signals
            - input      -> mark moments of input events with red spikes
            - SPD        -> plot synaptic flux
            - ref        -> plot refractory signal
            - weighting  -> weight dendritic signals by connection strength
            - spikes     -> plot output spikes over signal
            - legend_out -> place legend outside of plots
            - size       -> (x,y) size of figure
            - path       -> save plot to path
            
        '''
        from sim_soens.soen_plotting import activity_plot
        activity_plot([self],net,**kwargs)


    def get_structure(self):
        '''
        Returns structure of dendritic arbor by recursive search of neuron 
        dictionary tree.
            - Returns list of lists containing names of dendrites
            - List index associated with layer
            - List index within lists associated with branch
        * INCOMPLETE for assymetrical arbors due to recusion returning dendrites
          in branch order rather than layer --fixable
        '''
        # for w in weights:
        #     print(w)
        # print("\n")
        # Start with checking dendritic inputs to soma and getting their names
        soma_input = self.neuron.dend_soma.dendritic_inputs
        soma_input_names = list(self.neuron.dend_soma.dendritic_inputs.keys())[1:]

        # initialize arbor list and add soma inputs
        arbor = []
        strengths = []
        arbor.append(soma_input_names)
        s_list = list(self.neuron.dend_soma.dendritic_connection_strengths.values())
        strengths.append(s_list[1:])
        # call recursive function to explore all branches
        def recursive_search(input,names,leaf,arbor,count,strengths):
            '''
            Returns all inputs (however deep) feeding given input
                - Takes inputs (and their names) to a given denrite
                - Iterates over each of those input/name pairs
                - Adds new lists of the inputs and names to those dend indexes
                - Adds the new name list to the growing arbor
                - So long as names list is not empty, calls itself on new names
                - Once leaf node is reached (no new inputs), returns
                - Will exhaust all possible leaves
            '''
            # print(count)
            if leaf == True:
                names_ = []
                inputs_ = []
                strengths_ = []
                for d in names:
                    names_.append(list(input[d].dendritic_inputs))
                    inputs_.append(input[d].dendritic_inputs)
                    strengths_.append(list(input[d].dendritic_connection_strengths.values()))

                if len(names_) > 0:
                    if len(names_[0]) > 0:
                        arbor.append(names_)
                        strengths.append(strengths_)
                    # print("Leaf reached!")

                for i,input_ in enumerate(inputs_):
                    count+=1
                    # print(count)
                    recursive_search(input_,names_[i],leaf,arbor,count,strengths)
                    
            return arbor,strengths
        count=0
        arbor,strengths = recursive_search(soma_input,soma_input_names,True,
                                           arbor,count,strengths)
        
        # for s in strengths:
        #     print(s)
        # print("\n")
        # for a in arbor:
        #     print(a)
        # print("\n")
        # return
        def recursive_squish(b,s,a,strengths,i,count,i_count):
            c = a
            s_ = strengths[i]
            # print(i+1,i+len(arbor[i-1]))
            for j in range(i+1,i+len(arbor[i-1])):
                # print(i+1,i+len(arbor[i-1]))
                c += arbor[j]
                s_ += strengths[j]
                count+=1
            i_count+=len(arbor[i-1])
            b.append(c)
            s.append(s_)
            return b,s,a,strengths,i,count,i_count
            # if i+len(arbor[i-1]) == len(arbor):
            #     break

        b = []
        s = []
        count=1
        i_count=0
        for i,a in enumerate(arbor):
            if i < 2:
                b.append(a)
                s.append(strengths[i])
                count+=1
                i_count+=1
            else:
                b,s,a,strengths,i,count,i_count = recursive_squish(
                    b,s,a,strengths,i,count,i_count
                    )
                # print(i_count,count)
            if i_count == len(arbor):
                break


        arbor = b
        strengths = s
        return arbor,strengths


    def plot_custom_structure(self):
        '''
        Plots arbitrary neuron structure
            - Weighting represented in line widths
            - Dashed lines inhibitory
            - Star is cell body
            - Dots are dendrites
        '''
        self.get_structure()
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        colors = mcolors.TABLEAU_COLORS
        c_names = list(colors) + list(colors) + list(colors)
        # print(c_names[0])
        # print(colors[c_names[0]])
        arbor,strengths = self.get_structure()
        print(arbor,strengths)
        # colors = ['r','b','g',]
        # c_names = [1,2,3]
        Ns = []
        for i,a in enumerate(arbor):
            count = 0
            lsts = sum(type(el)== type([]) for el in a)
            if lsts > 0:
                for j in range(lsts):
                    count+=len(arbor[i][j])
            else: count = len(a)
            Ns.append(count)
        Ns.insert(0,1)
        Ns.reverse()
        arbor[0] = [arbor[0]]
        strengths[0] = [strengths[0]]
        # for a in arbor:
        #     print(a,"\n")
        layers=len(arbor)
        Ns_ = Ns[::-1]
        # Ns_.reverse()
        m=max(Ns)
        # print(Ns_)
        # arbor.reverse()
        c_dexes=[[] for _ in range(len(Ns))]
        for i,l in enumerate(arbor):
            count= 0
            row_sum = 0
            for j,b in enumerate(l):
                for k,d in enumerate(b):
                    if i == 0:
                        c_dexes[i].append(k)
                        if strengths[i][j][k] >= 0:
                            plt.plot([layers-i-.5, layers-i+.5], 
                                     [(m/2)+(len(b)/2)-(k+1),(m/2)-.5], 
                                     '-',color=colors[c_names[k]], 
                                     linewidth=strengths[i][j][k]*5)
                        else:
                            plt.plot([layers-i-.5, layers-i+.5], 
                                     [(m/2)+(len(b)/2)-(k+1),(m/2)-.5], 
                                     '--',color=colors[c_names[k]], 
                                     linewidth=strengths[i][j][k]*5*(-1))
                    else:
                        c_dexes[i].append(c_dexes[i-1][j])
                        c_index = c_dexes[i-1][j]
                        y1=(m/2)+Ns_[i+1]/2 - 1 - (j+k) - row_sum #-((Ns_[i]/2)%2)/2
                        y2=(m/2)+Ns_[i]/2 - j - 1 
                        # print(i,j,k,row_sum)
                        if strengths[i][j][k] >= 0:
                            plt.plot([layers-i-.5, layers-i+.5], [y1,y2], '-', 
                                     color=colors[c_names[c_index]], 
                                     linewidth=strengths[i][j][k]*5)
                        else:
                            plt.plot([layers-i-.5, layers-i+.5], [y1,y2], '--', 
                                     color=colors[c_names[c_index]], 
                                     linewidth=strengths[i][j][k]*5*(-1))
                    count+=1
                row_sum += len(arbor[i][j])-1 

        x_ticks=[]
        x_labels=[]
        # print(Ns)
        for i,n in enumerate(Ns):
            if (np.max(Ns)) < 10:
                size = 15
            else:
                size = 100/(np.max(Ns))
            x_labels.append(f"L{len(Ns)-(i+1)}")
            x_ticks.append(i+.5)
            if n == np.max(Ns):
                plt.plot(np.ones(n)*i+.5,np.arange(n),'ok',ms=size)
            elif n != 1:
                factor = 1 # make proportional
                plt.plot(np.ones(n)*i+.5,np.arange(n)*factor+(.5*np.max(Ns)-.5*n), 
                         'ok', ms=size)
            else:
                plt.plot(np.ones(n)*i+.5, np.arange(n)+(.5*np.max(Ns)-.5*n), '*k', ms=30)
                plt.plot(np.ones(n)*i+.5, np.arange(n)+(.5*np.max(Ns)-.5*n), '*y', ms=20)

        x_labels[-1]="soma"
        plt.yticks([],[])
        plt.xticks(x_ticks,x_labels)
        plt.xlim(0,len(Ns))
        plt.ylim(-1, max(Ns))
        plt.xlabel("Layers",fontsize=16)
        plt.ylabel("Dendrites",fontsize=16)
        plt.title('Dendritic Arbor',fontsize=20)
        plt.show()

    def plot_structure(self):
        '''
        Plots arbitrary neuron structure
            - syntax
                -> NeuralZoo.plot_structure()
            - Weighting represented in line widths
            - Dashed lines inhibitory
            - Star is cell body
            - Dots are dendrites
        '''
        from sim_soens.soen_plotting import structure
        structure(self)

    def plot_arbor_activity(self,net,**kwargs):
        '''
        Plots signal and optional flux over dendritic structure
        - syntax
            -> NeuralZoo.plot_arbor_activity(net,phir=True)
        '''
        from sim_soens.soen_plotting import arbor_activity
        arbor_activity(self,net,**kwargs)