import numpy as np

# from soen_component_library import common_synapse
from soen_sim import neuron, dendrite, synapse


class SuperNode():

    def __init__(self,**entries):
        '''
        Generate node object
            - node object is an object that makes and contains a neuron object
            - contains other structural and meta parameters about the neuron

        '''  

        self.random_syn = False
        self.__dict__.update(entries)
        self.params = self.__dict__  

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
                            dend_params["ib_di"] = dend_params["ib"]
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
                syn = synapse(name = f'{self.neuron.name}_syn{count}')
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
            print(f"   beta_di = {dend.beta_di}")
            print(f"   loops_present = {dend.loops_present}")
            print(f"   synaptic_inputs = {list(dend.synaptic_inputs.keys())}")
            print(f"   dendritic_inputs = {list(dend.dendritic_inputs.keys())}")
        print("\n\n")

        # print("\nCONNECTIVITY:")

    def check_arbor_structor(self,arbor):
        '''
        Checks if arboric structure is correct
            - print explanatory messages otherewise
        '''
        for i,layer in enumerate(arbor):
            for j,dens in enumerate(layer):
                for k,d in enumerate(dens):
                    if i == 0:
                        if len(layer) != 1:
                            print('''
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            ARBOR WARNING: First layer should only have one group.
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            ''')
                    else:
                        if len(layer) != len(np.concatenate(arbor[i-1])):
                            print(f'''
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            ARBOR ERROR: Groups in layer {i} must be equal to total dendrites in layer {i-1}
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            ''')
                            return
                        
    def plot_arbor_activity(self,net,**kwargs):
        from soen_plotting import arbor_activity
        arbor_activity(self,net,**kwargs)

    def plot_structure(self):
        from soen_plotting import structure
        structure(self)

    def plot_neuron_activity(self,net,**kwargs):
        from soen_plotting import activity_plot
        activity_plot([self],net,**kwargs)