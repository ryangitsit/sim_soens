import numpy as np

# from soen_component_library import common_synapse
from .soen_sim import neuron, dendrite, synapse


class SuperNode():

    def __init__(self,**entries):
        '''
        Generate node object
         - node object is an object that makes and contains a neuron object
         - contains other structural and meta parameters about the neuron
         - structure defined by the weights argument [layer][group][dends]
             - for example (values = weights of connections):
                 weights = [
                 [[0.2,0.5]], # 2 dends in lay1 (1 grp max), feed to soma
                 [[.1,.3,.5],[.7,.7]] # 3 dends feed to first dend of lay1
                 ]                    # 2 dends feed to dend2 of lay1
         - all general parameters, and those associated with dendrites (_di),
             refractory dendrites (_ref), and somas (_ni or _n) accepted
         - for parameters specfically arranged according to dendritic tree,
             pass in a list of lists of that parameter with the dendritic
             strucuture (biases, taus, types, betas)
             - note, this method applies to in-arbor dendrites only and the
                 parameter structure should exclude the soma
             - betas takes exponents
             - biases take list indices
         - Synapses will automatically be placed at every outermost dendrite
           unless synaptic_strucure used (a list of arbor structures [with soma]
           where each item is a synapse and values are strength of connection to 
           that component)
         - kwargs (use SuperNode.parameter_print() to view)
            # general params

            - ib_n
            - ib
            - ib_di
            - ib_ref

            - tau_ni
            - tau_di

            - tau_ref
            - beta_ni
            - beta_di
            - beta_ref

            - w_sd
            - w_dn
            - seed

            - loops_present
            - loops_present_ref

            # group params
            - weights
            - biases
            - taus
            - types
            - synaptic_structure
        '''  

        self.random_syn = False
        self.__dict__.update(entries)
        self.params = self.__dict__  

        # give neuron name if not already assigned
        if "name" not in self.params:
            self.params['name'] = f"rand_neuron_{int(np.random.rand()*100000)}"

        # create a neuron object given init params
        self.neuron = neuron(**self.params)
        self.neuron.dend_soma.branch=0

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
                            beta = self.betas[i][j][k]
                            dend_params["beta_di"] =(np.pi*2)*10**beta
                        if hasattr(self, 'biases'):
                            if hasattr(self, 'types'):
                                bias = self.biases[i][j][k]
                                if self.types[i][j][k] == 'ri':
                                    dend_params["ib"] = self.ib_list_ri[bias]
                                else:
                                    dend_params["ib"] = self.ib_list_rtti[bias]
                            else:
                                dend_params["ib"] = self.ib_list_ri[bias]
                            dend_params["ib_di"] = dend_params["ib"]
                        if hasattr(self, 'taus'):
                            dend_params["tau_di"] = self.taus[i][j][k]
                        if hasattr(self, 'types'):
                            dend_params["loops_present"] = self.types[i][j][k]
                            # print("HERE",self.types[i][j][k])
                        else:
                            dend_params["loops_present"] = 'ri'

                        # self.params = self.__dict__
                        name = f"{self.neuron.name}_lay{i+1}_branch{j}_den{k}"
                        dend_params["dend_name"] = name
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

        # make dendrites readable through node object
        if dendrites:
            self.dendrites = dendrites

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
                                if self.random_syn==False:
                                    connect=self.synaptic_structure[ii][i][j][k]
                                elif self.random_syn==True:
                                    connect=np.random.rand()
                                dendrites[i][j][k].add_input(s, 
                                    connection_strength = connect)
                                
        else:
            self.synaptic_layer()


        

    ############################################################################
    #                           input functions                                #
    ############################################################################  

    def synaptic_layer(self):
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
        uniform_input:
         - syntax -> SuperNode.uniform_input(SuperInput)
         - Adds the same input channel to all available synapses
         - note, the first channel of the SuperInput object will be used
        '''
        for S in self.synapse_list:
            S.add_input(input.signals[0])

    def one_to_one(self,input):
        '''
        one_to_one:
         - syntax -> SuperNode.one_to_one(SuperInput)
         - connects input channels to synapses, matching indices
         - len(synapse_list) == input.channels (required)
        '''
        for i,S in enumerate(self.synapse_list):
            S.add_input(input.signals[i])

    def custom_input(self,input,synapse_indices):
        '''
        custom_input:
         - syntax -> SuperNode.custom_input(SuperInput,synapse_indices)
            - synapse_indices = list of `synapse_list` indices to connect to
         - Adds the same input channel to specific synapses
         - Simply defined as list of indice tuples
        '''
        for connect in synapse_indices:
            self.synapses_list[connect].add_input(input.signals[0])
                            
    def multi_channel_input(self,input,connectivity=None):
        '''
        multi_channel_input:
         - syntax -> multi_channel_input(SuperInput,connectivity)]
            - connectivity = list of lists that define synapse_list index and 
            SuperInput.signal index to be connected
            - connectivity = [[synapse_index_1,SuperInput_index_7],[...],[...]]
         - Connects multi-channel input to multiple synapses according to
           specified connectivity
        '''
        for connect in connectivity:
            # print(connect[0],connect[1])
            self.synapse_list[connect[0]].add_input(input.signals[connect[1]])



    ############################################################################
    #                           helper functions                               #
    ############################################################################  

    def parameter_print(self):
        print("\nSOMA:")
        # print(f" ib = {self.neuron.ib}")
        print(f" ib_n = {self.neuron.ib_n}")
        print(f" tau_ni = {self.neuron.tau_ni}")
        print(f" beta_ni = {self.neuron.beta_ni}")
        # print(f" tau = {self.neuron.tau}")
        print(f" loops_present = {self.neuron.loops_present}")
        print(f" s_th = {self.neuron.s_th}")
        syn_in = list(self.neuron.dend_soma.synaptic_inputs.keys())
        print(f" synaptic_inputs = {syn_in}")
        dend_in = list(self.neuron.dend_soma.dendritic_inputs.keys())
        print(f" dendritic_inputs = {dend_in}")

        print("\nREFRACTORY DENDRITE:")
        print(f" ib_ref = {self.neuron.ib_ref}")
        print(f" tau_ref = {self.neuron.tau_ref}")
        print(f" beta_ref = {self.neuron.beta_ref}")
        print(f" loops_present = {self.neuron.loops_present}")
        ref_in = list(self.neuron.dend__ref.dendritic_inputs.keys())
        print(f" dendritic_inputs = {ref_in}")

        print("\nDENDRITIC ARBOR:")
        if len(self.dendrite_list) == 2: print ('  empty')
        for dend in self.dendrite_list:
            # name = " IN-ARBOR"
            # if "REFRACTORY:" in dend.name:
            #     name = 'refractory'
            # elif "SOMATIC:" in dend.name:
            #     name = "soma"
            if 'ref' not in dend.name and 'soma' not in dend.name:
                print(f" ", dend.name)
                print(f"   ib_di = {dend.ib}")
                print(f"   tau_di = {dend.tau_di}")
                print(f"   beta_di = {dend.beta_di}")
                print(f"   loops_present = {dend.loops_present}")
                syns_in = list(dend.synaptic_inputs.keys())
                dends_in = list(dend.dendritic_inputs.keys())
                print(f"   synaptic_inputs = {syns_in}")
                print(f"   dendritic_inputs = {dends_in}")

        # print("\n\n")

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
        from .soen_plotting import arbor_activity
        arbor_activity(self,net,**kwargs)

    def plot_structure(self):
        from .soen_plotting import structure
        structure(self)

    def plot_neuron_activity(self,net,**kwargs):
        '''
        Plots signal activity for a given neuron
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
        from .soen_plotting import activity_plot
        activity_plot([self],net,**kwargs)