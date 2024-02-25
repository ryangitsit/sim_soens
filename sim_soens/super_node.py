import numpy as np

from .soen_components import neuron, dendrite, synapse
from .soen_utilities import dend_load_arrays_thresholds_saturations, index_finder
from .super_functions import timer_func

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

        # default settings
        self.w_sd=1
        self.random_syn = False
        self.weights = []
        np.random.seed(None)

        # writing over default settings
        self.__dict__.update(entries)
        # self.params = self.__dict__  
        # print("PARAMS: ", self.__dict__.keys())
        # give neuron name if not already assigned
        if hasattr(self,'name')==False:
            self.name = f"rand_neuron_{int(np.random.rand()*100000)}"

        # create a neuron object given init params
        neuron_params = {k:v for k,v in self.__dict__.items() if (k!='dendrites' or 
                                                                k!='dendrite_list' or
                                                                k!='synapses' or
                                                                k!='synapse_list' or
                                                                k!='synaptice_structure' or
                                                                k!='params' 
                                                                )}
        # print("N-PARAMS: ", neuron_params.keys())
        self.neuron = neuron(**neuron_params)
        self.neuron.dend_soma.branch=0

        # add somatic dendrite (dend_soma) and refractory dendrite to list
        self.dendrite_list = [self.neuron.dend_soma,self.neuron.dend__ref]

        # default random seed
        np.random.seed(None)

        # for systematic seeding of multi-run experiments
        if hasattr(self, 'seed'):
            np.random.seed(self.seed)
            # print("random seed: ",self.seed)

        # check that the structure implied by .weights is compatible with construction method
        self.check_arbor_structor(self.weights)
                        
        self.make_dendrites()
        self.connect_dendrites()
        self.make_and_connect_synapses()

    ############################################################################
    #                           dendritic arbor                                #
    ############################################################################  

    def make_dendrites(self):
        '''
        Makes dendrite components with appropriate parameters
            - Uses self.weights to define structure
        '''
        # dendrites attribute will have some structure as arbor
        # [layer][group][dendrite]
        # populated with dendrite objects
        
        self.dendrites = [ [] for _ in range(len(self.weights)) ]

        if len(self.weights)>0:

            if (hasattr(self, 'betas') 
                or hasattr(self, 'biases') 
                or hasattr(self, 'types') 
                or hasattr(self, 'taus')):
                self.specified_arbor_params()
            
            else:
                self.global_arbor_params()

    def global_arbor_params(self):


        dend_params = {k:v for k,v in self.__dict__.items() if (k!='dendrites'
                                                                and k!='dendrite_list'
                                                                and k!='synapses' 
                                                                and k!='synapse_list' 
                                                                and k!='synaptice_structure' 
                                                                and k!='params' 
                                                                and k!= 'neuron'
                                                                and k!= 'weights'
                                                                )}

        # print("GLOB-D-PARAMS: ", dend_params.keys())

        count=0
        den_count = 0
        for i,layer in enumerate(self.weights):
            c=0
            for j,dens in enumerate(layer):
                sub = []
                for k,d in enumerate(dens):

                    # parameters for creating current dendrite
                    # dend_params = self.params
                    
                    dend_params["dend_name"] = f"{self.neuron.name}_lay{i+1}_branch{j}_den{k}"

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
                self.dendrites[i].append(sub)

    def specified_arbor_params(self):
        d_params_ri = dend_load_arrays_thresholds_saturations('default_ri')
        d_params_rtti = dend_load_arrays_thresholds_saturations('default_rtti')
        count=0
        den_count = 0

        dend_params = {k:v for k,v in self.__dict__.items() if (k!='dendrites'
                                                                and k!='dendrite_list'
                                                                and k!='synapses' 
                                                                and k!='synapse_list' 
                                                                and k!='synaptice_structure' 
                                                                and k!='params' 
                                                                and k!= 'neuron'
                                                                and k!= 'weights'
                                                                )}

        # print("SPEC-D-PARAMS: ", dend_params.keys())

        for i,layer in enumerate(self.weights):
            c=0
            for j,dens in enumerate(layer):
                sub = []
                for k,d in enumerate(dens):
                    #(todo) add flags and auto connects for empty connections

                    # parameters for creating current dendrite
                    # dend_params = self.params

                    # check for any dendrite-specific parameters
                    # if so, use in dend_parameters
                    # otherwise, one of the following will be used
                    #   - default parameters (defined in dendrite class)
                    #   - general dendrite parameters defined in this node's
                    #     initialization 
                    if hasattr(self, 'betas'):
                        beta = self.betas[i][j][k]
                        dend_params["beta_di"] = (np.pi*2)*10**beta
                    if hasattr(self, 'biases'):
                        if hasattr(self, 'types'):
                            bias = self.biases[i][j][k]
                            if self.types[i][j][k] == 'ri':
                                dend_params["ib"] = d_params_ri["ib__list"][bias]
                            else:
                                dend_params["ib"] = d_params_rtti["ib__list"][bias]
                        else:
                            dend_params["ib"] = d_params_ri["ib__list"][bias]
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
                self.dendrites[i].append(sub)



    def connect_dendrites(self):
        '''
        Connects dendrites in arbor-form implied by self.weights
        '''
        # iterate over dendrites and connect them as defined by structure
        for i,l in enumerate(self.dendrites):
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
                        receiving_dend = np.concatenate(self.dendrites[i-1])[j]
                        receiving_dend.add_input(d, 
                            connection_strength=self.weights[i][j][k])
                        d.branch = receiving_dend.branch
                    d.output_connection_strength = self.weights[i][j][k]

        # add the somatic dendrite to the 0th layer of the arboric structure
        self.dendrites.insert(0,[[self.neuron.dend_soma]])


    ############################################################################
    #                            New Arbor Methods                             #
    ############################################################################ 

    def make_weights(self,size,exin,fixed):
        '''
        '''
        ones = np.ones(size)
        symm = 1

        if exin != None:
            # print(exin)
            symm = np.random.choice([-1,0,1], p=[exin[0]/100,exin[1]/100,exin[2]/100], size=size)

        if fixed is not None:
            # print("fixed")
            w = ones*fixed*symm
        else:
            w = np.random.rand(size)*symm
        return w
    
    def recursive_downstream_inhibition_counter(self,dendrite,superdend):
        for out_name,out_dend in dendrite.outgoing_dendritic_connections.items():
            cs = out_dend.dendritic_connection_strengths[dendrite.name]
            if cs < 0:
                superdend.downstream_inhibition += 1
            self.recursive_downstream_inhibition_counter(out_dend,superdend)

    def add_inhibition_counts(self):
        '''
        '''
        for dendrite in self.dendrite_list:
            dendrite.downstream_inhibition = 0
            self.recursive_downstream_inhibition_counter(dendrite,dendrite)

    # @timer_func
    def max_s_finder(self,dendrite):
        '''
        '''
        d_params_ri = dend_load_arrays_thresholds_saturations('default_ri')
        ib_list = d_params_ri["ib__list"]
        s_max_plus__vec = d_params_ri["s_max_plus__vec"]
        _ind_ib = index_finder(ib_list[:],dendrite.ib) 
        return s_max_plus__vec[_ind_ib]

    def normalize_fanin(self,coeff):
        '''
        '''
        # print("NORMALIZING")
        for dendrite in self.dendrite_list:
            if len(dendrite.dendritic_connection_strengths) > 0:
                max_s = self.max_s_finder(dendrite) - dendrite.phi_th
                cs_list = []
                max_list = []
                influence = []
                for in_name,in_dend in dendrite.dendritic_inputs.items():
                    cs = dendrite.dendritic_connection_strengths[in_name]
                    if 'ref' in in_name: cs = 0
                    max_in = self.max_s_finder(in_dend)
                    cs_list.append(cs)
                    max_list.append(max_in)
                    influence.append(cs*max_in)
                if sum(influence) > max_s:
                    norm_fact = sum(influence)/max_s
                    cs_normed = cs_list/norm_fact
                    for i,(in_name,cs) in enumerate(dendrite.dendritic_connection_strengths.items()):
                        if 'ref' not in in_name:
                            dendrite.dendritic_connection_strengths[in_name] = cs_normed[i]*coeff

    def normalize_fanin_symmetric(self,buffer=0,coeff=1):
        '''
        
        '''
        for dendrite in self.dendrite_list:
            if len(dendrite.dendritic_connection_strengths) > 0:  

                # print(f"{dendrite.name} =>  phi_th = {dendrite.phi_th} :: max_s = {self.max_s_finder(dendrite)}")
                max_phi = 0.5 - dendrite.phi_th*buffer

                negatives = []
                neg_max   = []
                neg_dends = []

                positives = []
                pos_max   = []
                pos_dends = []


                for in_name,in_dend in dendrite.dendritic_inputs.items():
                    cs = dendrite.dendritic_connection_strengths[in_name]
                    if 'ref' in in_name: cs = 0
                    max_in = self.max_s_finder(in_dend)
                    # print(f"  {in_name} -> {cs}") 

                    if cs<0:
                        # print(cs)
                        negatives.append(cs)
                        neg_max.append(cs*max_in)
                        neg_dends.append(in_dend)

                    elif cs>0:
                        positives.append(cs)
                        pos_max.append(cs*max_in)
                        pos_dends.append(in_dend)
            

                if sum(pos_max) > max_phi:
                    # print(f" Normalizing input to {dendrite.name} from {sum(pos_max)} to {max_phi}")
                    for pos_dend in pos_dends:
                        cs = dendrite.dendritic_connection_strengths[pos_dend.name]
                        cs_max = cs*self.max_s_finder(pos_dend)
                        cs_proportion = cs_max/sum(pos_max)
                        cs_normalized = max_phi*cs_proportion/self.max_s_finder(pos_dend) 
                        # print(f"   {pos_dend} -> {cs_normalized}")
                        dendrite.dendritic_connection_strengths[pos_dend.name] = cs_normalized*coeff
                # print(sum(np.abs(neg_max)))
                if sum(np.abs(neg_max)) > max_phi:
                    # print(f" Normalizing input to {dendrite.name} from {sum(neg_max)} to {max_phi}")

                    for neg_dend in neg_dends:
                        cs = np.abs(dendrite.dendritic_connection_strengths[neg_dend.name])
                        cs_max = np.abs(cs*self.max_s_finder(neg_dend))
                        cs_proportion = cs_max/sum(np.abs(neg_max))
                        cs_normalized = np.abs(max_phi*cs_proportion/self.max_s_finder(neg_dend))*-1
                        # print(f"   {neg_dend} -> {cs_normalized}")
                        dendrite.dendritic_connection_strengths[neg_dend.name] = cs_normalized*coeff

                # print("\n")

    def random_flux(self,rand_flux):
        '''
        '''
        # print("RANDOM FLUX")
        for l,layer in enumerate(self.dendrites):
            for g,group in enumerate(layer):
                for d,dend in enumerate(group):
                    if 'ref' not in dend.name and 'soma' not in dend.name:
                        sign = np.random.choice([-1,1], p=[.5,.5], size=1)[0]
                        dend.offset_flux = np.random.rand()*rand_flux*sign



    ############################################################################
    #                              Synapses                                    #
    ############################################################################ 

    def make_and_connect_synapses(self):
        '''
        Creates and connects all synapses using a number of possible techniques
            - `syns` is for adding multiple synapses specically to leaf dendrites
            - `Synaptic_structure` uses a self.weights type list of lists to place synapses
            - `Synaptic_indices` applies synapses to specific leaf denrite indices
            - `Synaptic_layer` (default) places a synapse at all leaf dendrites
        '''
        # if syns attribute, connect as a function of grouping to final layer
        if hasattr(self, 'syns'):
            self.synapse_list = []
            self.synapses = [[] for _ in range(len(self.syns))]
            for i,group in enumerate(self.syns):
                for j,s in enumerate(group):
                    syn = synapse(name=s)
                    self.synapses[i].append(syn)
                    self.synapse_list.append(syn)
            count=0
            for j, subgroup in enumerate(self.dendrites[len(self.dendrites)-1]):
                for k,d in enumerate(subgroup):
                    for s in self.synapses[count]:
                        self.dendrites[len(self.dendrites)-1][j][k].add_input(s, 
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
                syn = synapse(name=f'{self.neuron.name}_syn{ii}')

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
                                self.dendrites[i][j][k].add_input(s, 
                                    connection_strength = connect)
                                
        elif hasattr(self, 'synaptic_indices'):
            self.synapse_list = []
            for i,layer in enumerate(self.dendrites):
                for j,group in enumerate(layer):
                    for k,dend in enumerate(group):
                        for ii,syn in enumerate(self.synaptic_indices):
                            name = f'{self.neuron.name}_syn{ii}'
                            s = synapse(name=name)
                            self.synapse_list.append(s)
                            if hasattr(self, 'synaptic_strengths'):
                                connect = self.synaptic_strengths[ii]
                            else:
                                connect = self.w_sd
                            self.dendrites[syn[0]][syn[1]][syn[2]].add_input(
                                s, 
                                connection_strength = connect
                                )                       
        else:
            self.synaptic_layer()

        # self.synapse_list.append(self.neuron.dend__ref.synaptic_inputs[f"{self.name}__syn_refraction"])
        self.refractory_synapse = self.neuron.dend__ref.synaptic_inputs[f"{self.name}__syn_refraction"]


    ############################################################################
    #                           input functions                                #
    ############################################################################  

    def synaptic_layer(self):
        '''
        Add synapse to all leaf-node dendrites in arbor
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
            if 'ref' not in S.name:
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
        '''
        Prints node parameters -- for user verification
        '''
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
                        
    def __copy__(self):
        '''
        Copy a SuperNode object
        '''
        copy_object = SuperNode()
        return copy_object

    def __deepcopy__(self, memodict={}):
        '''
        Deep copy a SuperNode object
        '''
        import copy
        copy_object = SuperNode()
        copy_object.neuron = self.neuron
        copy_object.dendrites = copy.deepcopy(self.dendrites)
        return copy_object
    
    ############################################################################
    #                           plotting functions                             #
    ############################################################################        
    def plot_arbor_activity(self,net,**kwargs):
        '''
        Plots activity of node (after simulation) superimposed on arbor morphology
        '''
        from sim_soens.soen_plotting import arbor_activity
        arbor_activity(self,net,**kwargs)

    def plot_structure(self):
        '''
        Plots structure of node
        '''
        from sim_soens.soen_plotting import structure
        structure(self)

    def plot_neuron_activity(self,**kwargs):
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
        from sim_soens.soen_plotting import activity_plot
        activity_plot([self],**kwargs)