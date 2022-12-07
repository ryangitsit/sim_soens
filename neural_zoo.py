#%%
import numpy as np

# from _util import (
#     physical_constants, index_finder)

from soen_component_library import (
    common_dendrite, common_synapse, common_neuron)

# from soen_utilities import (
#     dend_load_rate_array, dend_load_arrays_thresholds_saturations,
#     physical_constants, index_finder)

# from soen_sim import input_signal, synapse, neuron, network
# from super_input import SuperInput
# from params import default_neuron_params, nine_pixel_params, weights_3
# from _plotting__soen import raster_plot


# from super_input import SuperInput

'''
Here a class for calling from a 'zoo' of possible neurons is implemented.

Plan:
 - Syntax for custom neuron calls based on dendritic structures and parameters.
 - Library of predefined neurons, both bio-inspired and engineering-specific
 - Should include a testing/plotting paradigm inherent in the class
 - Add more explicit connectivity defintions and corresponding plotting
'''


class NeuralZoo():

    def __init__(self,**entries):
        self.__dict__.update(entries)

        if self.type == '3fractal':
            self.fractal_three()

        if self.type == 'single':
            self.single()

        if self.type == 'custom':
            self.custom()
    
    def single(self):

        self.synapse = common_synapse(1)

        self.dendrite = common_dendrite(1, 'ri', self.beta_di, 
                                          self.tau_di, self.ib)
                                    
        self.dendrite.add_input(self.synapse, connection_strength = self.w_sd)

        self.neuron = common_neuron(1, 'ri', self.beta_ni, self.tau_ni, 
                                      self.ib, self.s_th, 
                                      self.beta_ref, self.tau_ref, self.ib_ref)

        self.neuron.add_input(self.dendrite, connection_strength = self.w_dn)


    def fractal_three(self):
        H = 3 # depth
        n = [3,3] # fanning at each layer, (length = H-1), from soma to synapses

        fractal_neuron = common_neuron(1, 'ri', self.beta_ni, self.tau_ni, 
                                       self.ib, self.s_th, 
                                       self.beta_ref, self.tau_ref, self.ib_ref)
        fractal_neuron.name = 'name'
        dendrites = [ [] for _ in range(H-1) ]
        synapses = []

        count=0
        count_syn=0
        last_layer = 1
        # returns dendrites[layer][dendrite] = dendrites[H-1][n_h]
        for h in range(H-1): 
            for d in range(n[h]*last_layer):
                dendrites[h].append(common_dendrite(count, 'ri', self.beta_di, 
                                    self.tau_di, self.ib))

                if h == H-2:
                    synapses.append(common_synapse(d))
                    dendrites[h][d].add_input(synapses[d], 
                                              connection_strength = self.w_sd)
                count+=1
            last_layer = n[h]

        for i,layer in enumerate(dendrites):
            # print("layer:", i)
            for j,d in enumerate(layer):
                # print("  dendrite", j)
                if i < H-2:
                    for g in range(n[1]):
                        d.add_input(dendrites[i+1][j*n[1]+g], 
                                    connection_strength=self.w_dd)
                        # print(j,j*n[1]+g)
                    fractal_neuron.add_input(d, connection_strength=self.w_dn)
        self.dendrites = dendrites
        self.synapses = synapses
        self.fractal_neuron = fractal_neuron


    def custom(self):
        '''
        Arbitrary neuron generation
            - Define dendritic structure with weight or structure input
        '''    
        if hasattr(self, 's_th'):
            # print("structure")
            self.s_th = self.s_th
        else:
            self.s_th = self.s_th_factor_n*self.s_max_n
        # create a neuron body (soma and refractory loop) with called params
        custom_neuron = common_neuron(1,'ri',self.beta_ni, self.tau_ni,
                                      self.ib_n, self.s_th, self.beta_ref, 
                                      self.tau_ref, self.ib_ref)
        # custom_neuron.name = 'custom_neuron'
        custom_neuron.normalize_input_connection_strengths=1
        self.neuron = custom_neuron

        # check how arbor is defined
        # structure just gives arbor form
        if hasattr(self, 'structure'):
            # print("structure")
            arbor = self.structure
        # weights defines structure implicitly and defines connection strengths
        elif hasattr(self, 'weights'):
            # print("weights")
            arbor = self.weights
        else:
            arbor = []
            # dendrites = [[[]]]
        dendrites = [ [] for _ in range(len(arbor)) ]
        if len(arbor)>0:
            # initialize a list of lists for holding dendrits in each arbor layer
            # iterate over the defined structure
            # create dendritic arbor with defined parameters (may be dend specific)
            count=0
            den_count = 0
            for i,layer in enumerate(arbor):
                c=0
                for j,dens in enumerate(layer):
                    sub = []
                    for k,d in enumerate(dens):
                        if hasattr(self, 'betas'):
                            self.beta_di=(np.pi*2)*10**self.betas[i][j][k]
                        if hasattr(self, 'biases'):
                            if hasattr(self, 'types'):
                                if self.types[i][j][k] == 'ri':
                                    self.ib= self.ib_list_ri[self.biases[i][j][k]]
                                else:
                                    self.ib= self.ib_list_rtti[self.biases[i][j][k]]
                            else:
                                self.ib= self.ib_list_ri[self.biases[i][j][k]]
                        if hasattr(self, 'types'):
                            type = self.types[i][j][k]
                        if hasattr(self, 'taus'):
                            self.tau_di = self.taus[i][j][k]
                        else:
                            type = 'ri'
                        sub.append(common_dendrite(f"lay{i}_branch{j}_den{k}", type, 
                                            self.beta_di,self.tau_di, self.ib))
                        den_count+=1
                        c+=1
                    dendrites[i].append(sub)
        
            # iterate over dendrites and connect them as defined by structure
            for i,l in enumerate(dendrites):
                for j, subgroup in enumerate(l):
                    for k,d in enumerate(subgroup):
                        if i==0:
                            # print(i,j,k, " --> soma")
                            custom_neuron.add_input(d, 
                                connection_strength=self.weights[i][j][k])
                            # custom_neuron.add_input(d, 
                            #     connection_strength=self.w_dn)
                        else:
                            # print(i,j,k, " --> ", i-1,0,j)
                            np.concatenate(dendrites[i-1])[j].add_input(d, 
                                connection_strength=self.weights[i][j][k])

        dendrites.insert(0,[[custom_neuron.dend__nr_ni]])
        # print('dendrites:', dendrites)
        # if synapses also defined, connect as a function of grouping
        if hasattr(self, 'syns'):
            self.synapses = [[] for _ in range(len(self.syns))]
            for i,group in enumerate(self.syns):
                for j,s in enumerate(group):
                    self.synapses[i].append(common_synapse(s))
            count=0
            # print(len(dendrites[len(dendrites)-1][0]))
            for j, subgroup in enumerate(dendrites[len(dendrites)-1]):
                for k,d in enumerate(subgroup):
                    for s in self.synapses[count]:
                        dendrites[len(dendrites)-1][j][k].add_input(s, 
                            connection_strength = self.syn_w[j][k])
                    count+=1

        elif hasattr(self, 'synaptic_structure'):
            self.synapses = [[] for _ in range(len(self.synaptic_structure))]
            for i,layer in enumerate(self.synaptic_structure):
                self.synapses[i] = [[] for _ in range(len(self.synaptic_structure[i]))]
                for j,group in enumerate(layer):
                    for k,s in enumerate(group):
                        if s != 0:
                            # print('synapse')
                            self.synapses[i][j].append(common_synapse(s))
                        else:
                            # print('no synapse')
                            self.synapses[i][j].append(0)
            # print('synapses:', self.synapses)
            count=0
            for i,layer in enumerate(self.synapses):
                for j, subgroup in enumerate(layer):
                    for k,d in enumerate(subgroup):
                        s=self.synapses[i][j][k]
                        if s !=0:
                            dendrites[i][j][k].add_input(s, 
                                connection_strength = self.synaptic_structure[i][j][k])
                        count+=1
            # else:
            #     for i in self.synapses[0][0]:
            #         self.neuron.dend__nr_ni.add_input(self.synapses[0][0][i],
            #         connection_strength=self.synaptic_structure[0][0][i])
        else:
            self.synapses = []
            count=0
            for j,g in enumerate(dendrites[-1]):
                for k,d in enumerate(g):
                    self.synapses.append([common_synapse(f'branch_{j}syn_{k}')])
                    self.synapses[count][0].spd_duration=2
                    d.add_input(self.synapses[count][0],connection_strength=self.w_sd)
                    count+=1
        if dendrites:
            self.dendrites = dendrites

                    
    def plot_structure(self):
        '''
        This is only for 3fractal neuron
        '''

        import matplotlib.pyplot as plt
        layers = [[] for i in range(len(self.dendrites))]
        for i in range(len(layers)):
            for j in range(len(self.dendrites[i])):
                layers[i].append(list(self.dendrites[i][j].dendritic_inputs.keys()))
        # print(layers)
        colors = ['r','b','g',]
        Ns = [len(layers[i]) for i in range(len(layers))]
        Ns.reverse()
        Ns.append(1)
        for i,l in enumerate(layers):
            for j,d in enumerate(l):
                if len(d) > 0:
                    for k in layers[i][j]:
                        plt.plot([i+.5, i+1.5], [k-3,j+3], '-k', color=colors[j])
        for i in range(Ns[-2]):
            plt.plot([len(layers)-.5, len(layers)+.5], [i+len(Ns),len(Ns)+1], 
                '-k', color=colors[i], linewidth=1)
        for i,n in enumerate(Ns):
            if n == np.max(Ns):
                plt.plot(np.ones(n)*i+.5, np.arange(n), 'ok', ms=10)
            else:
                plt.plot(np.ones(n)*i+.5, np.arange(n)+(.5*np.max(Ns)-.5*n), 
                    'ok', ms=10)
        plt.xticks([.5, 1.5,2.5], ['Layer 1', 'layer 2', 'soma'])
        plt.yticks([],[])
        plt.xlim(0,len(layers)+1)
        plt.ylim(-1, max(Ns))
        plt.title('Dendritic Arbor')
        plt.show()





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
        soma_input = self.neuron.dend__nr_ni.dendritic_inputs
        soma_input_names = list(self.neuron.dend__nr_ni.dendritic_inputs.keys())[1:]

        # initialize arbor list and add soma inputs
        arbor = []
        strengths = []
        arbor.append(soma_input_names)
        s_list = list(self.neuron.dend__nr_ni.dendritic_connection_strengths.values())
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
                b,s,a,strengths,i,count,i_count = recursive_squish(b,s,a,strengths,i,count,i_count)
                # print(i_count,count)
            if i_count == len(arbor):
                break


        arbor = b
        strengths = s
        # for a in strengths:
        #     print(a)
        # print("\n")

        # b = [arbor[0],arbor[1],arbor[2]+arbor[3]+arbor[4]]
        # for a in b:
        #     print(a,"\n")
        # if len(self.weights) != len(arbor):
        #     for i,a in enumerate(arbor):
        #         lists =  sum(type(el)== type([]) for el in a)
        #         if lists > 1:
        #             layer = []
        #             s_layer = []
        #             for j in range(lists):
        #                 if len(arbor) >= lists+j:
        #                     if j == 0:
        #                         layer = arbor[lists+j]
        #                         s_layer = strengths[lists+j]
        #                     else:
        #                         layer = layer + arbor[lists+j]
        #                         s_layer = s_layer + strengths[lists+j]
        #             arbor[i+1] = layer
        #             strengths[i+1] = s_layer
        #             for j in range(lists-1):
        #                 del arbor[lists+1+j]
        #                 del strengths[lists+1+j]
        #         if len(arbor)==i+lists:
        #             break
                
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
        print(Ns)
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

    def arbor_activity_plot(self):
        import matplotlib.pyplot as plt
        # signal = net.neurons["custom_neuron"].dend__nr_ni.s
        # ref = net.neurons["custom_neuron"].dend__ref.s
        signal = self.neuron.dend__nr_ni.s
        S = []
        den_arb = self.dendrites
        L = len(den_arb)
        M = len(den_arb[-1])
        # print(M)
        plt.figure(figsize=(6,8))
        for i,l in enumerate(den_arb):
            for j,g in enumerate(l):
                for k,d in enumerate(g):
                    if i < 10:
                        s = d.s[::10]
                        t = np.linspace(0,1,len(s))
                        S.append(d.s)
                        # print(i,j,k,'  --  ',np.max(d.s))
                        # print(len(l),len(den_arb[i-1]))
                        plt.plot(t+(L-i)*1.2,(s+j*4.2+k*1.5)*(M/len(l))+1.75*(M-i), 
                                 linewidth=2, label=f'{i} mean dendritic signal')
        T = t + L*1.6
        s_n = signal[::10]+19.5
        plt.plot(T,s_n,linewidth=3,color='r')
        plt.xticks([],[])
        plt.yticks([],[])
        plt.title('Signal Propagation Across Arbor')
        plt.show()

    def arbor_activity_plot_(self):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        colors = mcolors.TABLEAU_COLORS
        c_names = list(colors) + list(colors) + list(colors)
        # print(c_names[0])
        # print(colors[c_names[0]])
        arbor,strengths = self.get_structure()

        # colors = ['r','b','g',]
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
        layers=len(arbor)
        Ns_ = Ns[::-1]
        m=max(Ns)
        c_dexes=[[] for _ in range(len(Ns))]
        den_arb = self.dendrites
        L = len(den_arb)
        M = len(den_arb[-1])
        for i,l in enumerate(arbor):
            count= 0
            row_sum = 0
            for j,b in enumerate(l):
                for k,d in enumerate(b):
                    # if i == 0:
                    #     c_dexes[i].append(k)
                    #     s = den_arb[i][j][k].s[::10]
                    #     t = np.linspace(0,1,len(s))
                    #     plt.plot(t+(layers-i-.5),s+((m/2)+(len(b)/2)-(k+1)))
                    # else:
                    # c_dexes[i].append(c_dexes[i-1][j])
                    # c_index = c_dexes[i-1][j]
                    y1=(m/2)+Ns_[i+1]/2 - 1 - (j+k) - row_sum #-((Ns_[i]/2)%2)/2
                    y2=(m/2)+Ns_[i]/2 - j - 1 
                    # print(i,j,k,row_sum)
                    s = den_arb[i][j][k].s[::10]
                    t = np.linspace(0,1,len(s))
                    if i == len(arbor)-1:
                        plt.plot(t+(layers-i-.5),s+y2)
                    else:
                        plt.plot(t+(layers-i-.5),s+y1)
                    # else:
                    #     plt.plot([layers-i-.5, layers-i+.5], [y1,y2], '--', 
                    #              color=colors[c_names[c_index]], 
                    #              linewidth=strengths[i][j][k]*5*(-1))
                    count+=1
                row_sum += len(arbor[i][j])-1 

        x_ticks=[]
        x_labels=[]
        for i,n in enumerate(Ns):
            x_labels.append(f"L{len(Ns)-(i+1)}")
            x_ticks.append(i+.5)
            if n == np.max(Ns):
                plt.plot(np.ones(n)*i+.5,np.arange(n),'ok',ms=100/(np.max(Ns)))
            elif n != 1:
                factor = 1 # make proportional
                plt.plot(np.ones(n)*i+.5,np.arange(n)*factor+(.5*np.max(Ns)-.5*n), 
                         'ok', ms=100/(np.max(Ns)))
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

