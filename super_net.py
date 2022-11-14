#%%
import numpy as np

from super_input import SuperInput

from _util import physical_constants, set_plot_params, index_finder
from _util__soen import dend_load_rate_array, dend_load_arrays_thresholds_saturations
from soen_sim import input_signal, synapse, neuron, network
from soen_sim_lib__common_components__simple_gates import common_dendrite, common_synapse, common_neuron

"""
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



# p = physical_constants()

# # plt.close('all')
# fig_case = 'presentation' # 'publication' 'presentation'
# fig_size = set_plot_params(fig_case, display_fonts = False)


# loops_present = 'ri' # 'ri' 'rtti'

# # ib__list, phi_r__array, i_di__array, r_fq__array, params_imported, _ = dend_load_rate_array('default_{}'.format(loops_present))
# # ib__list__ri, phi_r__array__ri, i_di__array__ri, r_fq__array__ri, phi_th_plus__vec__ri, phi_th_minus__vec__ri, s_max_plus__vec__ri, s_max_minus__vec__ri, s_max_plus__array__ri, s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')



#%%

class SuperNet:
    '''
    Organizes a system and structure of loop neurons
    '''

    def __init__(self,**entries):
        net_args = {
            # "N":100,
            "ns": 100,
            "connectivity": "random",
            "in_connect": "ordered",
            "recurrence": None,
            # "sim": 500,
            "input_p": 1,
            "reservoir_p":0.2,

            "beta_di": 2*np.pi*1e2,
            "tau_di": [1,2], #[900,1100],
            "ib": 9, # int 0-9 to draw from ib__list__ri[i] list
            # "s_max":,
            # "phi_th":,
            "ib_n": 9, # int 0-9 to draw from ib__list__ri[i] list
            "s_th_factor_n": 0.1,
            # "s_max_n":,
            # "phi_th_n":,
            "beta_ni": 2*np.pi*1e3,
            "tau_ni": 50,

            "w_sd": [2.5],
            "w_sid": [2.5], # two numbers for rand float range or single value for consant
            "w_dn": [.75], # two numbers for rand float range or single value for consant
            "norm_dn": 1,
            "norm_sd": 1,

            "beta_ni": 2*np.pi*1e3,
            "tau_ni": 50,
            "ib_ref": 8, # int 0-9 to draw from ib__list__ri[i] list
            "beta_ref": 2*np.pi*1e4,
            "tau_ref": 500,
            "dt_soen": 1, # simulation time-step
            "_t_on": 5,

        }
        self.N = 10
        self.duration = 100
        self.name = 'Super_Net'
        self.dend_type = 'default_ri',
        self.__dict__.update(net_args)
        # self.__dict__.update(entries['params'])
        self.__dict__.update(entries)
        self.ib__list__ri, self.phi_r__array__ri, self.i_di__array__ri, self.r_fq__array__ri, self.phi_th_plus__vec__ri, self.phi_th_minus__vec__ri, self.s_max_plus__vec__ri, self.s_max_minus__vec__ri, self.s_max_plus__array__ri, self.s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')
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
        ns = self.ns

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
            spikes[1].append(spike_t)
            count+=1
        spikes[0] =np.concatenate(spikes[0])
        spikes[1] = np.concatenate(spikes[1])/1000
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






# input_single = SuperInput(channels=50, type='random', total_spikes=100, duration=100)
# # input_MNIST = SuperInput(type='MNIST', index=0, slow_down=100, duration=1000)
# single_neuron = SuperNet(N=50,duration=100)#,params=net_args) #dendrites,synapses
# print(single_neuron.N)
# single_neuron.connect_input(input_single)
# #%%
# single_neuron.run()
# single_neuron.record(['spikes'])
# spikes = single_neuron.spikes
# #%%

# from _plotting__soen import raster_plot
# raster_plot(spikes,duration=100,input=input_single.spike_arrays)

#%%


###########################################################################
###########################################################################
###########################################################################

# make single line
# super_input = SuperInput(**input_args)
# mnist_data, mnist_indices, mnist_spikes = super_input.MNIST()
# spikes = [mnist_indices[0],mnist_spikes[0]]
# input = super_input.array_to_rows(spikes)

# # make single line
# super_net = SuperNet(dend_load_arrays_thresholds_saturations('default_ri'),**net_args)
# super_net.param_setup()
# super_net.make_input_signal(input)
# super_net.make_neurons()
# super_net.make_net()

# # Add preemptory monitor statement
# spiked, S = super_net.run()
# print("spikes = ", len(spiked[0]),"\n\n")

# # Call this from a separate plotting file/class
# # super_net.raster_plot(spiked)
# input_spikes = super_input.rows_to_array(input)
# super_net.raster_input_plot(spiked,input_spikes)

# labels = ["zero","zero","zero","one","one","one","two","two","two"]
# for i,pattern in enumerate(labels):

#     print("pattern index: ",i)
#     replica = str(i%3)

#     spikes = [mnist_indices[i],mnist_spikes[i]]
#     input = super_input.array_to_rows(spikes)

#     super_net = SuperNet(dend_load_arrays_thresholds_saturations('default_ri'),**net_args)
#     super_net.param_setup()
#     super_net.make_input_signal(input)

#     super_net.make_neurons()
#     super_net.make_net()

#     spiked, S = super_net.run()
#     print("spikes = ", len(spiked[0]),"\n\n")

#     # super_net.raster_plot(spiked)
#     # input_spikes = super_input.rows_to_array(input)
#     # super_net.raster_input_plot(spiked,input_spikes)
#     dir = 'test_cases'
#     super_net.spks_to_txt(spiked,3,dir,f"spikes_{pattern}_{replica}")

#%%

print("\n\n")


### Previous sweep
# arg-parser?

# # input = super_input.gen_rand_input(10,25)
# labels = ["zero","zero","zero","one","one","one","two","two","two"]
# # labels = ["zero"] 
# count = 0
# for in_p in np.arange(0.2,1,.2):
#     print(in_p)
#     for res_p in np.arange(0.2,1,.2):
#         for sd in [-1,0]:
#             for dn in [-1,0]:
#                 for tau_spread in np.arange(50,350,50):
#                     net_args["input_p"] = in_p
#                     net_args["reservoir_p"] = res_p
#                     net_args["w_sd"][0] = sd
#                     net_args["w_dn"][0] = dn
#                     net_args["tau_di"] = [1000-int(tau_spread/2),1000+int(tau_spread/2)]

#                     dir = f"spks_in{in_p[:3]}_res{res_p[:3]}_sd{sd}_dn{dn}_taudis{tau_spread}"
#                     print(dir)
#                     for i,pattern in enumerate(labels):

#                         print("pattern index: ",i)
#                         replica = str(i%3)

#                         spikes = [mnist_indices[i],mnist_spikes[i]]
#                         input = super_input.array_to_rows(spikes)

#                         super_net = SuperNet(dend_load_arrays_thresholds_saturations('default_ri'),**net_args)
#                         super_net.param_setup()
#                         super_net.make_input_signal(input)

#                         super_net.make_neurons()
#                         super_net.make_net()

#                         spiked, S = super_net.run()
#                         print("spikes = ", len(spiked[0]),"\n\n")

#                         # super_net.raster_plot(spiked)
#                         # input_spikes = super_input.rows_to_array(input)
#                         # super_net.raster_input_plot(spiked,input_spikes)

#                         super_net.spks_to_txt(spiked,3,dir,f"spikes_{pattern}_{replica}")

print("Complete!")
    
#%%


# import pickle 
# with open('test.npy', 'wb') as f:
#     np.save(f, spikes)
# with open('test.npy', 'rb') as f:
#     a = np.load(f,allow_pickle=True)

# for i in a:
#     print(i)

# %%
