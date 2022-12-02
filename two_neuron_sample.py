#%%
import numpy as np
from matplotlib import pyplot as plt

from _util import physical_constants, set_plot_params, index_finder
from _util__soen import dend_load_rate_array, dend_load_arrays_thresholds_saturations
from soen_sim import input_signal, synapse, neuron, network
from soen_sim_lib__common_components__simple_gates import common_dendrite, common_synapse, common_neuron

p = physical_constants()

# plt.close('all')
fig_case = 'presentation' # 'publication' 'presentation'
fig_size = set_plot_params(fig_case, display_fonts = False)


#%% specify loops present

loops_present = 'ri' # 'ri' 'rtti'

#%% load rate array to be sure to use an entry from ib__list

# ib__list, phi_r__array, i_di__array, r_fq__array, params_imported, _ = dend_load_rate_array('default_{}'.format(loops_present))
ib__list__ri, phi_r__array__ri, i_di__array__ri, r_fq__array__ri, phi_th_plus__vec__ri, phi_th_minus__vec__ri, s_max_plus__vec__ri, s_max_minus__vec__ri, s_max_plus__array__ri, s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')

#%% input parameters

# dendrites
beta_di_d1 = 2*np.pi*1e3
tau_di_d1 = 1000 # ns
ib_d1 = ib__list__ri[9]
beta_di_d2 = 2*np.pi*1e3
tau_di_d2 = 1000 # ns
ib_d2 = ib__list__ri[9]

s_max_d1 = s_max_plus__vec__ri[index_finder(ib_d1,ib__list__ri[:])]
phi_th_d1 = phi_th_plus__vec__ri[index_finder(ib_d1,ib__list__ri[:])]
s_max_d2 = s_max_plus__vec__ri[index_finder(ib_d1,ib__list__ri[:])]
phi_th_d2 = phi_th_plus__vec__ri[index_finder(ib_d2,ib__list__ri[:])]

# neurons
ib_n1 = ib__list__ri[9]
ib_n2 = ib__list__ri[7]
s_th_factor_n1 = 0.1
s_th_factor_n2 = 0.3

s_max_n1 = s_max_plus__vec__ri[index_finder(ib_n1,ib__list__ri[:])]
phi_th_n1 = phi_th_plus__vec__ri[index_finder(ib_n1,ib__list__ri[:])]
s_max_n2 = s_max_plus__vec__ri[index_finder(ib_n1,ib__list__ri[:])]
phi_th_n2 = phi_th_plus__vec__ri[index_finder(ib_n2,ib__list__ri[:])]
ib_ref = ib__list__ri[7]

beta_ni = 2*np.pi*1e2 # 2*np.pi*np.asarray([1e2,1e3,1e4,1e5])
tau_ni = 50 # np.asarray([10,50,250,1250]) # ns

# connections
connection_strength__syn_in_to_dend_1 = 0.9
connection_strength__syn_1_to_dend_1 = 0.9
connection_strength__syn_2_to_dend_2 = 1
connection_strength__dend_1_to_neu_1 = 0.5/s_max_d1
connection_strength__dend_2_to_neu_2 = 0.5/s_max_d2

# refraction loop
beta_ref = 2*np.pi*1e2
tau_ref = 50

# time
dt_soen = 0.1 # ns
_t_on = 5
time_first_spike = 5 # ns

# burst params
num_in_burst = 1 # np.unique(np.round(np.geomspace(1,100,24)))
burst_frequency = 10/tau_di_d1 # np.asarray([0.5,1,2,4,8])/tau_di

isi = 1/burst_frequency
spike_times = np.linspace( time_first_spike, time_first_spike+(num_in_burst-1)*isi, int(num_in_burst) )
            
#%% assemble components and network

# establish input
input_1 = input_signal(name = 'input_synaptic_drive', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
            
# establish synapses
synapse_in = common_synapse(1)
synapse_1 = common_synapse(2)
synapse_2 = common_synapse(3)

# add input to synapse
synapse_in.add_input(input_1)

# establish dendrites
dendrite_1 = common_dendrite(1, 'ri', beta_di_d1, tau_di_d1, ib_d1)
dendrite_2 = common_dendrite(2, 'ri', beta_di_d2, tau_di_d2, ib_d2)

# add synapses to dendrites
dendrite_1.add_input(synapse_in, connection_strength = connection_strength__syn_in_to_dend_1)
dendrite_1.add_input(synapse_1, connection_strength = connection_strength__syn_1_to_dend_1)
dendrite_2.add_input(synapse_2, connection_strength = connection_strength__syn_2_to_dend_2)

# establish neurons
neuron_1 = common_neuron(1, 'ri', beta_ni, tau_ni, ib_n1, s_th_factor_n1*s_max_n1, beta_ref, tau_ref, ib_ref)
neuron_2 = common_neuron(2, 'ri', beta_ni, tau_ni, ib_n2, s_th_factor_n2*s_max_n2, beta_ref, tau_ref, ib_ref)
# print("-----",s_th_factor_n1*s_max_n1)
# print("-----",s_th_factor_n2*s_max_n2)
# add dendrites to neurons
neuron_1.add_input(dendrite_1, connection_strength = connection_strength__dend_1_to_neu_1)
neuron_2.add_input(dendrite_2, connection_strength = connection_strength__dend_2_to_neu_2)

# add output synapses to neurons
neuron_1.add_output(synapse_2)
neuron_2.add_output(synapse_1)

# create network
net = network(name = 'network_under_test')

# add neurons to network
net.add_neuron(neuron_1)
net.add_neuron(neuron_2)

#%% run
net.run_sim(dt = dt_soen, tf = input_1.spike_times[-1] + np.max([dendrite_1.tau_di,dendrite_2.tau_di]))
print(net.neurons[2].dend__ref.synaptic_inputs)
#%% plot
neuron_1.plot_simple = True
neuron_2.plot_simple = True
net.plot()
plt.show()
# %%
