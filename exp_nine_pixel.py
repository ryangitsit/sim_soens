import numpy as np
import matplotlib.pyplot as plt

from super_library import NeuralZoo
from params import default_neuron_params
from super_input import SuperInput
from soen_sim import input_signal, synapse, neuron, network
from soen_plotting import raster_plot
from params import nine_pixel_params


def main():
    nine_pixel_params['weights']= [
            [[1.5,.9678933,.3]],
            [[0.5,0.5],[0.5,0.5],[0.5,0.5]],
            [[0.35,-0.65],[0.35,-0.65],[0.35,-0.65],[0.35,-0.65],[0.35,-0.65],[0.35,-0.65]]
        ]
    nine_pixel_params["s_th"] = 0.05
    # nine_pixel_params["tau_ref"] = 50
    nine_neuron = NeuralZoo(type="custom",**nine_pixel_params) 
    nine_neuron.plot_custom_structure()

    z = np.array([0,1,4,7,8]) # z-pixel array
    v = np.array([1,4,3,6,8])-1 # v
    n = np.array([2,4,6,7,9])-1 # n
    letters = [z,v,n]

    for let in letters:
        
        indices = let
        times = np.ones(len(indices))*20
        def_spikes = [indices,times]
        input = SuperInput(channels=9, type='defined', defined_spikes=def_spikes, duration=100)

        count = 0
        for g in nine_neuron.synapses:
            for s in g:
                for i,row in enumerate(input.spike_rows):
                    if i == int(s.name)-1:
                        s.add_input(input_signal(name = 'input_synaptic_drive', 
                        input_temporal_form = 'arbitrary_spike_train', spike_times = input.spike_rows[i]))
                        count+=1
        # print(count)

        net = network(name = 'network_under_test')
        net.add_neuron(nine_neuron.neuron)

        net.run_sim(dt = .1, tf = 100)
        net.get_recordings()
        nine_neuron.arbor_activity_plot()

        print("Number of spikes: ", len(net.spikes[0]))
        # import matplotlib.pyplot as plt
        # # spd = nine_neuron.dendrites[0][0][0].synaptic_inputs[1].phi_spd
        # dend_s = nine_neuron.dendrites[0][0][0].s
        # signal = nine_neuron.neuron.dend__nr_ni.s
        # ref = nine_neuron.neuron.dend__ref.s

        # plt.figure(figsize=(12,4))
        # plt.plot(net.t,signal, label='soma signal')
        # # plt.plot(net.spikes[1],net.spike_signals[0],'xk', label='neuron fires')
        # plt.axhline(y = nine_neuron.s_th, color = 'purple', linestyle = '--')
        # # plt.plot(net.t,spd, label='phi_spd')
        # # plt.plot(net.t,dend_s, label='dendrite signal')
        # # plt.plot(ref[::10], label='refractory signal')
        # plt.legend()
        # plt.show()

if __name__=='__main__':
    main()