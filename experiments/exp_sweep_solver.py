import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.soen_components import input_signal, network
from sim_soens.super_node import SuperNode
from sim_soens.super_input import SuperInput
from sim_soens.super_functions import array_to_rows
from sim_soens.soen_plotting import raster_plot

def main():
    fan_in = 2
    eurekas=0
    z = np.array([0,1,4,7,8]) # z-pixel np.array
    v = np.array([1,4,3,6,8])-1 # v
    n = np.array([2,4,6,7,9])-1 # n
    letters = [z,v,n]

    window = 100
    indices = np.concatenate(letters)
    times = np.concatenate([np.ones(len(z)),
                            np.ones(len(v))*(2+window),
                            np.ones(len(n))*(2+window*2)])

    def_spikes = [indices,times]
    input = SuperInput(channels=9, type='defined', defined_spikes=def_spikes, duration=window*len(letters)+window)
    runs = 1000
    for run in range(runs):
        if fan_in == 3:
            params= {
                "N":4,
                "s_th":.42,
                "ib":1.7,
                "ib_n":1.7,
                # "tau_ni":5,
                # "tau_di":5,
                "tau_ref":1,
                }
            c=.75
            W1 = [
                [np.random.rand(3)*c],
                [np.random.rand(3)*c,np.random.rand(3)*c,np.random.rand(3)*c]
                ]
            W2 = [
                [np.random.rand(3)*c],
                [np.random.rand(3)*c,np.random.rand(3)*c,np.random.rand(3)*c]
                ]
            W3 = [
                [np.random.rand(3)*c],
                [np.random.rand(3)*c,np.random.rand(3)*c,np.random.rand(3)*c]
                ]
            # W1 =  [[array([0.5185973 , 0.08859934])], [array([0.8364571 , 1.11023959]), array([1.01050156, 0.36978657])], [array([0.22807812, 0.77334947]), array([0.44326369, 0.35082848]), array([0.17582426, 1.00827661]), array([0.28652707, 0.10167135])]]
            # W2 =  [[array([0.70313158, 0.68237736])], [array([0.17591605, 0.63316584]), array([0.37303057, 0.65283589])], [array([0.58550773, 0.60827389]), array([0.61303645, 0.92577863]), array([0.23380835, 0.08352806]), array([0.77747619, 0.41303903])]]
            # W3 =  [[array([0.54903516, 0.27614041])], [array([0.487539  , 0.72172765]), array([1.15864806, 0.72621142])], [array([0.07032772, 0.39300242]), array([0.52805035, 0.43731229]), array([0.11327929, 0.28317854]), array([1.11306904, 0.99890398])]]

        elif fan_in == 2:
            params= {
                "N":4,
                "s_th":.1,
                "ib":1.7,
                "ib_n":1.7,
                # "tau_ni":5,
                # "tau_di":5,
                "tau_ref":1,
                }
            c=1.2
            W1 = [
                [np.random.rand(2)*c],
                [np.random.rand(2)*c,np.random.rand(2)*c],
                [np.random.rand(2)*c,np.random.rand(2)*c,np.random.rand(2)*c,np.random.rand(2)*c]
                ]
            W2 = [
                [np.random.rand(2)*c],
                [np.random.rand(2)*c,np.random.rand(2)*c],
                [np.random.rand(2)*c,np.random.rand(2)*c,np.random.rand(2)*c,np.random.rand(2)*c]
                ]
            W3 = [
                [np.random.rand(2)*c],
                [np.random.rand(2)*c,np.random.rand(2)*c],
                [np.random.rand(2)*c,np.random.rand(2)*c,np.random.rand(2)*c,np.random.rand(2)*c]
                ]

        ### some winning weights
        # W1 =  [[np.array([0.61995577, 0.40528169, 0.61201274])], 
        #       [np.array([0.22448095, 0.36561921, 0.59318018]), np.array([0.10906377, 0.51099142, 0.52778415]), np.array([0.6711903 , 0.12158972, 0.12529265])]]    
        # W2 =  [[np.array([0.54367844, 0.08013411, 0.57168362])], 
        #       [np.array([0.39790794, 0.65357395, 0.06886993]), np.array([0.42829011, 0.52775269, 0.06853158]), np.array([0.71721343, 0.38347931, 0.469729  ])]]    
        # W3 =  [[np.array([0.69576919, 0.35938346, 0.44992818])], 
        #       [np.array([0.5484415 , 0.23076954, 0.58153849]), np.array([0.48881396, 0.06747522, 0.10015432]), np.array([0.542167  , 0.16863354, 0.59559225])]] 
        # W1 =  [[array([0.42550626, 0.71497102, 0.55111378])], [array([0.27774475, 0.49909493, 0.52045573]), array([0.03473457, 0.39285863, 0.47745117]), array([0.3788371 , 0.56294001, 0.48846687])]]    
        # W2 =  [[array([0.23300583, 0.49036659, 0.71775794])], [array([0.69163736, 0.01522375, 0.65868786]), array([0.25675187, 0.21823065, 0.6987337 ]), array([0.53838714, 0.73767235, 0.65813144])]]    
        # W3 =  [[array([0.1280559 , 0.47768202, 0.64164514])], [array([0.18242561, 0.05261172, 0.36735059]), array([0.56271541, 0.37259406, 0.30359262]), array([0.67250937, 0.32851443, 0.04819799])]] 
        
        
        n_1 = SuperNode(weights=W1,**params) 
        n_2 = SuperNode(weights=W2,**params) 
        n_3 = SuperNode(weights=W3,**params) 

        n_1.synaptic_layer()
        n_2.synaptic_layer()
        n_3.synaptic_layer()

        neurons = [n_1,n_2,n_3]
        # for i in range(len(input.spike_rows)):
        for i in range(len(n_1.synapse_list)):
            for n in neurons:
                n.synapse_list[i].add_input(input.signals[i])
        
        net = network(dt=0.1,tf=window*len(letters),nodes=neurons)
        net.simulate()

        ## plotting for diagnostics 
        # for i in range(3):
        #     plt.plot(net.t,net.signal[i])
        # plt.show()
        # for i in range(3):
        #     plt.plot(net.t,net.phi_r[i])
        # plt.show()
        # neurons[0].plot_custom_structure()
        # neurons[0].arbor_activity_plot()
        # neurons[1].arbor_activity_plot()
        # neurons[2].arbor_activity_plot()
        # raster_plot(net.spikes)

        rows = array_to_rows(net.spikes,3)
        counts = [ [] for _ in range(len(rows))]

        for i in range(len(neurons)):
            for j in range(len(letters)):
                winset = [j*window,j*window+window]
                # print(winset)
                frame = [rows[i][idx] for  idx,val in enumerate(rows[i]) 
                        if winset[0]<val<winset[1]]
                counts[i] .append(len(frame))
        counts = np.transpose(counts)

        maxes = [np.argmax(arr) for arr in counts]
        peaks = [np.max(arr) for arr in counts]
        if len(set(maxes))==3 and np.min(peaks)>0:
            print("EUREKA!")
            print(counts)
            print("W1 = ",W1,"\nW2 = ",W2,"\nW3 = ",W3,"\n")
            # print(net.spikes)
            neurons[0].arbor_activity_plot()
            neurons[1].arbor_activity_plot()
            neurons[2].arbor_activity_plot()
            raster_plot(net.spikes)
            eurekas+=1
            print("-----------------------------------------\n")
        else:
            print(f"Attempt {run+1} --> Try again, {len(rows[0]),len(rows[1]),len(rows[2])}")

    print(f"Percent natural success: {eurekas}/{runs} = {eurekas/runs}")


if __name__=='__main__':
    main()

