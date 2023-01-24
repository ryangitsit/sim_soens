import numpy as np
import matplotlib.pyplot as plt

from soen_sim import input_signal, network

from super_library import NeuralZoo
from super_input import SuperInput
from super_functions import array_to_rows, spks_to_txt, picklit, picklin

from soen_plotting import raster_plot
import random
from super_argparse import setup_argument_parser

def main():

    args = setup_argument_parser()
    run = args.run
    runs = args.runs

    tile_time = 10
    classes = [0,1,2]
    window = tile_time*36*3
    # eurekas=0

    # input = SuperInput(type='saccade_MNIST', channels=36, tile_time=tile_time)
    # picklit(input,"results","saccade_mnist_10")
    # spks_to_txt(input.spike_arrays,36,10,"single_layer/","input")

    input = picklin("results","saccade_mnist_10")

    # print("input generated")
    # raster_plot(input.spike_arrays)

    params= {
        "N":4,
        # "s_th":.3,
        "s_th":.4,
        # "ib":1.7,
        # "ib_n":1.7,
        "ib":1.802395858835221,
        "ib_n":1.802395858835221,
        # "tau_ni":5,
        # "tau_di":5,
        "tau_ref":75,
        }

    # np.random.seed(None)
    def rnd_pm(vals):
        c=.6
        arr = np.random.rand(vals)*c*[-1,1][random.randrange(2)]
        # print(arr)
        return arr


    W1 =  [[np.array([-0.25732494, -0.34125771, -0.26904125])], [np.array([0.0276439 , 0.44972837, 0.57901368]), np.array([0.04016443, 0.51306481, 0.5256298 ]), np.array([-0.09819564, -0.08410133, -0.52008145])], [np.array([0.12510855, 0.14525595, 0.45678976, 0.39199201]), np.array([0.27374585, 0.40406476, 0.19878509, 0.31670493]), np.array([0.04761957, 0.5720136 , 0.2011468 , 0.56989305]), np.array([0.01356939, 0.58119426, 0.09262238, 0.55506909]), np.array([-0.4083557 , -0.49557829, -0.28030526, -0.35716912]), np.array([0.36952501, 0.07526286, 0.09328194, 0.11898012]), np.array([0.25013909, 0.36365812, 0.14750034, 0.47218095]), np.array([0.25908738, 0.1646264 , 0.27402361, 0.35596549]), np.array([0.30623764, 0.19744562, 0.02249533, 0.19985171])]]
    W2 =  [[np.array([0.3882577 , 0.36188725, 0.20522705])], [np.array([-0.42916033, -0.22117618, -0.38283165]), np.array([-0.23988397, -0.45551301, -0.09830606]), np.array([-0.52848658, -0.25843576, -0.06741026])], [np.array([-0.03896332, -0.34285907, -0.2234364 , -0.02197801]), np.array([0.49128107, 0.36724272, 0.01882161, 0.24881669]), np.array([-0.5606351 , -0.4256968 , -0.48772881, -0.22681596]), np.array([0.46056542, 0.24982304, 0.19253616, 0.48885291]), np.array([-0.04620455, -0.42103049, -0.01011089, -0.33042036]), np.array([0.01350135, 0.49978422, 0.05839576, 0.25534382]), np.array([0.03565049, 0.02002225, 0.37511452, 0.18699317]), np.array([-0.18930888, -0.12689792, -0.46455288, -0.2725641 ]), np.array([0.51625816, 0.11798277, 0.5914419 , 0.00813793])]]
    W3 =  [[np.array([-0.52752977, -0.41900608, -0.04629707])], [np.array([-0.02364441, -0.33915702, -0.26994239]), np.array([-0.50484561, -0.55917022, -0.59032025]), np.array([0.34589999, 0.22490057, 0.29812211])], [np.array([0.54905814, 0.50587667, 0.3310234 , 0.34638242]), np.array([0.1035894 , 0.16388841, 0.11186747, 0.03628561]), np.array([-0.46472384, -0.31730277, -0.47900104, -0.23682077]), np.array([-0.36292224, -0.39423829, -0.0749922 , -0.05216149]), np.array([-0.58073368, -0.56263935, -0.14850184, -0.3450179 ]), np.array([-0.2205613 , -0.46345103, -0.55508483, -0.39850353]), np.array([-0.23409645, -0.31872053, -0.24793407, -0.27027444]), np.array([-0.23297006, -0.24019577, -0.01469594, -0.55198111]), np.array([-0.09635262, -0.09562517, -0.23161685, -0.52053636])]]

    n_1 = NeuralZoo(type="custom",weights=W1,**params) 
    n_2 = NeuralZoo(type="custom",weights=W2,**params) 
    n_3 = NeuralZoo(type="custom",weights=W3,**params) 

    n_1.synaptic_layer()
    n_2.synaptic_layer()
    n_3.synaptic_layer()

    neurons = [n_1,n_2,n_3]
    # for i in range(len(input.spike_rows)):
    for i in range(len(n_1.synapse_list)):
        for n in neurons:
            n.synapse_list[i].add_input(input.signals[i])
    # print('tf = ',np.max(input.spike_arrays[1])+100)
    # print('total input spikes = ', len(input.spike_arrays[1]))
    net = network(dt=0.1,tf=np.max(input.spike_arrays[1])+360,nodes=neurons)
    net.simulate()

    # neurons[0].plot_custom_structure()
    # neurons[0].arbor_activity_plot()
    # neurons[1].arbor_activity_plot()
    # neurons[2].arbor_activity_plot()
    # raster_plot(net.spikes)

    rows = array_to_rows(net.spikes,3)
    counts = [ [] for _ in range(len(rows))]

    for i in range(len(neurons)):
        for j in range(len(classes)):
            # if i == len(neurons)-1 and j ==len(classes)-1:
            #     winset = [j*window,j*window+window+36*tile_time]
            # else:
            winset = [j*window,j*window+window]
            # print(winset)
            frame = [rows[i][idx] for  idx,val in enumerate(rows[i]) 
                    if winset[0]<val<winset[1]]
            counts[i] .append(len(frame))
    counts = np.transpose(counts)
    # print(counts)
    maxes = [np.argmax(arr) for arr in counts]
    peaks = [np.max(arr) for arr in counts]

    if len(set(maxes))==3 and np.min(peaks)>0:
        print(f"Attempt {run} --> EUREKA!")
        print(counts)
        print("W1 = ",W1,"\nW2 = ",W2,"\nW3 = ",W3,"\n")
        print(net.spikes)
        raster_plot(net.spikes)
        # path1 = 'results/single_layer_symm/{run}_act1.png'
        # path2 = 'results/single_layer_symm/{run}_act2.png'
        # path3 = 'results/single_layer_symm/{run}_act3.png'
        neurons[0].arbor_activity_plot()
        neurons[1].arbor_activity_plot()
        neurons[2].arbor_activity_plot()

        # with open(f'results/single_layer_symm/winner_weights_{run}.txt', 'w') as f:
        #     f.write("W1=")
        #     f.write(str(W1))
        #     f.write("\n")
        #     f.write("W2=")
        #     f.write(str(W2))
        #     f.write("\n")
        #     f.write("W3=")
        #     f.write(str(W3))
        # spks_to_txt(net.spikes,3,10,"single_layer_symm",f"winner_spikes_{run}")

        # eurekas+=1
        print("-----------------------------------------\n")
    else:
        print(f"Attempt {run} --> Try again, {len(rows[0]),len(rows[1]),len(rows[2])}")

    # print(f"Percent natural success: {eurekas}/{runs} = {eurekas/runs}")
if __name__=='__main__':
    main()

