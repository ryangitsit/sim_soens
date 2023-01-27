import numpy as np
# import matplotlib.pyplot as plt

from soen_sim import input_signal, network
from soen_component_library import common_synapse

from super_library import NeuralZoo
# from super_input import SuperInput
from super_functions import array_to_rows, spks_to_txt, picklit, picklin,save_dict

# from soen_plotting import raster_plot
import random
from super_argparse import setup_argument_parser

def main():

    args = setup_argument_parser()
    run = args.run + 21
    runs = args.runs
    form = args.form
    beta = 2*np.pi*10**(args.beta)
    tau = args.tau
    tau_ref = args.tau_ref
    inhibit = -(1/args.inhibit)
    path = f'results/single_inhibit/{args.dir}/'

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
        "s_th":.5,
        # "ib":1.7,
        # "ib_n":1.7,
        "ib":1.802395858835221,
        "ib_n":1.802395858835221,
        "beta_di":beta,
        "beta_ni":beta,
        "tau_ni":tau,
        "tau_di":tau,
        "tau_ref":150,
        "c":0.6,
        "range":[-1,1],
        "inhibtion":inhibit,
        }

    save_dict(params,path,'params')

    # np.random.seed(None)
    def rnd_pm(vals):
        c=.6
        arr = np.random.rand(vals)*params['c']*params['range'][random.randrange(2)]
        # print(arr)
        return arr

    
    W1 = [
        [rnd_pm(3)],
        [rnd_pm(3),rnd_pm(3),rnd_pm(3)],
        [rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4)]
        ]
    W2 = [
        [rnd_pm(3)],
        [rnd_pm(3),rnd_pm(3),rnd_pm(3)],
        [rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4)]
        ]
    W3 = [
        [rnd_pm(3)],
        [rnd_pm(3),rnd_pm(3),rnd_pm(3)],
        [rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4),rnd_pm(4)]
        ]


    # The lost winner
    # W1 =  [[np.array([0.10027454, 0.58133644, 0.55944452])], [np.array([0.24048415, 0.68690423, 0.26635529]), np.array([0.43997153, 0.22140561, 0.60989458]), np.array([0.84197269, 0.75627611, 0.05974083])], [np.array([0.84229829, 0.10177496, 0.33385362, 0.00593317]), np.array([0.25905758, 0.36695748, 0.33077299, 0.67711857]), np.array([0.75972404, 0.76973308, 0.72250814, 0.39805298]), np.array([0.31977279, 0.56224649, 0.27373613, 0.26746589]), np.array([0.24285249, 0.06188361, 0.6210575 , 0.09703051]), np.array([0.37902298, 0.6922418 , 0.05391373, 0.60615722]), np.array([0.5249293 , 0.0367024 , 0.49675437, 0.40722666]), np.array([0.6217751 , 0.55227423, 0.41749913, 0.77550667]), np.array([0.59720308, 0.53510703, 0.36952971, 0.22036814])]]
    # W2 =  [[np.array([0.78851345, 0.27407871, 0.30803972])], [np.array([0.2549654 , 0.67385715, 0.57587251]), np.array([0.559625  , 0.44622867, 0.60465558]), np.array([0.04828061, 0.27477089, 0.69585055])], [np.array([0.11005697, 0.19632138, 0.39042952, 0.12591339]), np.array([0.16299855, 0.33534171, 0.1850659 , 0.45389967]), np.array([0.06385376, 0.51709817, 0.82845702, 0.01157012]), np.array([0.23659099, 0.31359119, 0.75644638, 0.4096279 ]), np.array([0.20260746, 0.14851041, 0.051098  , 0.05830404]), np.array([0.62678092, 0.75935016, 0.03536966, 0.49896926]), np.array([0.037     , 0.66069881, 0.48698776, 0.35310226]), np.array([0.55448332, 0.2248576 , 0.53403215, 0.33103462]), np.array([0.19723041, 0.06784297, 0.63379481, 0.78139109])]]
    # W3 =  [[np.array([0.15910554, 0.34525008, 0.82845176])], [np.array([0.6074373 , 0.74592134, 0.4729844 ]), np.array([0.64526965, 0.30684274, 0.72314327]), np.array([0.54478218, 0.68085802, 0.00556301])], [np.array([0.24749122, 0.55304051, 0.74067455, 0.82094564]), np.array([0.61945409, 0.45426127, 0.38886517, 0.34920898]), np.array([0.46183024, 0.82483073, 0.48337518, 0.54794052]), np.array([0.40051994, 0.2327313 , 0.83357001, 0.23183207]), np.array([0.40931346, 0.42232051, 0.13605793, 0.20694464]), np.array([0.57282178, 0.73085263, 0.16810304, 0.30714569]), np.array([0.64450811, 0.72173314, 0.57867988, 0.82859903]), np.array([0.51109458, 0.67535165, 0.81185724, 0.62344794]), np.array([0.24820321, 0.1297074 , 0.03402139, 0.44610799])]]

    n_1 = NeuralZoo(type="custom",weights=W1,**params) 
    n_2 = NeuralZoo(type="custom",weights=W2,**params) 
    n_3 = NeuralZoo(type="custom",weights=W3,**params) 

    n_1.synaptic_layer()
    n_2.synaptic_layer()
    n_3.synaptic_layer()

    if form == 'standalone':
        pass
    elif form == 'WTA':
        syn11=common_synapse(f'soma_synapse_1{n_1.neuron.name}')
        n_1.neuron.dend__nr_ni.add_input(syn11,connection_strength=inhibit)
        n_1.synapse_list.append(syn11)

        syn12=common_synapse(f'soma_synapse_2{n_1.neuron.name}')
        n_1.neuron.dend__nr_ni.add_input(syn12,connection_strength=inhibit)
        n_1.synapse_list.append(syn12)

        syn21=common_synapse(f'soma_synapse_1{n_2.neuron.name}')
        n_2.neuron.dend__nr_ni.add_input(syn21,connection_strength=inhibit)
        n_2.synapse_list.append(syn21)

        syn22=common_synapse(f'soma_synapse_2{n_2.neuron.name}')
        n_2.neuron.dend__nr_ni.add_input(syn22,connection_strength=inhibit)
        n_2.synapse_list.append(syn22)

        syn31=common_synapse(f'soma_synapse_1{n_3.neuron.name}')
        n_3.neuron.dend__nr_ni.add_input(syn31,connection_strength=inhibit)
        n_3.synapse_list.append(syn31)

        syn32=common_synapse(f'soma_synapse_2{n_3.neuron.name}')
        n_3.neuron.dend__nr_ni.add_input(syn32,connection_strength=inhibit)
        n_3.synapse_list.append(syn32)

    
        n_1.neuron.add_output(n_2.synapse_list[-1])
        n_1.neuron.add_output(n_3.synapse_list[-1])

        n_2.neuron.add_output(n_1.synapse_list[-1])
        n_2.neuron.add_output(n_3.synapse_list[-2])

        n_3.neuron.add_output(n_1.synapse_list[-2])
        n_3.neuron.add_output(n_2.synapse_list[-2])
    # print(len(n_3.synapse_list))  
    # print(n_2.synapse_list[-1].name)  
    # pass

    neurons = [n_1,n_2,n_3]
    # for i in range(len(input.spike_rows)):
    for i in range(len(n_1.synapse_list)-2):
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
        print(f"{path} - Attempt {run} --> EUREKA!")
        print(counts)
        print("W1 = ",W1,"\nW2 = ",W2,"\nW3 = ",W3,"\n")
        # print(net.spikes)
        # raster_plot(net.spikes)
        # path1 = 'results/single_layer_symm/{run}_act1.png'
        # path2 = 'results/single_layer_symm/{run}_act2.png'
        # path3 = 'results/single_layer_symm/{run}_act3.png'
        # neurons[0].arbor_activity_plot(path=path1)
        # neurons[1].arbor_activity_plot(path=path2)
        # neurons[2].arbor_activity_plot(path=path3)

        with open(f'{path}winner_weights_{run}.txt', 'w') as f:
            f.write("W1=")
            f.write(str(W1))
            f.write("\n")
            f.write("W2=")
            f.write(str(W2))
            f.write("\n")
            f.write("W3=")
            f.write(str(W3))
        spks_to_txt(net.spikes,3,10,args.dir,f"winner_spikes_{run}")

        # eurekas+=1
        print("-----------------------------------------\n")
    else:
        print(f"{path} - Attempt {run} --> Try again, {len(rows[0]),len(rows[1]),len(rows[2])}")

    # print(f"Percent natural success: {eurekas}/{runs} = {eurekas/runs}")
if __name__=='__main__':
    main()

