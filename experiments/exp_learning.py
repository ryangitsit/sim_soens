import numpy as np
import matplotlib.pyplot as plt
import sys
from csv import writer
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.soen_plotting import raster_plot, activity_plot
from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_sim import network, dendrite, HardwareInTheLoop, synapse
from sim_soens.argparse import setup_argument_parser

print(np.random.randint(0, 100, 10))
np.random.seed(None)

def main():
    '''
    Trains neurons incrementally based on activity and error
        - Plasticity via flux offset
        - robust to ~10% noise in input data
        - training class-neurons as a mutually inhibited network speeds up convergence
    '''

    def make_letters():
        '''
        returns list of letters in pixel-array form
        '''
        # non-noisy nine-pixel letters
        letters = {
            'z': [1,1,0,
                  0,1,0,
                  0,1,1],

            'v': [1,0,1,
                  1,0,1,
                  0,1,0],

            'n': [0,1,0,
                  1,0,1,
                  1,0,1]
        }

        return letters

    def make_spikes(letter,spike_time):
        '''
        Converts a letter-array into spikes = [indices,times]
        '''
        # convert to indices of nine-channel input
        idx = np.where(np.array(letter)==1)[0]

        # all non-zero indices will spike at spike_time
        times = (np.ones(len(idx))*spike_time).astype(int)
        spikes = [idx,times]

        return spikes

    def one_pixel_noise(letter):
        '''
        Flips a random bit from letter array
        '''
        noise_idx = np.random.randint(0,len(letter))
        if letter[noise_idx] == 1:
            letter[noise_idx] = 0
        else:
            letter[noise_idx] = 1
        return letter

    def make_noise_set(letters):
        '''
        Makes the for 30 samples noisy nine-pixel dataset
            - return dict of class name keys, and pixel arrays lists 
        '''
        noise_set = {}
        for k,v in letters.items():
            noise_set[k] = [v]
        
        for i in range(9):
            for name,letter in letters.items():
                noisy = []
                for j in range(len(letter)):
                    if j==i:
                        if letter[i] == 1:
                            noisy.append(0)
                        else:
                            noisy.append(1)
                    else:
                        noisy.append(letter[j])
                noise_set[name].append(noisy)
        return noise_set


    def plot_letter(letter,name=None):
        '''
        Plots a given pixel array as nine-pixel image
        '''
        import matplotlib.cm as cm
        arrays = [[] for i in range(3)]
        count = 0
        for col in range(3):
            for row in range(3):
                arrays[col].append(letter[count])
                count+=1

        pixels = np.array(arrays).reshape(3,3)
        plt.figure()
        if name != None:
            plt.title(f"Pixel Plot - Letter {name}")
        else:
            plt.title("Pixel Plot")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(
            pixels,
            interpolation='nearest',
            cmap=cm.Blues
            )
        plt.show()

    def plot_noise_set(noise_set):
        '''
        Plots the full noisy dataset
        '''
        fig, axs = plt.subplots(3, 10,figsize=(14,4))
        
        import matplotlib.cm as cm
        for i,(name,letter) in enumerate(noise_set.items()):
            for j,pixels in enumerate(letter):
                arrays = [[] for i in range(3)]
                count = 0
                for col in range(3):
                    for row in range(3):
                        arrays[col].append(pixels[count])
                        count+=1
                pixels = np.array(arrays).reshape(3,3)
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
                axs[i][j].imshow(
                    pixels,
                    interpolation='nearest',
                    cmap=cm.Blues
                    )
                axs[2][j].set_xticks([1],[j],fontsize=18)
            axs[i][0].set_yticks([1],[list(letters.keys())[i]],fontsize=18)
        fig.suptitle("Noisy Nine-Pixel Dataset",fontsize=22)
        plt.show()

    def train_9pixel_classifier(
            letters,
            inhibition,
            elasticity,
            int_val,
            ib,
            tau,
            beta,
            s_th,
            eta,
            weights='preset',
            mutual_inhibition=True,
            c=1,
            backend='python'
            ):
        
        # Initialize dendritic weights to moderate low-spike setting
        if weights == 'preset':
            W = [
                [[.5,.5,.5]],
                [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
            ]

        # Initialize dendritic weights randomly
        elif weights == 'random':
            W = [
                    [np.random.rand(3)*c],
                    [np.random.rand(3)*c,np.random.rand(3)*c,np.random.rand(3)*c]
                    ]
            print(weights)

        # Create a node for each class, init with input params, name accordingly
        node_z = SuperNode(
            name='node_z',
            weights=W,
            ib=ib,
            ib_n=ib,
            ib_di=ib,
            tau=tau,
            tau_ni=tau,
            tau_di=tau,
            beta=2*np.pi*10**beta,
            beta_ni=2*np.pi*10**beta,
            beta_di=2*np.pi*10**beta,
            s_th=s_th,
            )
        node_v = SuperNode(
            name='node_v',
            weights=W,
            ib=ib,
            ib_n=ib,
            ib_di=ib,
            tau=tau,
            tau_ni=tau,
            tau_di=tau,
            beta=2*np.pi*10**beta,
            beta_ni=2*np.pi*10**beta,
            beta_di=2*np.pi*10**beta,
            s_th=s_th,
            )
        node_n = SuperNode(
            name='node_n',
            weights=W,
            ib=ib,
            ib_n=ib,
            ib_di=ib,
            tau=tau,
            tau_ni=tau,
            tau_di=tau,
            beta=2*np.pi*10**beta,
            beta_ni=2*np.pi*10**beta,
            beta_di=2*np.pi*10**beta,
            s_th=s_th,
            )
        
        # Add an extra synapse to each soma that takes (inhibitory) output from all other nodes
        if mutual_inhibition == True:
            syn_z = synapse(name=node_z.name+'_somatic_synapse_z')
            node_z.synapse_list.append(syn_z)
            node_z.neuron.dend_soma.add_input(syn_z,connection_strength=inhibition)

            syn_v = synapse(name=node_v.name+'_somatic_synapse_v')
            node_v.synapse_list.append(syn_v)
            node_v.neuron.dend_soma.add_input(syn_v,connection_strength=inhibition)

            syn_n = synapse(name=node_n.name+'_somatic_synapse_n')
            node_n.synapse_list.append(syn_n)
            node_n.neuron.dend_soma.add_input(syn_n,connection_strength=inhibition)

            node_z.neuron.add_output(node_v.synapse_list[-1])
            node_z.neuron.add_output(node_n.synapse_list[-1])

            node_v.neuron.add_output(node_z.synapse_list[-1])
            node_v.neuron.add_output(node_n.synapse_list[-1])

            node_n.neuron.add_output(node_z.synapse_list[-1])
            node_n.neuron.add_output(node_v.synapse_list[-1])


        # make a list of all nodes
        nodes = [node_z,node_v,node_n]
        run_times = []
        init_times = []

        # track offset trajectories with key,val=dend,[offsets] for each node
        trajects = [{},{},{}]
        # for ii,node in enumerate(nodes):
        #     count = 0
        #     for dend in node.dendrite_list:
        #         if 'ref' not in dend.name:
        #             trajects[ii][dend.name] = [0]

        # define expected spiking output for each node according to each input class            
        expect_z = [5,0,0]
        expect_v = [0,5,0]
        expect_n = [0,0,5]
        expects = [expect_z,expect_v,expect_n]

        # tracking tools
        accs = []

        trial_counter = 0

        # iterate over total dataset for some amount of runs (epochs)
        for run in range(25):

            # start with no error for each node

            total_error_z = 0
            total_error_v = 0
            total_error_n = 0
            
            # within-run success trackers
            converged = 0
            success = 0

            # itereate over ten noisy samples

            for j in range(len(letters[list(letters.keys())[0]])):

                # track predictions
                predictions = []

                # iterate over letter-classes
                for i,(name,pixels_list) in enumerate(letters.items()):

                    # make spikes from pixel arrays for the j-th sample of each class

                    defined_spikes = make_spikes(pixels_list[j],20)
                    # plot_letter(letters[name][j]) # for visualizing

                    # create input object for given spikes
                    input_ = SuperInput(type='defined',defined_spikes=defined_spikes)

                    # feed same input to all nodes
                    node_z.one_to_one(input_)
                    node_v.one_to_one(input_)
                    node_n.one_to_one(input_)

                    # run network of nodes

                    net = network(
                        sim=True,
                        dt=.1,
                        tf=100,
                        nodes=[node_z,node_v,node_n],
                        backend=backend
                        )
                    trial_counter+=1
                    # print(node_z.synapse_list[-1].name)
                    # plt.plot(node_z.synapse_list[-1].phi_spd,label='node_z synapse')
                    # plt.plot(node_n.neuron.dend_soma.s,label='node_n soma')
                    # plt.plot(node_v.neuron.dend_soma.s,label='node_v soma')
                    # plt.title(node_z.synapse_list[-1].name)
                    # plt.legend()
                    # plt.show()
                    
                    # node_z.plot_arbor_activity(net,phir=True) # for visualizing
                    # node_z.plot_neuron_activity(net=net,phir=True,ref=True,dend=False,spikes=False)

                    # run time tracking
                    run_times.append(net.run_time)
                    init_times.append(net.init_time)
                    
                    # outputs spikes of whole network
                    spikes = array_to_rows(net.spikes,3)

                    # error for each node

                    error_z = expects[0][i] - len(spikes[0])
                    error_v = expects[1][i] - len(spikes[1])
                    error_n = expects[2][i] - len(spikes[2])

                    # output spikes for each nodes
                    outputs = [len(spikes[0]),len(spikes[1]),len(spikes[2])]

                    # print(f"{j} -- {name} -- {outputs}") # for watching in real-time

                    # tracking predictions for each trial
                    predictions.append(np.argmax(outputs))


                    # tracking absolute error for each node

                    total_error_z+=np.abs(error_z)
                    total_error_v+=np.abs(error_v)
                    total_error_n+=np.abs(error_n)

                    # reset node information
                    nodes[0].neuron.spike_times=[]
                    nodes[1].neuron.spike_times=[]
                    nodes[2].neuron.spike_times=[]

                    nodes[0].neuron.spike_indices=[]
                    nodes[1].neuron.spike_indices=[]
                    nodes[2].neuron.spike_indices=[]

                    nodes[0].neuron.electroluminescence_cumulative_vec=[]
                    nodes[1].neuron.electroluminescence_cumulative_vec=[]
                    nodes[2].neuron.electroluminescence_cumulative_vec=[]

                    nodes[0].neuron.time_params=[]
                    nodes[1].neuron.time_params=[]
                    nodes[2].neuron.time_params=[]

                   
                    # init offset dicts

                    offsets_z = {}
                    offsets_v = {}
                    offsets_n = {}
                    
                    # init total offset change for trial
                    total_changes = np.zeros((3))

                    # itereate overall all dendrites

                    for ii in range(len(node_z.dendrite_list)):

                        # no updates to refracatory dendrites
                        if 'ref' not in node_z.dendrite_list[ii].name:

                            # parallel for all nodes

                            dend_z = node_z.dendrite_list[ii] 
                            dend_v = node_v.dendrite_list[ii]
                            dend_n = node_n.dendrite_list[ii]

                            dends = [dend_z,dend_v,dend_n]

                            # calculate step size according to alg 1
                            step_z = (np.mean(dend_z.s))*error_z*eta
                            step_v = (np.mean(dend_v.s))*error_v*eta
                            step_n = (np.mean(dend_n.s))*error_n*eta

                            # calculate prospective flux given past flux and step size
                            flux_z = np.mean(dend_z.phi_r) + dend_z.offset_flux
                            flux_v = np.mean(dend_v.phi_r) + dend_v.offset_flux
                            flux_n = np.mean(dend_n.phi_r) + dend_n.offset_flux

                            fluxes = [flux_z,flux_v,flux_n]
                            steps = [step_z,step_v,step_n]

                            # Ammendments to algorithm 1

                            # Bounce step backwards if rollover value (phi=0.5) would be exceeded
                            if elasticity == "True":
                                if run == 0 and ii == 0 and j ==0 and i == 0: print("elasticity = true")
                                for iii,flux in enumerate(fluxes):
                                    if flux > 0.5 or flux < -0.5:
                                        steps[iii] = -steps[iii]
                                    else:
                                        steps[iii] = steps[iii]
        
                            # Cancel update if rollover value would be exceed
                            elif elasticity == "False":
                                if run == 0 and ii == 0 and j ==0 and i == 0: print("elasticity = false")
                                for iii,flux in enumerate(fluxes):
                                    if flux > 0.5 or flux < -0.5:
                                        steps[iii] = 0

                            # all updates allowed
                            else:
                                if run == 0 and ii == 0 and j ==0 and i == 0: print("elasticity = none")

                            # make the update
                            dend_z.offset_flux += steps[0]
                            dend_v.offset_flux += steps[1]
                            dend_n.offset_flux += steps[2]

                            # accumulate total change accross all dendrites for each node
                            total_changes[0]+=step_z
                            total_changes[1]+=step_v
                            total_changes[2]+=step_n

                            # track the changing offsets for each node
                            offsets_z[dend_z.name] = dend_z.offset_flux
                            offsets_v[dend_v.name] = dend_v.offset_flux
                            offsets_n[dend_n.name] = dend_n.offset_flux

                            # append them to trajectories of each dendrite for each node
                            # trajects[0][dend_z.name].append(dend_z.offset_flux)
                            # trajects[1][dend_v.name].append(dend_v.offset_flux)
                            # trajects[2][dend_n.name].append(dend_n.offset_flux)

                        # clear old info
                        for dend in dends:
                            dend.s = []
                            dend.phi_r = []

                    # clear old info
                    del(input_)
                    del(net)
                
                if predictions == [0,1,2]:
                    # print("Correct!",predictions)
                    success += 1

            acc = np.round(success/10,2)*100
            accs.append(acc)

            print(f"  Run {run} accuracy = {acc}%")

            # if intermittent validation is true, test whole dataset w/ no updates
            if int_val == "True":
                if run == 0: print("intermittent = true")
                offsets = [offsets_z,offsets_v,offsets_n]
                early_converge = test_noise_set(letters,offsets,W,mutual_inhibition,backend)

                # if passed w/ 100% acc, converge early
                if early_converge == 1:
                    print("  Early Converge!")
                    break
            else:
                if run == 0: print("intermittent = false")

            # generally, converge if all predictions correct
            if success == 10:
                converged += 1
                if converged >= 1:
                    print(f"  Converged! (in {run} runs)")
                    # print([total_error_z,total_error_z,total_error_z])
                    # print(spikes)
                break
        # return offsets for repeatability

        offsets = [offsets_z,offsets_v,offsets_n]

        return offsets, accs, trajects, trial_counter


    def test_noise_set(noise_set,offsets,W,mutual_inhibition,backend):
        '''
        Tests given offset settings on entire noisy dataset without making updates
        '''

        weights_z = W
        weights_v = W
        weights_n = W

        node_z = SuperNode(weights=weights_z)
        node_v = SuperNode(weights=weights_v)
        node_n = SuperNode(weights=weights_n)

        if mutual_inhibition == True:
            syn_z = synapse(name='somatic_synapse_z')
            node_z.synapse_list.append(syn_z)
            node_z.neuron.dend_soma.add_input(syn_z,connection_strength=inhibition)

            syn_v = synapse(name='somatic_synapse_v')
            node_v.synapse_list.append(syn_v)
            node_v.neuron.dend_soma.add_input(syn_v,connection_strength=inhibition)

            syn_n = synapse(name='somatic_synapse_n')
            node_n.synapse_list.append(syn_n)
            node_n.neuron.dend_soma.add_input(syn_n,connection_strength=inhibition)

            node_z.neuron.add_output(node_v.synapse_list[-1])
            node_z.neuron.add_output(node_n.synapse_list[-1])

            node_v.neuron.add_output(node_z.synapse_list[-1])
            node_v.neuron.add_output(node_n.synapse_list[-1])

            node_n.neuron.add_output(node_z.synapse_list[-1])
            node_n.neuron.add_output(node_v.synapse_list[-1])

        nodes = [node_z,node_v,node_n]
        
        # give recorded offsets
        for ii,node in enumerate(nodes):
            count = 0
            for dend in node.dendrite_list:
                if 'ref' not in dend.name:
                    dend.offset_flux = offsets[ii][list(offsets[ii].keys())[count]]
                    count+=1
        
        corrects = 0
        for i, (name,pixel_list) in enumerate(noise_set.items()): 
            correct = 0
            for j,pixels in enumerate(pixel_list):
                # print('-')
                defined_spikes = make_spikes(pixels,20)
                # plot_letter(noise_set[name][j])

                input_ = SuperInput(type='defined',defined_spikes=defined_spikes)

                node_z.one_to_one(input_)
                node_v.one_to_one(input_)
                node_n.one_to_one(input_)

                net = network(
                    sim=True,
                    dt=.1,
                    tf=100,
                    nodes=[node_z,node_v,node_n],
                    backend=backend
                    )
                
                spikes = array_to_rows(net.spikes,3)

                outputs = [len(spikes[0]),len(spikes[1]),len(spikes[2])]
                # print(i,outputs)

                if np.argmax(outputs) == i:
                    correct +=1

                for node in nodes:
                    node.neuron.spike_times=[]
                    node.neuron.spike_indices=[]
                    node.neuron.electroluminescence_cumulative_vec=[]
                    node.neuron.time_params=[]
                    for dend in node.dendrite_list:
                        dend.s = []
                        dend.phi_r = []
                        
            # print(f"test {name} --> accuracy = {100*correct/len(pixel_list)}%")


            if correct==10: corrects+=1
        if corrects == 3:
            return 1
        else:
            return 0


    np.random.seed(10)

    letters = make_letters()
    names = list(letters.keys())
    noise_set = make_noise_set(letters)

    config = setup_argument_parser()

    ib         = config.ib 
    tau        = config.tau
    beta       = config.beta
    s_th       = config.s_th
    eta        = config.eta
    elasticity = config.elast
    int_val    = config.valid
    inhibition = config.inhibit
    backend    = config.backend

    regimes = ['Elastic', 'Inelastic', 'Unbounded']
    if elasticity == "True":
        regime = regimes[0]
    elif elasticity == "False":
        regime = regimes[1]
    else:
        regime = regimes[2]

    if int_val == "True":
        converge_type = 'Intermittent'
    else:
        converge_type = 'Update'

    path = f"results/{config.exp_name}/"
    sub_name = f"{regime}_{converge_type}_{ib}_{tau}_{beta}_{s_th}_{eta}"

    print(sub_name)

    offsets, accs, trajects, conv_time = train_9pixel_classifier(
        noise_set,
        inhibition,
        elasticity,
        int_val,
        ib,
        tau,
        beta,
        s_th,
        eta,
        backend=backend,
        mutual_inhibition=False
        )


    # picklit(
    #     accs,
    #     path,
    #     f"{sub_name}_accs"
    #     )

    # picklit(
    #     trajects,
    #     path,
    #     f"{sub_name}_trajects"
    #     )

    # plt.figure(figsize=(8,4))
    # plt.plot(accs)
    # plt.title(f"Learning Accuracy for {regime} Noisy 9-Pixel Classifier",fontsize=16)
    # plt.xlabel("Total Iterations",fontsize=14)
    # plt.ylabel("Percent Accuracy",fontsize=14)
    # plt.subplots_adjust(bottom=.15)
    # plt.savefig(path+sub_name+'_accs_plot.png')
    # plt.close()

    # plt.style.use('seaborn-v0_8-muted')
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # for i,traject in enumerate(trajects):
    #     plt.figure(figsize=(8,4))
    #     count1=0
    #     count2=0
    #     for name,offset in reversed(traject.items()):
    #         if 'soma' in name:
    #             name = 'soma'
    #             converge_length = len(offset)
    #             # plt.plot(offset,color=colors[i],label=name,linewidth=4)
    #             plt.plot(offset,color=colors[0],label=name,linewidth=4)
    #         elif 'lay1' in name:
    #             col = colors[1]

    #             if count1 == 0:
    #                 plt.plot(offset,'--',color=col,linewidth=2,label='Layer 1')
    #             else:
    #                 # plt.plot(offset,color=colors[0],label=name,linewidth=3)
    #                 plt.plot(offset,'--',color=col,linewidth=2)
    #             count1+=1

    #         elif 'lay2' in name:
    #             col = colors[2]
    #             if count2 == 0:
    #                 plt.plot(offset,':',color=col,label='Layer 2',linewidth=1)
    #             else:
    #                 plt.plot(offset,':',color=col,linewidth=1)
    #             # plt.plot(offset,color=colors[4],label=name)
    #             count2+=1

    #     plt.title(f"Noisy 9-Pixel Classifier {regime} {converge_type} Convergence - {names[i]}",fontsize=16)
    #     plt.xlabel("Total Iterations",fontsize=14)
    #     plt.ylabel("Flux Offset",fontsize=14)
    #     plt.subplots_adjust(bottom=.15)
    #     plt.legend()
    #     plt.savefig(path+sub_name+f'_offsets_{names[i]}_plot.png')
    #     plt.close()

    List = [
        regime,
        converge_type,
        ib,
        tau,
        beta,
        s_th,
        eta,
        conv_time,
    ]
    with open(path+'pixels.csv', 'a') as f_object:
    
        writer_object = writer(f_object)
    
        writer_object.writerow(List)

        f_object.close()

    # test_noise_set(noise_set,offsets)

if __name__=='__main__':
    main()
