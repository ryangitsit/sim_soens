import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.soen_plotting import raster_plot, activity_plot
from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_sim import network, dendrite, HardwareInTheLoop, synapse

print(np.random.randint(0, 100, 10))
np.random.seed(None)

def main():
    '''
    Trains neurons incrementally based on activity and error
        - Plasticity via flux offset
        - robust to ~10% noise in input data
        - training neurons as a mutually inhibited network speeds up convergence
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
        Converts pixel array into spikes (indices and times)
        '''
        # convert to indices of nine-channel input
        idx = np.where(np.array(letter)==1)[0]

        # all non-zero indices will spike at spike_time
        times = (np.ones(len(idx))*spike_time).astype(int)
        spikes = [idx,times]

        return spikes

    def one_pixel_noise(letter):
        '''
        Shuffles one random pixel of pixel-array
        '''
        noise_idx = np.random.randint(0,len(letter))
        if letter[noise_idx] == 1:
            letter[noise_idx] = 0
        else:
            letter[noise_idx] = 1
        return letter

    def make_noise_set(letters):
        '''
        Makes the full noisy pixel dataset (30 pixel arrays in total)
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
        plots pixels!
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

    letters = make_letters()
    # print(letters)
    noise_set = make_noise_set(letters)
    # print(noise_set)

    def plot_noise_set(noise_set):
        '''
        plots fully noisy pixel dataset
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
                # plot_letter(pixels,name=name)
                axs[2][j].set_xticks([1],[j],fontsize=18)
            axs[i][0].set_yticks([1],[list(letters.keys())[i]],fontsize=18)
        # plt.yticks([0,1,2],['z','v','n'],fontsize=12)
        fig.suptitle("Noisy Nine-Pixel Dataset",fontsize=22)
        # plt.xlabel("Sample")
        # plt.ylabel("Class")
        # fig.supxlabel('common_x',fontsize=18)
        # fig.supylabel('common_y',fontsize=18)
        # axs.yaxis.set_label_coords(-.1, .1)
        # plt.subplots_adjust(left=-1.8)
        # plt.subplots_adjust(bottom=.15)
        plt.show()
    plot_noise_set(noise_set)

    np.random.seed(10)

    def train_9pixel_classifier(
            letters,all_spikes,learning_rate,inhibition,elasticity,int_val
            ):
        '''Trains 3-Neuron WTA Network on Full Noisy 9-Pixel Dataset'''

        # moderate weight initialization (handpicked for dataset)
        weights = [
            [[.5,.5,.5]],
            [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
        ]

        # make a node to associate with each class
        node_z = SuperNode(weights=weights)
        node_v = SuperNode(weights=weights)
        node_n = SuperNode(weights=weights)

        # add extra synapse on soma for each node and wire w/ mutual inhibition
        syn_z = synapse(name='somatic_synapse_z')
        node_z.synapse_list.append(syn_z)
        node_z.neuron.dend_soma.add_input(syn_z,connection_strength=inhibition)
        node_z.neuron.add_output(node_v.synapse_list[-1])
        node_z.neuron.add_output(node_n.synapse_list[-1])

        syn_v = synapse(name='somatic_synapse_v')
        node_v.synapse_list.append(syn_v)
        node_v.neuron.dend_soma.add_input(syn_v,connection_strength=inhibition)
        node_v.neuron.add_output(node_z.synapse_list[-1])
        node_v.neuron.add_output(node_n.synapse_list[-1])

        syn_n = synapse(name='somatic_synapse_n')
        node_n.synapse_list.append(syn_n)
        node_n.neuron.dend_soma.add_input(syn_n,connection_strength=inhibition)
        node_n.neuron.add_output(node_z.synapse_list[-1])
        node_n.neuron.add_output(node_v.synapse_list[-1])

        # node_z.plot_structure()
        names = list(letters.keys())

        # make a list of these nodes
        nodes = [node_z,node_v,node_n]

        # initialize dictionaries for tracking updates made to all dendrites 
        # for each node
        trajects = [{},{},{}]

        # populate dicts with dendrite names as keys initialize value-lists
        for ii,node in enumerate(nodes):
            count = 0
            for dend in node.dendrite_list:
                if 'ref' not in dend.name:
                    trajects[ii][dend.name] = [0]
        
        # set spiking expectation for each node according to each input
        expect_z = [5,0,0] # z-node should spike 5 times for z input, 0 otherwise
        expect_v = [0,5,0]
        expect_n = [0,0,5]
        expects = [expect_z,expect_v,expect_n]
        # spike_trajectories = [[] for i in range(len(names))]

        # init some tracking vars
        accs = []
        preds = [[0],[0],[0]]
        p_count = 0
        running   = 0
        run_count = 0

        # train for some max number of epochs over whole dataset
        for run in range(100):

            # init run-specific tracking vars
            total_error_z = 0
            total_error_v = 0
            total_error_n = 0
            converged     = 0
            success = 0

            # iterate over each class
            for j in range(len(letters[list(letters.keys())[0]])):
                predictions = []

                # iterate over all samples in that class
                for i,(name,pixels_list) in enumerate(letters.items()):

                    # create spikes of that class
                    defined_spikes = make_spikes(pixels_list[j],20)
                    # plot_letter(letters[name][j])

                    # create input object with those spikes
                    input = SuperInput(type='defined',defined_spikes=defined_spikes)

                    # feed to all nodes
                    node_z.one_to_one(input)
                    node_v.one_to_one(input)
                    node_n.one_to_one(input)

                    # run network
                    net = network(
                        sim=True,
                        dt=.1,
                        tf=100,
                        nodes=[node_z,node_v,node_n]
                        )
                    
                    # take output spikes
                    spikes = array_to_rows(net.spikes,3)

                    # measure error for each node relative to expectation
                    error_z = expects[0][i] - len(spikes[0])
                    error_v = expects[1][i] - len(spikes[1])
                    error_n = expects[2][i] - len(spikes[2])
                    
                    # take note of outputs
                    outputs = [len(spikes[0]),len(spikes[1]),len(spikes[2])]

                    # record prediction (node with max output)
                    predictions.append(np.argmax(outputs))

                    # track predictions
                    for pred in preds:
                        pred.append(pred[p_count])
                    p_count+=1
                    preds[np.argmax(outputs)][p_count] += 1

                    # check if correct
                    if np.argmax(outputs) == j:
                        running += 1
                    run_count +=1
                    accs.append(np.round(running/run_count,2)*100)

                    # increment total error for each node
                    total_error_z+=np.abs(error_z)
                    total_error_v+=np.abs(error_v)
                    total_error_n+=np.abs(error_n)

                    # reset spike times
                    # spike_trajectories[i].append(len(out_spikes))
                    node_z.neuron.spike_times=[]
                    node_v.neuron.spike_times=[]
                    node_n.neuron.spike_times=[]
                    
                    # total_change_z = 0
                    # total_change_v = 0
                    # total_change_n = 0

                    # init flux_offset dicts for each node
                    offsets_z = {}
                    offsets_v = {}
                    offsets_n = {}
                    
                    # track total change to each node
                    total_changes = np.zeros((3))

                    # iterate over all dendrites
                    for ii in range(len(node_z.dendrite_list)):

                        # no updates to refracatory dendrites
                        if 'ref' not in node_z.dendrite_list[ii].name:

                            # address dendrite of each node in parallel
                            dend_z = node_z.dendrite_list[ii] 
                            dend_v = node_v.dendrite_list[ii]
                            dend_n = node_n.dendrite_list[ii]

                            # calculate update
                            step_z = (np.mean(dend_z.s))*error_z*learning_rate
                            step_v = (np.mean(dend_v.s))*error_v*learning_rate
                            step_n = (np.mean(dend_n.s))*error_n*learning_rate

                            # calculate mean flux + offsets
                            # *** check that flux not for all trials
                            flux_z = np.mean(dend_z.phi_r) + dend_z.offset_flux
                            flux_v = np.mean(dend_v.phi_r) + dend_v.offset_flux
                            flux_n = np.mean(dend_n.phi_r) + dend_n.offset_flux

                            fluxes = [flux_z,flux_v,flux_n]
                            steps = [step_z,step_v,step_n]

                            # elastic regime
                            # if update would push offset to rollover regime,
                            # reverse sign of update
                            if elasticity == True:
                                for iii,flux in enumerate(fluxes):
                                    if flux > 0.5 or flux < -0.5:
                                        steps[iii] = -steps[iii]
                                    else:
                                        steps[iii] = steps[iii]

                            # inelastic regime
                            # if update would push offset to rollover regime,
                            # update = 0
                            elif elasticity == False:
                                for iii,flux in enumerate(fluxes):
                                    if flux > 0.5 or flux < -0.5:
                                        steps[iii] = 0

                            # apply above derived updates
                            dend_z.offset_flux += steps[0]
                            dend_v.offset_flux += steps[1]
                            dend_n.offset_flux += steps[2]

                            # increment total changes
                            total_changes[0]+=step_z
                            total_changes[1]+=step_v
                            total_changes[2]+=step_n

                            # ***second update incrementation?
                            # this means eta = 0.02 officially
                            dend_z.offset_flux += step_z
                            dend_v.offset_flux += step_v
                            dend_n.offset_flux += step_n

                            # record offsets of each dendrite
                            offsets_z[dend_z.name] = dend_z.offset_flux
                            offsets_v[dend_v.name] = dend_v.offset_flux
                            offsets_n[dend_n.name] = dend_n.offset_flux

                            # track offset trajectories for each dendrite
                            trajects[0][dend_z.name].append(dend_z.offset_flux)
                            trajects[1][dend_v.name].append(dend_v.offset_flux)
                            trajects[2][dend_n.name].append(dend_n.offset_flux)

                    # if run%10==0:
                    #     print(f"  {names[i]}_{j}")
                    #     for iii,s in enumerate(spikes):
                    #         print(f"   {np.around(s,2)} --> {total_changes[iii]}")
                            
                # node_z.plot_arbor_activity(net,phir=True)
                # node_v.plot_arbor_activity(net,phir=True)
                # node_n.plot_arbor_activity(net,phir=True)
                # print(predictions)

                # check if each class sample was predicted correctly
                if predictions == [0,1,2]:
                    # print("Correct!",predictions)
                    success += 1
            
            # print run accuracy
            acc = np.round(success/10,2)*100
            # accs.append(acc)
            print(f"Run {run} accuracy = {acc}%")

            # if intermittent validation is true
            # test performance on whole dataset without making updates
            # if accuracy is 100% on all three class, converge early!
            if int_val == True:
                offsets = [offsets_z,offsets_v,offsets_n]
                early_converge = test_noise_set(letters,offsets)
                if early_converge == 1:
                    print("Early Converge")
                    break

            # if epoch accuracy 100%, converge!
            if success == 10:
                converged += 1
                if converged >= 1:
                    print(f"Converged! (in {run} runs)")
                    # print(" ",names[i],"spikes = ",len(out_spikes)," error = ",
                    # error,out_spikes,total_change)
                    print([total_error_z,total_error_z,total_error_z])
                    print(spikes)
                    print(" z offset =",offsets_z)
                    print(" v offset =",offsets_v)
                    print(" n offset =",offsets_n)
                # node_z.plot_arbor_activity(net)

                break

        # record offsets (trained effective weights for correct classification)
        offsets = [offsets_z,offsets_v,offsets_n]

        return offsets, preds, accs, trajects


    def test_noise_set(noise_set,offsets):
        expects = [
            [5,0,0],
            [0,5,0],
            [0,0,5]
        ]

        weights = [
            [[.5,.5,.5]],
            [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
        ]

        weights_z = weights
        weights_v = weights
        weights_n = weights

        node_z = SuperNode(weights=weights_z)
        node_v = SuperNode(weights=weights_v)
        node_n = SuperNode(weights=weights_n)

        syn_z = synapse(name='somatic_synapse_z')
        node_z.synapse_list.append(syn_z)
        node_z.neuron.dend_soma.add_input(syn_z,connection_strength=inhibition)
        node_z.neuron.add_output(node_v.synapse_list[-1])
        node_z.neuron.add_output(node_n.synapse_list[-1])

        syn_v = synapse(name='somatic_synapse_v')
        node_v.synapse_list.append(syn_v)
        node_v.neuron.dend_soma.add_input(syn_v,connection_strength=inhibition)
        node_v.neuron.add_output(node_z.synapse_list[-1])
        node_v.neuron.add_output(node_n.synapse_list[-1])

        syn_n = synapse(name='somatic_synapse_n')
        node_n.synapse_list.append(syn_n)
        node_n.neuron.dend_soma.add_input(syn_n,connection_strength=inhibition)
        node_n.neuron.add_output(node_z.synapse_list[-1])
        node_n.neuron.add_output(node_v.synapse_list[-1])

        nodes = [node_z,node_v,node_n]
        
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

                input = SuperInput(type='defined',defined_spikes=defined_spikes)

                node_z.one_to_one(input)
                node_v.one_to_one(input)
                node_n.one_to_one(input)


                net = network(
                    sim=True,
                    dt=.1,
                    tf=100,
                    nodes=[node_z,node_v,node_n]
                    )
                
                spikes = array_to_rows(net.spikes,3)

                outputs = [len(spikes[0]),len(spikes[1]),len(spikes[2])]
                # print(i,outputs)

                # if j==0:
                #     for node in nodes:
                #         node.plot_arbor_activity(net,phir=True)

                if np.argmax(outputs) == i:
                    correct +=1

                for node in nodes:
                    node.neuron.spike_times=[]

            print(f"  test {name} --> accuracy = {100*correct/len(pixel_list)}%")

            if correct==10: corrects+=1
        if corrects == 3:
            return 1
        else:
            return 0




    # partial noise, no bounce
    correct_z = {
        'rand_neuron_77132_soma'             : -0.17018820352206204, 
        'rand_neuron_77132_lay1_branch0_den0': -0.1005137433783909, 
        'rand_neuron_77132_lay1_branch0_den1': -0.15793987522376604, 
        'rand_neuron_77132_lay1_branch0_den2': -0.09882142011047565, 
        'rand_neuron_77132_lay2_branch0_den0': -0.03885238533504326, 
        'rand_neuron_77132_lay2_branch0_den1': 0.1779857429759991, 
        'rand_neuron_77132_lay2_branch0_den2': -0.5318486717211699, 
        'rand_neuron_77132_lay2_branch1_den0': -0.5951666245254208, 
        'rand_neuron_77132_lay2_branch1_den1': 0.22541383358572367, 
        'rand_neuron_77132_lay2_branch1_den2': -0.5951666245254208, 
        'rand_neuron_77132_lay2_branch2_den0': -0.5366149403665759, 
        'rand_neuron_77132_lay2_branch2_den1': -0.03885238533504326, 
        'rand_neuron_77132_lay2_branch2_den2': 0.1779857429759991
                }

    correct_v = {
        'rand_neuron_77132_soma'             : 0.044736378133675835, 
        'rand_neuron_77132_lay1_branch0_den0': 0.0982066950642038, 
        'rand_neuron_77132_lay1_branch0_den1': -0.0023132695052485467, 
        'rand_neuron_77132_lay1_branch0_den2': -0.08514403864197762, 
        'rand_neuron_77132_lay2_branch0_den0': 0.10824719509768488, 
        'rand_neuron_77132_lay2_branch0_den1': -0.24870807367155753, 
        'rand_neuron_77132_lay2_branch0_den2': 0.2661981451385887, 
        'rand_neuron_77132_lay2_branch1_den0': 0.023510432122448473, 
        'rand_neuron_77132_lay2_branch1_den1': -0.17347386222102582, 
        'rand_neuron_77132_lay2_branch1_den2': 0.023510432122448473, 
        'rand_neuron_77132_lay2_branch2_den0': -0.20381303305920312, 
        'rand_neuron_77132_lay2_branch2_den1': 0.10824719509768488, 
        'rand_neuron_77132_lay2_branch2_den2': -0.24870807367155753
                }

    correct_n = {
        'rand_neuron_77132_soma'             : 0.04693267924276726, 
        'rand_neuron_77132_lay1_branch0_den0': -0.08684334793735499, 
        'rand_neuron_77132_lay1_branch0_den1': -0.006008094102381742, 
        'rand_neuron_77132_lay1_branch0_den2': 0.09389567597814484, 
        'rand_neuron_77132_lay2_branch0_den0': -0.2533917538265178, 
        'rand_neuron_77132_lay2_branch0_den1': 0.09256149661966233, 
        'rand_neuron_77132_lay2_branch0_den2': -0.20536055171250206, 
        'rand_neuron_77132_lay2_branch1_den0': 0.022896546042728123, 
        'rand_neuron_77132_lay2_branch1_den1': -0.18109626627015565, 
        'rand_neuron_77132_lay2_branch1_den2': 0.022896546042728123, 
        'rand_neuron_77132_lay2_branch2_den0': 0.26382593975165725, 
        'rand_neuron_77132_lay2_branch2_den1': -0.2533917538265178, 
        'rand_neuron_77132_lay2_branch2_den2': 0.09256149661966233
                }




    # mutual inhibtion noise set offsets with bounce
    z_offset = {
        'rand_neuron_77132_soma': 0.2650382370927349, 
        'rand_neuron_77132_lay1_branch0_den0': -0.093861435157056, 
        'rand_neuron_77132_lay1_branch0_den1': -0.03323869108980681, 
        'rand_neuron_77132_lay1_branch0_den2': -0.14638246822187134, 
        'rand_neuron_77132_lay2_branch0_den0': 0.2526671461080392, 
        'rand_neuron_77132_lay2_branch0_den1': 0.16823237330860874, 
        'rand_neuron_77132_lay2_branch0_den2': -0.20733871542361854, 
        'rand_neuron_77132_lay2_branch1_den0': -0.34928537180482944, 
        'rand_neuron_77132_lay2_branch1_den1': 0.1924460141126753, 
        'rand_neuron_77132_lay2_branch1_den2': -0.34538475046402284, 
        'rand_neuron_77132_lay2_branch2_den0': -0.2027569303236612, 
        'rand_neuron_77132_lay2_branch2_den1': 0.20031476560183542, 
        'rand_neuron_77132_lay2_branch2_den2': 0.17045227184262535
        }
    v_offset = {
        'rand_neuron_32432_soma': 0.09882751347073629, 
        'rand_neuron_32432_lay1_branch0_den0': 0.05884059880021676, 
        'rand_neuron_32432_lay1_branch0_den1': -0.04098210452862971, 
        'rand_neuron_32432_lay1_branch0_den2': -0.15156209001231472, 
        'rand_neuron_32432_lay2_branch0_den0': 0.16624696205213663, 
        'rand_neuron_32432_lay2_branch0_den1': -0.34521914413377364, 
        'rand_neuron_32432_lay2_branch0_den2': 0.15733325541624782, 
        'rand_neuron_32432_lay2_branch1_den0': 0.1572538642045978, 
        'rand_neuron_32432_lay2_branch1_den1': -0.1971534847182594, 
        'rand_neuron_32432_lay2_branch1_den2': 0.15729722432569274, 
        'rand_neuron_32432_lay2_branch2_den0': -0.30450876081723155, 
        'rand_neuron_32432_lay2_branch2_den1': 0.15593982278258978, 
        'rand_neuron_32432_lay2_branch2_den2': -0.34908548286225927
        }
    n_offset = {
        'rand_neuron_55497_soma': 0.21393537762054354, 
        'rand_neuron_55497_lay1_branch0_den0': -0.18721420415198847, 
        'rand_neuron_55497_lay1_branch0_den1': -0.25145395389347136, 
        'rand_neuron_55497_lay1_branch0_den2': -0.0006933217927859708, 
        'rand_neuron_55497_lay2_branch0_den0': -0.3492164440342981, 
        'rand_neuron_55497_lay2_branch0_den1': 0.15837445973964892, 
        'rand_neuron_55497_lay2_branch0_den2': -0.35568954978217576, 
        'rand_neuron_55497_lay2_branch1_den0': 0.15767025397709697, 
        'rand_neuron_55497_lay2_branch1_den1': -0.3476209075643123, 
        'rand_neuron_55497_lay2_branch1_den2': 0.15648896076476415, 
        'rand_neuron_55497_lay2_branch2_den0': 0.1662113645470249, 
        'rand_neuron_55497_lay2_branch2_den1': -0.34766398420577305, 
        'rand_neuron_55497_lay2_branch2_den2': -0.17686624342045137
        }


    np.random.seed(10)
    learning_rate=.01
    inhibition=-1
    letters = make_letters()
    names = list(letters.keys())

    all_spikes = []
    for name,pixels in letters.items():
        # plot_letter(pixels)
        all_spikes.append(make_spikes(pixels,20))

    elasticity = True
    int_val = True
    els = [None,False,True]
    vals = [True,False]

    for el in els:
        for val in vals:

            elasticity = el
            int_val = val

            regimes = ['Elastic', 'Inelastic', 'Unbounded']
            if elasticity == True:
                regime = regimes[0]
            elif elasticity == False:
                regime = regimes[1]
            else:
                regime = regimes[2]

            if int_val == True:
                converge_type = 'Intermittent'
            else:
                converge_type = 'Update'

            print(f"Regime = {regime}, Converge = {converge_type}")

            offsets, preds, accs, trajects = train_9pixel_classifier(
                noise_set,
                all_spikes,
                learning_rate,
                inhibition,
                elasticity,
                int_val
                )


            path = "results/pixels_WTA_icons_RERUN/"
            picklit(
                preds,
                path,
                f"{regime}_{converge_type}_predictions"
                )
            picklit(
                accs,
                path,
                f"{regime}_{converge_type}_accs"
                )
            picklit(
                trajects,
                path,
                f"{regime}_{converge_type}_trajects"
                )

            plt.figure(figsize=(8,4))
            for i,p in enumerate(preds):
                plt.plot(p/np.arange(1,len(p)+1,1),label=names[i])
            plt.legend()
            plt.title(f"Class Predictions for {regime} Noisy 9-Pixel Classifier",fontsize=16)
            plt.xlabel("Cycles Over All Samples",fontsize=14)
            plt.ylabel("Percent Predicted",fontsize=14)
            plt.subplots_adjust(bottom=.15)
            plt.savefig(path+regime+converge_type+'_pred_plot')
            plt.close()

            plt.figure(figsize=(8,4))
            plt.plot(accs)
            plt.title(f"Learning Accuracy for {regime} Noisy 9-Pixel Classifier",fontsize=16)
            plt.xlabel("Total Iterations",fontsize=14)
            plt.ylabel("Percent Accuracy",fontsize=14)
            plt.subplots_adjust(bottom=.15)
            plt.savefig(path+regime+converge_type+'_accs_plot')
            plt.close()


            plt.style.use('seaborn-v0_8-muted')
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            for i,traject in enumerate(trajects):
                plt.figure(figsize=(8,4))
                count1=0
                count2=0
                for name,offset in reversed(traject.items()):
                    if 'soma' in name:
                        name = 'soma'
                        # plt.plot(offset,color=colors[i],label=name,linewidth=4)
                        plt.plot(offset,color=colors[0],label=name,linewidth=4)
                    elif 'lay1' in name:
                        col = colors[1]

                        if count1 == 0:
                            plt.plot(offset,'--',color=col,linewidth=2,label='Layer 1')
                        else:
                            # plt.plot(offset,color=colors[0],label=name,linewidth=3)
                            plt.plot(offset,'--',color=col,linewidth=2)
                        count1+=1

                    elif 'lay2' in name:
                        col = colors[2]
                        if count2 == 0:
                            plt.plot(offset,':',color=col,label='Layer 2',linewidth=1)
                        else:
                            plt.plot(offset,':',color=col,linewidth=1)
                        # plt.plot(offset,color=colors[4],label=name)
                        count2+=1

                plt.title(f"Noisy 9-Pixel Classifier {regime} {converge_type} Convergence - {names[i]}",fontsize=16)
                plt.xlabel("Total Iterations",fontsize=14)
                plt.ylabel("Flux Offset",fontsize=14)
                plt.subplots_adjust(bottom=.15)
                plt.legend()
                plt.savefig(path+regime+converge_type+f'_offsets_{names[i]}_plot')
                plt.close()
            test_noise_set(noise_set,offsets)
    

    # correct_offsets = [correct_z,correct_v,correct_n] # partial noise, no bounce

    # offsets = [z_offset,v_offset,n_offset] # full noise, bounce

    # correct_offsets = offsets 
    # test_on_noise(correct_offsets)

    # test_noise_set(noise_set,offsets)

if __name__=='__main__':
    main()










# offsets = train_9pixel_classifier(letters,all_spikes,learning_rate,inhibition)
# print(names)


# def single_9pixel_classifier(let_idxs,spike_times,expect):

#     weights = [
#         [[.5,.5,.5]],
#         [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
#     ]
#     node_z = SuperNode(weights=weights)
#     names = ['z','v','n']
#     spike_trajectories = [[] for i in range(len(letters))]

#     for run in range(300):
#         if run%10==0:print(f"Run {run}:")
#         total_error=0
#         for i,let_idx in enumerate(let_idxs):
#             defined_spikes=[let_idx,spike_times]
#             input = SuperInput(type='defined',defined_spikes=defined_spikes)
#             node_z.one_to_one(input)

#             net = network(sim=True,dt=.1,tf=100,nodes=[node_z])
#             out_spikes = net.spikes[1]

#             error = expect[i] - len(out_spikes)
#             total_error+=np.abs(error)
#             spike_trajectories[i].append(len(out_spikes))
#             node_z.neuron.spike_times=[]

#             total_change = 0
#             offsets = {}
#             for dend in node_z.dendrite_list:
#                 if 'ref' not in dend.name:
#                     step = error*np.mean(dend.s)*.01
#                     total_change+=step
#                     dend.offset_flux += step
#                     offsets[dend.name] = dend.offset_flux

#             if run%10==0:
#                 print("  ",names[i],"spikes = ",len(out_spikes)," error = ",
#                         error,out_spikes,total_change)

#             # if i == 0:
#             #   node_z.plot_arbor_activity(net,phir=True)

#         if total_error==0:
#             print(f"Converged! (in {run} runs)")
#             print(" ",names[i],"spikes = ",len(out_spikes)," error = ",
#                     error,out_spikes,total_change)
#             print(" ",offsets)
#             node_z.plot_arbor_activity(net)
#             break
#     for i,spk_trj in enumerate(spike_trajectories):
#         plt.plot(spk_trj,label=names[i])

#     plt.legend()
#     plt.show()

# def test_9pixel_classifier(correct_offset,expect,letters,spike_time,let):
#     # np.random.seed(None)
#     # make new noisy letters
#     let_idxs = []
#     spike_times = []

#     for ii,(name, pixels) in enumerate(letters.items()):
#         idx = np.where(np.array(letters[name])==1)[0]
#         let_idxs.append(idx)
#         s_ts = (np.ones(len(idx))*spike_time).astype(int)
#         spike_times.append(s_ts)

#     weights = [
#         [[.5,.5,.5]],
#         [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
#     ]
#     np.random.seed(10)
#     node = SuperNode(weights=weights)

#     count = 0
#     for dend in node.dendrite_list:
#         if 'ref' not in dend.name:
#             dend.offset_flux = correct_offset[list(correct_offset.keys())[count]]
#             count+=1
    
#     names = ['z','v','n']

#     total_error=0
#     max_spikes = []
#     for i,let_idx in enumerate(let_idxs):
#         # plot_letter(letters[names[i]])
#         defined_spikes=[let_idx,spike_times[i]]
#         input = SuperInput(type='defined',defined_spikes=defined_spikes)
#         node.one_to_one(input)

#         net = network(sim=True,dt=.1,tf=100,nodes=[node])
#         out_spikes = net.spikes[1]

#         error = expect[i] - len(out_spikes)
#         max_spikes.append(len(out_spikes))
#         total_error+=np.abs(error)
#         node.neuron.spike_times=[]

#         print(" ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes)    

#     if np.argmax(max_spikes)==let:
#         # print(f"Correct!")
#         # print(" ",offsets)
#         # node.plot_arbor_activity(net)
#         return 1
#     else: return 0


# def test_on_noise(correct_offsets):
#     tests=10
#     expects = [
#         [5,0,0],
#         [0,5,0],
#         [0,0,5]
#     ]

#     for l in range(3):
#         correct = 0
#         for i in range(tests):
#             print("-")
#             letters = make_letters()
#             looters = {}
#             # noise_int = np.random.randint(3)
#             for ii,(name, pixels) in enumerate(letters.items()):
#                 p = pixels
#                 looters[name] = one_pixel_noise(p)
#                 # if ii == noise_int:
#                 #     # print(ii)
#                 #     looters[name] = one_pixel_noise(p)
#                 # else:
#                 #     looters[name] = letters[name]

#                 # plot_letter(looters[name])
            
#             correct += test_9pixel_classifier(correct_offsets[l],expects[l],looters,20,l)
#         print(f"{names[l]} --> accuracy = {100*correct/tests}%")




        # names = ['z','v','n']
    # expect = [0,0,5]
    # spike_trajectories = [[] for i in range(len(letters))]
    # for run in range(300):
    #     if run%10==0:print(f"Run {run}:")
    #     total_error=0
    #     for i,let_idx in enumerate(let_idxs):
    #         defined_spikes=[let_idx,spike_times]
    #         input = SuperInput(type='defined',defined_spikes=defined_spikes)
    #         node_z.one_to_one(input)

    #         net = network(sim=True,dt=.1,tf=100,nodes=[node_z])
    #         out_spikes = net.spikes[1]

    #         error = expect[i] - len(out_spikes)
    #         total_error+=np.abs(error)
    #         spike_trajectories[i].append(len(out_spikes))
    #         node_z.neuron.spike_times=[]

    #         total_change = 0
    #         offsets = {}
    #         count = 0
            # for dend in node_z.dendrite_list:
            #     if 'ref' not in dend.name:
    #                 step = error*np.mean(dend.s)*.01
    #                 total_change+=step
    #                 dend.offset_flux += step
    #                 offsets[dend.name] = dend.offset_flux

    #                 dend.offset_flux = correct_n[list(correct_n.keys())[count]]
    #                 count+=1
    #                 # dend.offset_flux = correct_n[dend.name]

    #         # if run%10==0:
    #         #     print("  ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)

    #         # print(names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)
    #         # if i == 0:
    #         # node_z.plot_arbor_activity(net,phir=True)

    #     if total_error==0:
    #         print(f"Converged! (in {run} runs)")
    #         print(" ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)
    #         print(" ",offsets)
    #         node_z.plot_arbor_activity(net)
    #         break


    ## triple set (within inhibition)
    ## Converged! (in 116 runs)
    #  z offset = {'rand_neuron_77132_soma': -0.05133196321696619, 'rand_neuron_77132_lay1_branch0_den0': 0.06340336185773317, 'rand_neuron_77132_lay1_branch0_den1': -0.005232154795712155, 'rand_neuron_77132_lay1_branch0_den2': 0.09154152422813781, 'rand_neuron_77132_lay2_branch0_den0': 0.1401798469186167, 'rand_neuron_77132_lay2_branch0_den1': 0.29043747997572833, 'rand_neuron_77132_lay2_branch0_den2': -0.23704030737407575, 'rand_neuron_77132_lay2_branch1_den0': -0.24893444751833296, 'rand_neuron_77132_lay2_branch1_den1': 0.3216253506588986, 'rand_neuron_77132_lay2_branch1_den2': 
    # -0.24893444751833296, 'rand_neuron_77132_lay2_branch2_den0': -0.16774743225179672, 'rand_neuron_77132_lay2_branch2_den1': 0.1401798469186167, 'rand_neuron_77132_lay2_branch2_den2': 0.29043747997572833}
    #  v offset = {'rand_neuron_20449_soma': -0.04263633299098412, 'rand_neuron_20449_lay1_branch0_den0': 0.0701461826670961, 'rand_neuron_20449_lay1_branch0_den1': 0.06877023093800687, 'rand_neuron_20449_lay1_branch0_den2': -0.07495985689875015, 'rand_neuron_20449_lay2_branch0_den0': 0.27633012555480013, 'rand_neuron_20449_lay2_branch0_den1': -0.2371128130333391, 'rand_neuron_20449_lay2_branch0_den2': 0.380266967900081, 'rand_neuron_20449_lay2_branch1_den0': 0.36882194594968254, 'rand_neuron_20449_lay2_branch1_den1': -0.23748206487042078, 'rand_neuron_20449_lay2_branch1_den2': 0.36882194594968254, 'rand_neuron_20449_lay2_branch2_den0': -0.22195471349909865, 'rand_neuron_20449_lay2_branch2_den1': 0.27633012555480013, 'rand_neuron_20449_lay2_branch2_den2': -0.24873131000482887}
    #  n offset = {'rand_neuron_36358_soma': -0.07486545497234011, 'rand_neuron_36358_lay1_branch0_den0': -0.022340479767096275, 'rand_neuron_36358_lay1_branch0_den1': 0.03443815318245595, 'rand_neuron_36358_lay1_branch0_den2': 0.1239558153546989, 'rand_neuron_36358_lay2_branch0_den0': -0.23237041636107128, 'rand_neuron_36358_lay2_branch0_den1': 0.3746560351812661, 'rand_neuron_36358_lay2_branch0_den2': -0.2306915957822732, 'rand_neuron_36358_lay2_branch1_den0': 0.07975166684240759, 'rand_neuron_36358_lay2_branch1_den1': -0.13258086770115457, 'rand_neuron_36358_lay2_branch1_den2': 
    # 0.07975166684240759, 'rand_neuron_36358_lay2_branch2_den0': 0.3878146763449766, 'rand_neuron_36358_lay2_branch2_den1': -0.23237041636107128, 'rand_neuron_36358_lay2_branch2_den2': 0.16311564190585467}
