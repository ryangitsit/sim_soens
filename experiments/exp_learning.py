
import numpy as np
import matplotlib.pyplot as plt
import sys
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

        # convert to indices of nine-channel input
        idx = np.where(np.array(letter)==1)[0]

        # all non-zero indices will spike at spike_time
        times = (np.ones(len(idx))*spike_time).astype(int)
        spikes = [idx,times]

        return spikes

    def one_pixel_noise(letter):
        noise_idx = np.random.randint(0,len(letter))
        if letter[noise_idx] == 1:
            letter[noise_idx] = 0
        else:
            letter[noise_idx] = 1
        return letter

    def plot_letter(letter,name=None):
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

    np.random.seed(10)

    def train_9pixel_classifier(letters,all_spikes):
        weights = [
            [[.5,.5,.5]],
            [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
        ]

        node_z = SuperNode(weights=weights)
        node_v = SuperNode(weights=weights)
        node_n = SuperNode(weights=weights)

        syn_z = synapse(name='somatic_synapse_z')
        node_z.synapse_list.append(syn_z)
        node_z.neuron.dend_soma.add_input(syn_z,connection_strength=-1)
        node_z.neuron.add_output(node_v.synapse_list[-1])
        node_z.neuron.add_output(node_n.synapse_list[-1])

        syn_v = synapse(name='somatic_synapse_v')
        node_v.synapse_list.append(syn_v)
        node_v.neuron.dend_soma.add_input(syn_v,connection_strength=-1)
        node_v.neuron.add_output(node_z.synapse_list[-1])
        node_v.neuron.add_output(node_n.synapse_list[-1])

        syn_n = synapse(name='somatic_synapse_n')
        node_n.synapse_list.append(syn_n)
        node_n.neuron.dend_soma.add_input(syn_n,connection_strength=-1)
        node_n.neuron.add_output(node_z.synapse_list[-1])
        node_n.neuron.add_output(node_v.synapse_list[-1])

        node_z.plot_structure()
        names = list(letters.keys())

        expect_z = [5,0,0]
        expect_v = [0,5,0]
        expect_n = [0,0,5]
        expects = [expect_z,expect_v,expect_n]
        # spike_trajectories = [[] for i in range(len(names))]

        for run in range(300):
            if run%10==0:print(f"Run {run}:")
            total_error_z=0
            total_error_v=0
            total_error_n=0
            
            for i,let_idx in enumerate(all_spikes):
                
                # defined_spikes=[let_idx,spike_times]
                defined_spikes = all_spikes[i]
                input = SuperInput(type='defined',defined_spikes=defined_spikes)

                node_z.one_to_one(input)
                node_v.one_to_one(input)
                node_n.one_to_one(input)

                net = network(sim=True,dt=.1,tf=100,nodes=[node_z,node_v,node_n])
                
                spikes = array_to_rows(net.spikes,3)
                
                if run%10==0:
                    print(" ",names[i])
                    for s in spikes:
                        print("  ",s)

                error_z = expects[0][i] - len(spikes[0])
                error_v = expects[1][i] - len(spikes[1])
                error_n = expects[2][i] - len(spikes[2])
                

                total_error_z+=np.abs(error_z)
                total_error_v+=np.abs(error_v)
                total_error_n+=np.abs(error_n)


                # spike_trajectories[i].append(len(out_spikes))
                node_z.neuron.spike_times=[]
                node_v.neuron.spike_times=[]
                node_n.neuron.spike_times=[]

                total_change_z = 0
                total_change_v = 0
                total_change_n = 0

                offsets_z = {}
                offsets_v = {}
                offsets_n = {}

                for i in range(len(node_z.dendrite_list)):
                    if 'ref' not in node_z.dendrite_list[i].name:
                        dend_z = node_z.dendrite_list[i]
                        dend_v = node_v.dendrite_list[i]
                        dend_n = node_n.dendrite_list[i]


                        step_z = error_z*np.mean(dend_z.s)*.01
                        step_v = error_v*np.mean(dend_v.s)*.01
                        step_n = error_n*np.mean(dend_n.s)*.01

                        total_change_z+=step_z
                        total_change_v+=step_v
                        total_change_n+=step_n

                        dend_z.offset_flux += step_z
                        dend_v.offset_flux += step_v
                        dend_n.offset_flux += step_n

                        offsets_z[dend_z.name] = dend_z.offset_flux
                        offsets_v[dend_v.name] = dend_v.offset_flux
                        offsets_n[dend_n.name] = dend_n.offset_flux

                # if run%10==0:
                #     print("  ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)

            if np.sum([total_error_z,total_error_z,total_error_z]) == 0:
                print(f"Converged! (in {run} runs)")
                # print(" ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)
                print(" z offset =",offsets_z)
                print(" v offset =",offsets_v)
                print(" n offset =",offsets_n)
                # node_z.plot_arbor_activity(net)

                break

    np.random.seed(10)
    letters = make_letters()
    names = list(letters.keys())
    # all_spikes = []
    # for name,pixels in letters.items():
    #     # plot_letter(pixels)
    #     all_spikes.append(make_spikes(pixels,20))
    # train_9pixel_classifier(letters,all_spikes)
    # print(names)


    def single_9pixel_classifier(let_idxs,spike_times,expect):

        weights = [
            [[.5,.5,.5]],
            [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
        ]
        node_z = SuperNode(weights=weights)
        names = ['z','v','n']
        spike_trajectories = [[] for i in range(len(letters))]

        for run in range(300):
            if run%10==0:print(f"Run {run}:")
            total_error=0
            for i,let_idx in enumerate(let_idxs):
                defined_spikes=[let_idx,spike_times]
                input = SuperInput(type='defined',defined_spikes=defined_spikes)
                node_z.one_to_one(input)

                net = network(sim=True,dt=.1,tf=100,nodes=[node_z])
                out_spikes = net.spikes[1]

                error = expect[i] - len(out_spikes)
                total_error+=np.abs(error)
                spike_trajectories[i].append(len(out_spikes))
                node_z.neuron.spike_times=[]

                total_change = 0
                offsets = {}
                for dend in node_z.dendrite_list:
                    if 'ref' not in dend.name:
                        step = error*np.mean(dend.s)*.01
                        total_change+=step
                        dend.offset_flux += step
                        offsets[dend.name] = dend.offset_flux

                if run%10==0:
                    print("  ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)

                # if i == 0:
                #   node_z.plot_arbor_activity(net,phir=True)

            if total_error==0:
                print(f"Converged! (in {run} runs)")
                print(" ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes,total_change)
                print(" ",offsets)
                node_z.plot_arbor_activity(net)
                break
        for i,spk_trj in enumerate(spike_trajectories):
            plt.plot(spk_trj,label=names[i])

        plt.legend()
        plt.show()

    def test_9pixel_classifier(correct_offset,expect,letters,spike_time,let):
        # np.random.seed(None)
        # make new noisy letters
        let_idxs = []
        spike_times = []

        for ii,(name, pixels) in enumerate(letters.items()):
            idx = np.where(np.array(letters[name])==1)[0]
            let_idxs.append(idx)
            s_ts = (np.ones(len(idx))*spike_time).astype(int)
            spike_times.append(s_ts)

        weights = [
            [[.5,.5,.5]],
            [[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]]
        ]
        np.random.seed(10)
        node = SuperNode(weights=weights)

        count = 0
        for dend in node.dendrite_list:
            if 'ref' not in dend.name:
                dend.offset_flux = correct_offset[list(correct_offset.keys())[count]]
                count+=1
        
        names = ['z','v','n']

        total_error=0
        max_spikes = []
        for i,let_idx in enumerate(let_idxs):
            defined_spikes=[let_idx,spike_times[i]]
            input = SuperInput(type='defined',defined_spikes=defined_spikes)
            node.one_to_one(input)

            net = network(sim=True,dt=.1,tf=100,nodes=[node])
            out_spikes = net.spikes[1]

            error = expect[i] - len(out_spikes)
            max_spikes.append(len(out_spikes))
            total_error+=np.abs(error)
            node.neuron.spike_times=[]

            # print(" ",names[i],"spikes = ",len(out_spikes)," error = ",error,out_spikes)    

        if np.argmax(max_spikes)==let:
            # print(f"Correct!")
            # print(" ",offsets)
            # node.plot_arbor_activity(net)
            return 1
        else: return 0


    correct_z = {'rand_neuron_77132_soma': -0.17018820352206204, 
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
                'rand_neuron_77132_lay2_branch2_den2': 0.1779857429759991}

    correct_v = {'rand_neuron_77132_soma': 0.044736378133675835, 
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
                'rand_neuron_77132_lay2_branch2_den2': -0.24870807367155753}

    correct_n = {'rand_neuron_77132_soma': 0.04693267924276726, 
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
                'rand_neuron_77132_lay2_branch2_den2': 0.09256149661966233}

    correct_offsets = [correct_z,correct_v,correct_n]
    tests=100
    expects = [
        [5,0,0],
        [0,5,0],
        [0,0,5]
    ]

    for l in range(3):
        correct = 0
        for i in range(tests):
            print("-")
            letters = make_letters()
            looters = {}
            # noise_int = np.random.randint(3)
            for ii,(name, pixels) in enumerate(letters.items()):
                p = pixels
                looters[name] = one_pixel_noise(p)
                # if ii == noise_int:
                #     # print(ii)
                #     looters[name] = one_pixel_noise(p)
                # else:
                #     looters[name] = letters[name]

                # plot_letter(looters[name])

            correct += test_9pixel_classifier(correct_offsets[l],expects[l],looters,20,l)
        print(f"{names[l]} --> accuracy = {100*correct/tests}%")



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
    #         for dend in node_z.dendrite_list:
    #             if 'ref' not in dend.name:
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

if __name__=='__main__':
    main()