import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')

from sim_soens.super_node import SuperNode
from sim_soens.super_functions import *
from sim_soens.soen_components import network
from sim_soens.soen_plotting import activity_plot




def main():
    '''
    Nine-Pixel Classifier

    '''
    params = {
        
        'weights': [
            [[.5,.4,.3]],
            [[0.5,0.5],[0.5,0.5],[0.5,0.5]],
            [[0.35,-0.65],[0.35,-0.65],[0.35,-0.65],[0.35,-0.65],[0.35,-0.65],[0.35,-0.65]]
        ],

        # the time constant at every dendrite
        'taus': [
            [[10,150,1000]],
            [[250,250],[250,250],[250,250]],
            [[250,250],[250,250],[250,250],[250,250],[250,250],[250,250]]
        ],

        # numbervalues refer to indices of a list of bias values for which 
        # rate-arrays have been generated
        'biases': [
            [[3,3,3]],
            [[5,5],[5,5],[5,5]],
            [[-4,3],[-4,3],[-4,3],[-4,3],[-4,3],[-4,3]]
        ],

        # defines dendrite type for each dendrite
        'types': [
            [['rtti','rtti','rtti']],
            [['ri','ri'],['ri','ri'],['ri','ri']],
            [['rtti','ri'],['rtti','ri'],['rtti','ri'],['rtti','ri'],['rtti','ri'],['rtti','ri']]
        ],

        # input from this (number) channel goes to the (index position) synapse
        'syns': [['2','5'],['4','6'],['5','8'],['4','6'],['1','3'],['7','9'],
                ['4','6'],['2','5'],['7','9'],['1','3'],['4','6'],['5','8']],

        # with this associated weight
        'syn_w': [[.6,.6],[.5,.5],[.6,.6],[.5,.5],[.6,.6],[.5,.5],
                [.6,.6],[.5,.5],[.6,.6],[.5,.5],[.6,.6],[.5,.5]],
        
        # other neuron and denrite parameters
        "tau_di": 250,
        "ib_n"  : 1.5523958588352207, 
        "tau_ni": 50,
        "ib_ref": 1.7523958588352209, 
    }


    # create a neuron with this structure and parameters
    nine_neuron = SuperNode(s_th=.1,**params) 
    nine_neuron.parameter_print()
    nine_neuron.plot_structure()

    letters=make_letters()
    inputs = make_inputs(letters,20)
    plot_letters(letters)

    # for saving neuron states
    run_neurons = []

    # test on letters
    for i,let in enumerate(letters):

        # make a nine-pixel classifier neuron
        nine_neuron = SuperNode(s_th=.1,**params) 

        # letter defined input
        input = inputs[i]

        # add input channels to appropriate synapses
        # this has since been automated
        count = 0
        for g in nine_neuron.synapses:
            for s in g:
                for i,row in enumerate(input.spike_rows):
                    if i == int(s.name)-1:
                        s.add_input(input.signals[i])
                        count+=1
        run_neurons.append(nine_neuron)

    # run all neurons simultaneously
    net = network(sim=True,dt=.1,tf=150,nodes=run_neurons) #,new_way=True)

    # plot!
    title = 'Responses to All Three 9-Pixel Images'
    subtitles =['Z','V','N']
    # activity_plot(run_neurons,net,dend=False,phir=True,size=(12,8),title=title,subtitles=subtitles)

    for n in run_neurons:
        n.plot_arbor_activity(net,phir=True)
        # n.activity_plot()



if __name__=='__main__':
    main()