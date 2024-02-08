import numpy as np
from matplotlib import pyplot as plt
import time
from matplotlib.collections import PolyCollection
import matplotlib as mp
import pickle
from itertools import cycle

from sim_soens.soen_utilities import (
    depth_of_dendritic_tree, 
    dend_load_arrays_thresholds_saturations, 
    color_dictionary, 
    physical_constants, 
    colors_gist, 
    index_finder
)

colors = color_dictionary()
p = physical_constants()

fig_size = plt.rcParams['figure.figsize']

# =============================================================================
# Plots added by Ryan
# =============================================================================

def raster_plot(
        spikes,duration=None,title=None,input=[],notebook=False,size=(10, 6)
        ):
    from matplotlib import pyplot as plt
    if notebook==True:
        plt.figure(figsize=(6, 4))
    else:
        plt.figure(figsize=size)
    plt.plot(spikes[1], spikes[0], '.k')
    if len(input)>0:
        plt.plot(input[1], input[0]*(max(spikes[0])/max(input[0])), '.', markersize=7.5, color='r')
    if title:
        plt.title(title,fontsize=18)
    else:
        plt.title('Rasterplot of Spiking Activity',fontsize=18)
    plt.xlabel('Spike Time (ns)',fontsize=16)
    plt.ylabel('Index',fontsize=16)
    if duration:
        plt.xlim(0,duration+int(duration/20))
    plt.show()


def activity_plot(
        neurons,net=None,phir=False,dend=True,title=None,input=None,weighting=True,
        docstring=False,lay=100000,spikes=True, path=None,SPD=False,ref=False,
        legend_out=False,size=(12,4), y_range=None,subtitles=None,legend=True,
        legend_all=False, S=True, phi_th=True
        ):
    
    
    '''
    Plots signal activity for a given neuron or network of neurons
     - syntax     -> NeuralZoo.plot_neuron_activity(net,spikes=True,input=input)
     - kwargs:
        - neurons    -> list of all neurons to be plotted
        - net        -> network within which neurons were simulated
        - phir       -> plot phi_r of soma and phi_r thresholds
        - dend       -> plot dendritic signals
        - input      -> mark moments of input events with red spikes
        - SPD        -> plot synaptic flux
        - ref        -> plot refractory signal
        - weighting  -> weight dendritic signals by their connection strength
        - spikes     -> plot output spikes over signal
        - legend_out -> place legend outside of plots
        - size       -> (x,y) size of figure
        - path       -> save plot to path
        
    '''
    if docstring == True:
        print(activity_plot.__doc__)

    if net == None:
        time_vec = np.arange(0,len(neurons[0].dendrites[0][0][0].s),1) 
    else:
        time_vec = net.t

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(neurons) > 1:

        fig, axs = plt.subplots(len(neurons), 1,figsize=(size))
        fig.subplots_adjust(hspace=0)

        for ii,n in enumerate(neurons): 
            # if ii != len(neurons)-1:
            #     axs[ii].get_xaxis().set_visible(False)

            signal = n.dendrites[0][0][0].s
            refractory = n.neuron.dend__ref.s
            phi_r = n.dendrites[0][0][0].phi_r
            axs[ii].plot(time_vec,signal,  label='soma signal', linewidth=4)

            if phir==True:
                if phi_th == True:
                    from sim_soens.soen_functions import phi_thresholds
                    phi_ths = phi_thresholds(n.neuron)
                    axs[ii].axhline(
                        y = phi_ths[1], color = 'purple', 
                        linestyle = '--',linewidth=.5,label=r"$\phi_{th}$"
                        )
                    if any(ele < 0 for ele in phi_r):
                        axs[ii].axhline(y = phi_ths[0], color = 'purple', 
                                        linestyle = '--',linewidth=.5)
                axs[ii].plot(time_vec,phi_r, color = colors[1], label=r'$\phi_r$ (soma)')

            if dend:
                for i,layer in enumerate(n.dendrites):
                    if i < lay +1 :
                        for j,branch in enumerate(layer):
                            for k,dendrite in enumerate(branch):
                                if i == 0 and j == 0 and k ==0:
                                    pass
                                else:
                                    if weighting == True:
                                        weight = n.weights[i-1][j][k]
                                        dend_s = dendrite.s*weight
                                    else:
                                        dend_s = dendrite.s
                                    axs[ii].plot(
                                        time_vec,dend_s,'--', 
                                        label='w*dendrite_'+str([i,j,k])
                                        )
                                if SPD==True:
                                    for spd in dendrite.synaptic_inputs:
                                        axs[ii].plot(
                                            time_vec,
                                            dendrite.synaptic_inputs[spd].phi_spd,
                                            label="SPD"
                                            )
            if S == True:
                axs[ii].plot(time_vec,signal, color='#1f77b4',linewidth=4)

            if input:
                axs[ii].plot(
                    input.spike_arrays[1],np.zeros(len(input.spike_arrays[1])),
                    'xr', markersize=5, label='input event'
                    )
           
            if ref==True:
                axs[ii].plot(time_vec,refractory,':',color = 'r',
                             label='refractory signal')

            ## add input/output spikes
            if spikes==True and hasattr(net,'spikes'):

                # ind = np.where(net.spikes[0]==ii)[0]
                # spike_times=np.array([net.spikes[1][i] for i in ind]).reshape(len(ind),)
                
                # rows = array_to_rows(net.spikes,2)
                # print(rows)
                axs[ii].plot(
                    n.neuron.spike_times,n.neuron.s_th,'xk', markersize=8,
                    label='neuron fires'
                    )
                axs[ii].axhline(
                    y = n.neuron.s_th, color = 'purple', linestyle = '--',
                    label='Firing Threshold'
                    )
            if ii != len(neurons)-1:
                axs[ii].set_xticks([])
            if legend_all == True:
                axs[ii].legend(loc=1)
            elif ii == 0:
                plt.legend()
        label_size = np.min([10+2*len(neurons),14])
        plt.xlabel("Simulation Time (ns)",fontsize=label_size)
        print(int(np.floor((len(neurons)-1)/2)))
        axs[int(np.floor((len(neurons)-1)/2))].set_ylabel(
            "Signal (Ic)",
            fontsize=label_size
            )
            #, labelpad=20)

        axs[int(np.floor(len(neurons)/2))].yaxis.set_label_coords(-1.05,1)
        # fig.set_ylabel("ylabel")
        # plt.subplots_adjust(bottom=.25)
        if title:
            title_size=np.min([10+2*len(neurons)+2,20])
            title_size = 16
            if subtitles:
                fig.suptitle(title,fontsize=title_size) 
            else:
                axs[0].set_title(title,fontsize=title_size)
        if subtitles:
            for i,sub in enumerate(subtitles):
                axs[i].set_title(sub,fontsize=14)

    else:
        signal = neurons[0].dendrites[0][0][0].s
        refractory = neurons[0].neuron.dend__ref.s
        phi_r = neurons[0].dendrites[0][0][0].phi_r

        
        plt.figure(figsize=size)
        if S == True:
            plt.plot(time_vec,signal,  label='soma signal', linewidth=4)

        if phir:
            if phi_th == True:
                from sim_soens.soen_functions import phi_thresholds
                phi_ths = phi_thresholds(neurons[0].neuron)
                plt.axhline(y = phi_ths[1], color = 'purple', linestyle = '--',
                            linewidth=.5,label=r"$\phi_{th}$")
                if any(ele < 0 for ele in phi_r):
                    plt.axhline(y = phi_ths[0], color = 'purple', linestyle = '--',
                                linewidth=.5)
            plt.plot(time_vec,phi_r, color = colors[1],linewidth=2, label=r'$\phi_r$ (soma)')

        if dend:
            dend_names = [f"excitatory_{_}" for _ in range(9)]
            dend_names.append("inhibitory")
            count=0
            for i,layer in enumerate(neurons[0].dendrites):
                if i < lay +1 :
                    for j,branch in enumerate(layer):
                        for k,dendrite in enumerate(branch):
                            if i == 0 and j == 0 and k ==0:
                                pass
                            else:
                                if weighting == True:
                                    weight = neurons[0].weights[i-1][j][k]
                                    dend_s = dendrite.s*weight
                                else:
                                    dend_s = dendrite.s
                                plt.plot(
                                    time_vec,dend_s,'--', 
                                    label=f'w*dend.{i}.{j}.{k}'
                                    # label = dend_names[count]
                                    )
                                count+=1
                            if SPD==True:
                                for spd in dendrite.synaptic_inputs:
                                    plt.plot(
                                        time_vec,
                                        dendrite.synaptic_inputs[spd].phi_spd,
                                        label="SPD"
                                        )

        if ref==True:
            plt.plot(time_vec,refractory,':',color='r',label='refractory signal')

        ## add input/output spikes
        if spikes==True and hasattr(net,'spikes'):
            s_th = neurons[0].neuron.s_th
            spike_times = neurons[0].neuron.spike_times
            if len(spike_times) > 0:
                plt.plot(spike_times,np.ones(len(spike_times))*s_th,'xk', markersize=8, 
                         label='neuron fires')
                plt.axhline(y = s_th, color = 'purple', 
                            linestyle = '--',label='Firing Threshold')
            if input:
                plt.plot(
                    input.spike_arrays[1],np.zeros(len(input.spike_arrays[1])),
                    'xr', markersize=8, label='input event'
                    )
        if S == True:
            plt.plot(time_vec,signal,  color='#1f77b4',linewidth=4)
        plt.xlabel("Simulation Time (ns)",fontsize=18)
        plt.ylabel("Signal (Ic)",fontsize=18)
        plt.subplots_adjust(bottom=.25)
        plt.title(title,fontsize=12)
        if legend==True:
            if legend_out==True:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1.01))
                plt.subplots_adjust(right=.8)
                # plt.subplots_adjust(bottom=.15)
            else:
                plt.legend(loc=1)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()


def arbor_activity(
        node,net,
        phir=False,size=(12,6),norm_soma=False,show=True,title=None,spikes=False
        ):
    '''
    Plots signal and optional flux over dendritic structure
     - syntax
        -> SuperNode.plot_arbor_activity(net,phir=True)
    '''
    plt.style.use('seaborn-muted')
    # print(plt.__dict__['pcolor'].__doc__)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    t = net.t
    # print(node.dendrites)
    plt.figure(figsize=size)
    signals = []
    layer_sizes = []
    for i,layer in enumerate(node.dendrites):
        count = 0
        for j,group in enumerate(layer):
            for k,dend in enumerate(group):
                signals.append(dend.s)
                # print(dend.name)
                count+=1
        layer_sizes.append(count)
    max_s = np.max(signals)
    min_s = np.min(signals)

    layers = []
    x_ticks = []
    x_labels = []
    ys = []
    for i,layer in enumerate(node.dendrites[::-1]):
        count=0
        groups = []
        for j,group in enumerate(layer):
            g = []
            for k,dend in enumerate(group):
                signal=dend.s
                x=t+net.tf*i*1.1
                if i == 0:
                    y_offset = (count*max_s)*1.2+max_s*j*1.5
                    y = signal + y_offset
                    y2 = dend.phi_r + y_offset
                    ys.append(np.max(y))
                else:
                    y_offset = layers[i-1][count]-max_s*.5
                    y = signal + y_offset
                    y2 = dend.phi_r + y_offset
                g.append(y_offset+max_s*.7)

                if layer_sizes[-i] < 30:
                    lw = 3
                else:
                    lw = 3 # 1/layer_sizes[-i]

                if 'soma' in dend.name:
                    s_factor=1
                    plot = plt.plot(
                        x[:min(len(x),len(y))],y[:min(len(x),len(y))],
                        label=dend.name[18:],linewidth=lw,color=colors[0]
                        )
                    if spikes==True:
                        print("arbor plot spikes to come :)")
                        # if len(dend.spike_times) > 0:
                            # convert = net.time_params['t_tau_conversion']
                            # print(convert)
                            # spike_t = dend.spike_times/convert
                            # spike_signals = []
                            # for spike in spike_t:
                            #     spot = int(spike/net.dt)
                            #     spread = int(5/net.dt)
                            #     spike_signals.append(
                            #         np.max(
                            #         dend.s[np.max([0,spot-spread]):spot+spread]
                            #         )
                            #         )
                            # times = np.array(dend.spike_times)/(net.tf)
                            # times = times*(1+(net.tf-convert)/net.tf)
                            # times *= 1.0155
                            # plt.plot(
                            #     times+np.min(x),
                            #     spike_signals+np.min(y),'xk', 
                            #     markersize=4, 
                            #     label='neuron fires'
                            #     )
                else:
                    plot = plt.plot(
                        x[:min(len(x),len(y))],y[:min(len(x),len(y))],label=dend.name[18:],linewidth=lw,
                        color=colors[(dend.branch+1)%len(colors)]
                        )
                color = plot[0].get_color()
                if phir==True:
                    plt.plot(x[:min(len(x),len(y2))],y2[:min(len(x),len(y2))],'--',color=color)
                count+=1
            groups.append(np.mean(g))
        x_ticks.append(net.tf*i*1.1+.5*net.tf)
        x_labels.append(f"layer {len(node.dendrites)-(i+1)}")
        layers.append(groups)        
    x_labels[-1] += " (soma)"
    # plt.xlim(-.3*net.tf,net.tf*((len(node.dendrites)-1)*1.1+1.3))
    plt.xticks(x_ticks,x_labels,fontsize=18)

    plt.yticks([])
    #max_s*(np.max(layer_sizes)+len(node.dendrites[-1])-.5))
    # plt.ylim(-1*max_s,np.max(ys)+1*max_s)
    plt.ylabel(f"Signal Range = [{np.round(min_s,2)},{np.round(max_s,2)}]",
               fontsize=18)
    if title==None:
        plt.title("Dendritic Arbor Activity",fontsize=20)
    else:
        plt.title(title,fontsize=20)
    # plt.legend()
    if show==True:
        plt.show()
    


def structure(node):
    '''
    Plots arbitrary neuron structure
        - syntax
            -> SuperNode.plot_structure()
        - Weighting represented in line widths
        - Dashed lines inhibitory
        - Star is cell body
        - Dots are dendrites
    '''
    # import matplotlib.colors as mcolors
    # colors = mcolors.viridis
    # c_names = list(colors) + list(colors) + list(colors)

    # from matplotlib import cm

    # from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    # # color_map = cm.get_cmap('viridis', 3)
    # color_map = cm.get_cmap('tab10', 8)
    
    plt.style.use('seaborn-muted')
    # print(plt.__dict__['pcolor'].__doc__)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    arbor = node.dendrites
    strengths = node.weights

    Ns = []
    for i,a in enumerate(arbor):
        count = 0
        lsts = sum(type(el)== type([]) for el in a)
        if lsts > 0:
            for j in range(lsts):
                count+=len(arbor[i][j])
        else: count = len(a)
        Ns.append(count)
    m=max(Ns)
    
    Y = [[] for i in range(len(arbor))]
    G = []

    Ydot = []
    Xdot = []
    X_synapses = []
    Y_synapses = []
    syn_values = []
    dots = []
    x_ticks = []
    x_labels = []
    x_factor = 1
    y_factor = 5
    # colors = ['r','b','g']
    for i,layer in enumerate(node.dendrites[::-1]):
        count=0
        groups = []
        for j,group in enumerate(layer):
            g = []
            for k,dend in enumerate(group):
                x = 1 + i*x_factor
                if i==0:
                    y = np.round(1+count*y_factor,2)
                    Y[i].append(y)
                elif i==len(arbor)-1:
                    y = np.round(np.mean(G[i-1]),2)
                    Y[i].append(y)
                else:
                    y = G[i-1][count]
                    Y[i].append(y)

                Xdot.append(x)
                Ydot.append(y)
                
                syns = len(list(dend.synaptic_inputs))
                
                if syns>0:
                    if syns>1:
                        y_space = np.arange(y-.5,y+.501,1/(syns-1))
                    else:
                        y_space = [y]
                    for s,syn in enumerate(dend.synaptic_inputs):
                        X_synapses.append(x-.1)
                        Y_synapses.append(y_space[s])
                        idx = list(dend.synaptic_connection_strengths.keys())[s]
                        syn_values.append(dend.synaptic_connection_strengths[idx])
                
                if hasattr(dend, 'branch'):
                    branch = dend.branch
                else:
                    branch = None

                if hasattr(dend,'output_connection_strength'):
                    output = dend.output_connection_strength
                else:
                    output=None

                dot = [x,y,i,j,k,count,branch,output]
                dots.append(dot)
                g.append(y)
                count+=1
                x_ticks.append(x)
                x_labels.append(f"layer {len(node.dendrites)-(i+1)}")
            groups.append(np.mean(g))
        G.append(groups)
    plt.figure(figsize=(8,5))

    # labels = ['Basal','Proximal']
    count=0
    for i,dot1 in enumerate(dots):
        
        for ii,dot2 in enumerate(dots):
            if dot1[3] == dot2[5] and dot1[2] == dot2[2]-1:
                to_dot = dot2
        x1 = dot1[0]
        x2 = to_dot[0]
        y1 = dot1[1]
        y2 = to_dot[1]

        if dot1[6] != None:
            # color = color_map.colors[dot1[6]]
            color = colors[(dot1[6]+1)%len(colors)]
        else:
            color = 'k'
        if dot1[7] != None and dot1[7] != 0:
            width = np.max([int(np.abs(dot1[7]*5)),1])
        else:
            width = .01
        # print(i,dot1,'-->',to_dot)

        line_style = '-'
        if dot1[7] != None and dot1[7]<0:
            line_style='--'

        if to_dot[2]==len(arbor)-1 and to_dot!=dot1:
            # print("to soma")
            
            plt.plot(
                [x1,x2],[y1,y2],linestyle=line_style,
                color=color,linewidth=width,label=f'branch {dot1[6]}' #f"{labels[count]} Branch" 
                )
            count+=1
        else:
            plt.plot(
                [x1,x2],[y1,y2],
                linestyle=line_style,
                color=color,
                linewidth=width*1.5
                )
    
    if sum(Ns) > 30:
        ms = np.array([30,20,15,8])*15/sum(Ns)
        syn_values = np.array(syn_values)*8*15/sum(Ns)

    else:
        # ms = np.array([30,20,15,8])
        ms = np.array([30,20,11,11]) #NICEplot
        syn_values = np.array(syn_values)*200


    plt.plot(Xdot[-1],Ydot[-1],'*k',ms=ms[0])
    plt.plot(Xdot[-1],Ydot[-1],'*y',ms=ms[1],label='Soma')
    plt.plot(Xdot[0],Ydot[0],'ok',ms=ms[2],label='Dendrites')
    plt.plot(Xdot[1:-1],Ydot[1:-1],'ok',ms=ms[2])


    syn_colors = []
    for s in syn_values:
        if s > 0:
            syn_colors.append('r')
        else:
            syn_colors.append('b')
    syn_values = np.abs(syn_values)
    plt.scatter(
        X_synapses,
        Y_synapses,
        marker='>', 
        c=syn_colors,
        s=75,#syn_values,
        label='Synapses',
        )

    # plt.legend(borderpad=1)


    plt.legend(borderpad=1,markerscale=.7)

    x_labels[-1] += " (soma)"
    plt.xticks(x_ticks,x_labels,fontsize=14)
    plt.xlim(1-.1*len(arbor),len(arbor)*1.1)
    plt.ylim(1-y_factor,1+m*y_factor)
    plt.yticks([])
    plt.ylabel("Dendrites",fontsize=18)
    plt.xlabel("Layers",fontsize=18)
    plt.title("Dendritc Arbor",fontsize=20)
    plt.tight_layout()
    plt.show()



def plot_basal_proximal(node,net,phir=False,dend=True,title=None,
                        input=None,input_2=None,weighting=True,docstring=False):
    '''
    Plots signal activity for a given network or neuron
        - phir      -> plot phi_r of soma and phi_r thresholds
        - dend      -> plot dendritic signals
        - input     -> mark moments of input events with red spikes
        - weighting -> weight dendritic signals by their connection strength
    '''
    if docstring == True:
        print(node.plot_neuron_activity.__doc__)
        return
    signal = node.dendrites[0][0][0].s
    ref = node.neuron.dend__ref.s
    phi_r = node.dendrites[0][0][0].phi_r

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.plot(net.t,signal,  label='soma signal', linewidth=2.5)
    dend_names = ['basal', 'proximal', 'inhibitory']
    if dend:
        for i,layer in enumerate(node.dendrites):
            for j,branch in enumerate(layer):
                for k,dendrite in enumerate(branch):
                    if i == 0 and j == 0 and k ==0:
                        pass
                    else:
                        # print(dendrite.__dict__.keys())
                        # print(dendrite.external_connection_strengths)
                        if weighting == True:
                            weight = dendrite.weights[i-1][j][k]
                            dend_s = dendrite.s*weight
                        else:
                            dend_s = dendrite.s

                        plt.plot(net.t,dend_s,'--', label='w * '+dend_names[k])
    colors = ['r','g']
    spike_times = net.neurons[node.neuron.name].spike_t
    # print(spike_times)

    plt.plot(
        spike_times,np.ones(len(spike_times))*node.neuron.s_th,
        'xk', 
        markersize=8, 
        label=f'neuron fires'
        )

    plt.axhline(
        y = node.neuron.s_th, 
        color = 'purple', 
        linestyle = ':',
        label='Firing Threshold'
        )
    
    if input:
        plt.plot(
            input.spike_arrays[1],
            np.zeros(len(input.spike_arrays[1])),
            'x',color='orange', 
            markersize=5, 
            label='proximal input event'
            )
    if input_2:
        plt.plot(
            input_2.spike_arrays[1],
            np.zeros(len(input_2.spike_arrays[1])),
            'xg',
            markersize=5,
            label='basal input event'
            )
    # plt.plot(net.t,phi_r,  label='phi_r (soma)')
    plt.xlabel("Simulation Time (ns)")
    plt.ylabel("Signal (Ic)")
    if title:
        plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=.8)
    plt.subplots_adjust(bottom=.15)
    # plt.legend()
    plt.show()









def plot_MNIST_nodes(nodes,digit,sample,run,name,path):
    import os
    try:
        os.makedirs(path+name+'plots/')    
    except FileExistsError:
        pass
    for n,node in enumerate(nodes):
        lays = [[] for _ in range(len(node.dendrites))]
        phays = [[] for _ in range(len(node.dendrites))]
        for l,layer in enumerate(node.dendrites):
            for g,group in enumerate(layer):
                for d,dend in enumerate(group):
                    lays[l].append(dend.s)
                    phays[l].append(dend.phi_r)
        # try:
        #     plt.style.use('seaborn-muted')  
        # except MatplotlibDeprecationWarning:
        #     plt.style.use("seaborn-v0_8")

        # plt.style.use('seaborn-muted')
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure(figsize=(8,4))
        for l,lay in enumerate(lays):
            if l == 0:
                lw = 4
            else:
                lw = 2
            plt.plot(
                np.mean(lay,axis=0),
                linewidth=lw,
                color=colors[l],
                label=f'Layer {l} Mean Signal'
                )
            plt.plot(
                np.mean(phays[l],axis=0),
                '--',
                linewidth=.5,
                color=colors[l],
                # label=f'Layer {l} Mean Flux'
                )
        plt.legend(loc='upper right')
        plt.title(f'Node {n} - Digit {digit} Sample {sample} - Run {run}')
        plt.savefig(path+name+f'plots/node_{n}_digit_{digit}_sample_{sample}_run_{run}.png')
        plt.close()


# =============================================================================
# End plots added by Ryan
# =============================================================================






































# =============================================================================
#  Jeff Plots
# =============================================================================

# =============================================================================
#  dendrite
# =============================================================================

def plot_dendrite(d_i):
    
    _max_min_plotting_threshold = 1e-3
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 2*fig_size[1]) )
    plt.suptitle( 'soen dendrite: {}, loops present = {}\nib = {:5.3f}\nbeta_c = {:5.3f}, beta_di/2pi = {:5.3e}, tau_di = {:3.1e}ns'.format( d_i.name, d_i.loops_present, d_i.bias_current, d_i.beta_c, d_i.circuit_betas[-1]/(2*np.pi), d_i.tau_di ) )
    
    d_params = dend_load_arrays_thresholds_saturations(f'default_{d_i.loops_present}')
    ib__list = d_params["ib__list"]
    phi_r__array = d_params["phi_r__array"]
    i_di__array = d_params["i_di__array"]
    r_fq__array = d_params["r_fq__array"]
    phi_th_plus__vec = d_params["phi_th_plus__vec"]
    phi_th_minus__vec  = d_params["phi_th_minus__vec"]
    s_max_plus__vec  = d_params["s_max_plus__vec"]
    s_max_minus__vec = d_params["s_max_minus__vec"]
    s_max_plus__array = d_params["s_max_plus__array"]
    s_max_minus__array  = d_params["s_max_minus__array"]
    
    
    ib_vec = np.asarray(ib__list)
    
    _time_vec = d_i.output_data['tau_vec']/d_i.time_params['t_tau_conversion']
    
    # applied flux
    ax[0].plot(_time_vec, d_i.output_data['phi_r'], color = colors['blue3'], label = 'phi_r') 
    
    # plot maximum and minimum input flux
    _max = np.max(d_i.output_data['phi_r'])
    if _max > _max_min_plotting_threshold: # only proceed if it is above noise
        _ind_max = ( np.abs( d_i.output_data['phi_r'] - _max ) ).argmin()
        ax[0].plot(_time_vec[_ind_max], _max, 'x', color = colors['red5'], label = 'max = {:5.3f}'.format(_max)) 
        
    _min = np.min(d_i.output_data['phi_r'])
    if _min < -_max_min_plotting_threshold: # only proceed if it is below noise
        _ind_min = ( np.abs( d_i.output_data['phi_r'] - _min ) ).argmin()
        ax[0].plot(_time_vec[_ind_min], _min, 'x', color = colors['red5'], label = 'min = {:5.3f}'.format(_min)) 
    
    # plot dendrite flux thresholds at this value of ib
    if _max > _max_min_plotting_threshold:
        _ind_ib = ( np.abs( ib_vec[:] - d_i.ib )).argmin()
        ax[0].plot([_time_vec[0],_time_vec[-1]], [phi_th_plus__vec[_ind_ib],phi_th_plus__vec[_ind_ib]], ':', color = colors['green5'], label = 'phi+ = {:5.3f}'.format(phi_th_plus__vec[_ind_ib])) 
        
    if _min < -_max_min_plotting_threshold:
        _ind_ib = ( np.abs( ib_vec[:] - d_i.ib )).argmin()
        ax[0].plot([_time_vec[0],_time_vec[-1]], [phi_th_minus__vec[_ind_ib],phi_th_minus__vec[_ind_ib]], ':', color = colors['green5'], label = 'phi- = {:5.3f}'.format(phi_th_minus__vec[_ind_ib])) 
    
    ax[0].set_ylabel(r'$\phi_r$ [$\Phi_0$]')          
    ax[0].legend(loc = 'lower right')
    
    # accumulated signal
    ax[1].plot(_time_vec, d_i.output_data['s'], color = colors['blue3'], label = 's_di')
    _max = np.max(d_i.output_data['s'])
    _ind_max = ( np.abs( d_i.output_data['s'] - _max ) ).argmin()
    ax[1].plot(_time_vec[_ind_max], _max, 'x', color = colors['red5'], label = '{:5.3f}'.format(_max))
    ax[1].set_ylabel(r'$s$ [$I_c$]')
    ax[1].legend(loc = 'lower right')
    # ax[1].set_ylim([-2,2])
    
    ax[1].set_xlabel(r'Time [ns]')
    # ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    # ax.grid(which='minor', linewidth = 0.15)
            
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # plt.show() 
    
    return

def recursive_dendrite_plotter(dendrite,already_plotted):
        
    if dendrite.name not in already_plotted:
        
        already_plotted.append(dendrite.name)
        dendrite.plot()
        
    for dend_key in dendrite.dendritic_inputs:
        
        already_plotted = recursive_dendrite_plotter(dendrite.dendritic_inputs[dend_key],already_plotted)
    
    return already_plotted
    
def plot_dendrite__compare_soen_ode(d_i, data_dict, dt_soen, chi_squared__drive, chi_squared__signal, fig_case):
    
    params = data_dict['params']
    ib = data_dict['ib']
    phi_r = data_dict['phi_r']
    idi = data_dict['idi']
    _time_vec = data_dict['time_vec']
    time_vec = params['tau_0']*_time_vec*1e9
    
    if fig_case == 'publication':
        fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    elif fig_case == 'presentation':
        fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 2*fig_size[1]) )
    _str = 'comparing soen to ode dendrite, loops present = {}\n'.format(d_i.loops_present)
    _str = '{}ib = {:5.3f}, dt = {:5.2f}ns\nbeta_c = {:5.3f}, beta_di/2pi = {:5.3e}, tau_di = {:5.3e}ns\n'.format(_str, d_i.time_params['dt'], d_i.bias_current, d_i.beta_c, d_i.circuit_betas[-1]/(2*np.pi), d_i.tau_di )
    if d_i.output_data['time_to_solution'] > 0:
        _str = '{}t_sim_ode/t_sim_soen = {:5.3e}'.format(_str,data_dict['time_to_solution']/d_i.output_data['time_to_solution'])
    else:
        _str = '{}{}'.format(_str,'t_sim_soen to short to measure')
    _str = '{}\node_accuracy = {:3.1e} (abs), {:3.1e} (rel), dt_soen = {:5.3e}ns'.format(_str,data_dict['params']['absolute_tolerance'],data_dict['params']['relative_tolerance'],dt_soen)
    _str = '{}\nchi_drive = {:4.2e}, chi_signal = {:4.2e}'.format(_str,chi_squared__drive,chi_squared__signal)
        
    plt.suptitle( _str )
    
    check = 0
    if d_i.bias_current != ib:
        print('error: the two simulations had different values of ib')
        check += 1
    if d_i.beta_c != params['beta_c']:
        print('error: the two simulations had different values of beta_c')
        check += 1
    if d_i.circuit_betas[-1] != params['beta_di']:
        print('error: the two simulations had different values of beta_di')
        check += 1
    if d_i.tau_di != params['tau_0']*params['tau_di']*1e9:
        print('error: the two simulations had different values of tau_di. the normalized difference was {:5.3e}'.format( (2*np.abs((d_i.tau_di-params['tau_0']*params['tau_di']*1e9))/(d_i.tau_di+params['tau_0']*params['tau_di']*1e9)) ))
        check += 1
    if check == 0:
        print('the two simulations had the same parameters')
        
    _time_vec__soen = d_i.time_params['time_vec'][:]
    ax[0].plot(time_vec, phi_r, linestyle = 'solid', color = colors['red3'], label = 'ode')
    ax[0].plot(_time_vec__soen, d_i.output_data['phi_r'], linestyle = 'dashdot', color = colors['blue3'], label = 'soen')  
    ax[0].set_ylabel(r'$\phi$ [$\Phi_0$]')          
    ax[0].legend(loc = 'lower right')
    
    ax[1].plot(time_vec, idi, linestyle = 'solid', color = colors['red3'], label = 'ode') 
    ax[1].plot(_time_vec__soen, d_i.output_data['s'], linestyle = 'dashdot', color = colors['blue3'], label = 'soen') 
    ax[1].set_ylabel(r'$s$ [$I_c$]')
    ax[1].legend(loc = 'lower right')
    
    ax[1].set_xlabel(r'Time [ns]')
    # ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    # ax.grid(which='minor', linewidth = 0.15)
            
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # plt.show() 
    
    return

def plot_dendrite__compare_soen_ode__multi(dendrite_object, beta_di__vec, tau_di__vec, dt_soen__vec, chi_squared__array, chi_squared_drive__array, t_sim__soen__array, data_dicts__soen, data_dicts__ode, fig_case):
    
    
    _t1 = 40 # 0
    _t2 = 150 # data_dicts__soen[0][0][0]['time_vec'][-1]
        
    data_dict__base = data_dicts__ode[0][0]
    params = data_dict__base['params']
    ib = data_dict__base['ib']
       
    
    if fig_case == 'publication':
        fig, ax = plt.subplots(nrows = len(beta_di__vec)+1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 2*fig_size[1]) )
    elif fig_case == 'presentation':
        fig, ax = plt.subplots(nrows = len(beta_di__vec)+1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 2*fig_size[1]) )
    _str = 'comparing soen to ode dendrite, loops present = {}\n'.format(dendrite_object.loops_present)
    _str = '{}ib = {:5.3f}, beta_c = {:5.3f}'.format(_str, dendrite_object.bias_current, dendrite_object.beta_c )
        
    plt.suptitle( _str )
    
    check = 0
    if dendrite_object.bias_current != ib:
        print('error: the two simulations had different values of ib')
        check += 1
    if dendrite_object.beta_c != params['beta_c']:
        print('error: the two simulations had different values of beta_c')
        check += 1
    if dendrite_object.circuit_betas[-1] != params['beta_di']:
        print('error: the two simulations had different values of beta_di')
        check += 1
    if check == 0:
        print('the two simulations had the same parameters')
        
    color_array__ode = ['grey3','grey6','grey9','black']
    color_array__soen = [ ['blue1','red1','green1','yellow1'] , ['blue3','red3','green3','yellow3'] ]
    linestyle__soen = ['dotted','dashed']
    for ii in range(len(tau_di__vec)):
        
        
        _time_vec__ode = data_dicts__ode[0][ii]['time_vec'][:]*params['tau_0']*1e9
        
        _ind1__ode = index_finder(_t1,_time_vec__ode)
        _ind2__ode = index_finder(_t2,_time_vec__ode)
        
        ax[0].plot(_time_vec__ode[_ind1__ode:_ind2__ode], data_dicts__ode[0][ii]['phi_r'][_ind1__ode:_ind2__ode], linestyle = 'solid', color = colors[color_array__ode[ii]], label = 'ode')
        for kk in range(len(beta_di__vec)):
            _time_vec__ode = data_dicts__ode[kk][ii]['time_vec'][:]*params['tau_0']*1e9
            _ind1__ode = index_finder(_t1,_time_vec__ode)
            _ind2__ode = index_finder(_t2,_time_vec__ode)
            ax[kk+1].plot(_time_vec__ode[_ind1__ode:_ind2__ode], data_dicts__ode[kk][ii]['idi'][_ind1__ode:_ind2__ode], linestyle = 'solid', color = colors[color_array__ode[ii]], label = 'beta_di/2pi = {:3.1e}'.format(beta_di__vec[kk]/(2*np.pi)))
                
        for jj in range(len(dt_soen__vec)):
            
            dt = dt_soen__vec[jj]
            
            # drive signal
            _time_vec__soen = data_dicts__soen[0][ii][jj]['time_vec'][:]
            _ind1__soen = index_finder(_t1,_time_vec__soen)
            _ind2__soen = index_finder(_t2,_time_vec__soen)
            ax[0].plot(_time_vec__soen[_ind1__soen:_ind2__soen], data_dicts__soen[0][ii][jj]['phi_r'][_ind1__soen:_ind2__soen], linestyle = linestyle__soen[jj], color = colors[color_array__soen[jj][ii]], label = 'soen, dt = {:3.1f}ns, chi_drive = {:4.2e}'.format(dt,chi_squared_drive__array[0][ii][jj]))  
            ax[0].set_ylabel(r'$\phi_r$ [$\Phi_0$]')   
                
            # beta_di 
            for kk in range(len(beta_di__vec)):
                _label_str = 'soen, tau_di = {:4.2e}ns, dt = {:3.1f}, chi^2 = {:4.2e}, t_ratio = {:4.2e}'.format(tau_di__vec[ii], dt_soen__vec[jj], chi_squared__array[kk,ii,jj], data_dicts__ode[kk][ii]['time_to_solution']/t_sim__soen__array[kk][ii][jj])
                ax[kk+1].plot(_time_vec__soen[_ind1__soen:_ind2__soen], data_dicts__soen[kk][ii][jj]['s'][_ind1__soen:_ind2__soen], linestyle = 'dashdot', color = colors[color_array__soen[jj][ii]], label = _label_str) 
                ax[kk+1].set_ylabel(r'$s$ [$I_c$]')
       
    for kk in range(len(beta_di__vec)+1):                 
        ax[kk].legend(loc = 'right')
        
    ax[-1].set_xlabel(r'Time [ns]')
                
    plt.subplots_adjust(wspace = 0.3, hspace = 0)
    
    return

def plot_dendrite__compare_soen_ode__error_vs_dt(dendrite_object, beta_di__vec, tau_di__vec, num_pulses__list, num_cases, dt_soen__vec, chi_squared__array, chi_squared_drive__array, t_sim__ode__array, t_sim__soen__array, T_sim__array, data_dict__ode):
    
    _t_min_soen = np.min(np.min(np.min(np.min(np.min(t_sim__soen__array[:,:,:,:,:])))))
    
    color_list = [ ['blue5','blue3','blue1'] , ['red5','red3','red1'] , ['green5','green3','green1'] , ['yellow5','yellow3','yellow1'] ]
    color_list__drive = [ ['bluegrey1','bluegrey2','bluegrey3'] , ['redgrey1','redgrey2','redgrey3'] , ['greengrey1','greengrey2','greengrey3'] , ['yellowgrey1','yellowgrey2','yellowgrey3'] ]
    
    for ii in range(len(num_pulses__list)):
        for jj in range(num_cases):
    
            fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , p['golden_ratio']*fig_size[1]) )
            _str = 'comparing soen to ode dendrite, loops present = {}\n'.format(dendrite_object.loops_present)
            _str = '{}ib = {:5.3f}\nbeta_c = {:5.3f}, num_pulses = {:d}, case = {:d}\n'.format(_str, dendrite_object.bias_current, dendrite_object.beta_c, num_pulses__list[ii], jj+1 )
            _str = '{}ode_accuracy = {:3.1e} (abs), {:3.1e} (rel)'.format(_str,data_dict__ode['params']['absolute_tolerance'],data_dict__ode['params']['relative_tolerance'])
            plt.suptitle( _str )
            
            for pp in range(len(beta_di__vec)):
                for qq in range(len(tau_di__vec)):
                    ax.loglog(dt_soen__vec, chi_squared_drive__array[pp,qq,ii,jj], '-o', color = colors[color_list__drive[pp][qq]], label = 'drive')   
                    ax.loglog(dt_soen__vec, chi_squared__array[pp,qq,ii,jj], '-o', color = colors[ color_list[pp][qq] ], label = 'beta = {:3.1e}, tau = {:3.1e}ns'.format(beta_di__vec[pp],tau_di__vec[qq]))       
            ax.set_ylabel(r'$\chi^2$') 
            ax.set_xlabel(r'$dt$ [ns]') 
            ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
            ax.grid(which='minor', linewidth = 0.15)
            # ax.set_ylim([1e-7,1])
            ax.legend() # loc = 'lower right'
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , p['golden_ratio']*fig_size[1]) )
            plt.suptitle( _str )
            for pp in range(len(beta_di__vec)):
                for qq in range(len(tau_di__vec)):
                    time_to_solution__soen__vec = t_sim__soen__array[pp,qq,ii,jj,:]
                    time_to_solution__soen__vec[ np.where( time_to_solution__soen__vec < _t_min_soen ) ] = _t_min_soen
                    ax.loglog(dt_soen__vec, time_to_solution__soen__vec/t_sim__ode__array[pp,qq,ii,jj], '-o', color = colors[ color_list[pp][qq] ], label = 'beta = {:3.1e}, tau = {:3.1e}ns'.format(beta_di__vec[pp],tau_di__vec[qq]))     
            ax.set_ylabel(r'$t_{sim}^{soen}/t_{sim}^{ode}$') 
            ax.set_xlabel(r'$dt$ [ns]')
            ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
            ax.grid(which='minor', linewidth = 0.15)
            ax.legend() # loc = 'lower right'
               
    # chi squared and time ratio vs simuated time
    _ind_100ps = ( np.abs( dt_soen__vec - 0.1 ) ).argmin()
    _ind_1ns = ( np.abs( dt_soen__vec - 1 ) ).argmin()   
    
    color_list_1 = ['blue1','blue3','blue5']
    color_list_2 = ['yellow1','yellow3','yellow5']
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , p['golden_ratio']*fig_size[1]) ) 
    for pp in range(len(beta_di__vec)):
        for qq in range(len(tau_di__vec)):
            
            ax[0].loglog(T_sim__array[pp,qq,:,0]*1e-3, chi_squared__array[pp,qq,:,0,_ind_1ns], '-o', color = colors[color_list_1[pp]], label = 'dt_soen = 1ns, beta = {:3.1e}, tau = {:3.1e}ns'.format(beta_di__vec[pp], tau_di__vec[qq]))  
            ax[0].loglog(T_sim__array[pp,qq,:,0]*1e-3, chi_squared__array[pp,qq,:,0,_ind_100ps], '-o', color = colors[color_list_2[pp]], label = 'dt_soen = 100ps, beta = {:3.1e}, tau = {:3.1e}ns'.format(beta_di__vec[pp], tau_di__vec[qq]))  
            ax[1].loglog(T_sim__array[pp,qq,:,0]*1e-3, t_sim__soen__array[pp,qq,:,0,_ind_1ns]/t_sim__ode__array[pp,qq,:,0], '-o', color = colors[color_list_1[pp]], label = 'dt_soen = 1ns, beta = {:3.1e}, tau = {:3.1e}ns'.format(beta_di__vec[pp], tau_di__vec[qq]))     
            ax[1].loglog(T_sim__array[pp,qq,:,0]*1e-3, t_sim__soen__array[pp,qq,:,0,_ind_100ps]/t_sim__ode__array[pp,qq,:,0], '-o', color = colors[color_list_2[pp]], label = 'dt_soen = 100ps, beta = {:3.1e}, tau = {:3.1e}ns'.format(beta_di__vec[pp], tau_di__vec[qq])) 
    
    ax[0].set_ylabel(r'$\chi^2$') 
    ax[1].set_ylabel(r'$t_{sim}^{soen}/t_{sim}^{ode}$') #[$x10^4$]
    ax[1].set_xlabel(r'$T_{sim}$ [$\mu$s]')
    for ii in range(2):
        ax[ii].grid(which = 'both', axis = 'both', color = colors['grey4'])
        ax[ii].grid(which='minor', linewidth = 0.15)
        ax[ii].legend() # loc = 'lower right'
        
    # ax[0].set_ylim([1e-5,1e-3])
    # ax[1].set_ylim([1e-5,1e-3])
    
    plt.subplots_adjust(wspace = 0.3, hspace = 0)
    
    return

def plot_multiple_dendrites(dendrite_list):
    
    _max_min_plotting_threshold = 1e-3
    ib__list__ri, phi_r__array__ri, i_di__array__ri, r_fq__array__ri, phi_th_plus__vec__ri, phi_th_minus__vec__ri, s_max_plus__vec__ri, s_max_minus__vec__ri, s_max_plus__array__ri, s_max_minus__array__ri = dend_load_arrays_thresholds_saturations('default_ri')
    ib__list__rtti, phi_r__array__rtti, i_di__array__rtti, r_fq__array__rtti, phi_th_plus__vec__rtti, phi_th_minus__vec__rtti, s_max_plus__vec__rtti, s_max_minus__vec__rtti, s_max_plus__array__rtti, s_max_minus__array__rtti = dend_load_arrays_thresholds_saturations('default_rtti')
    
    _num_dend = len(dendrite_list)
    color_list = colors_gist(np.linspace(.1, 1,_num_dend))
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 2*fig_size[1]) )
    _str = ''
    for ii in range(_num_dend):
        _str = '{}{}, '.format(_str,dendrite_list[ii].name)
    
    plt.suptitle(_str[:-2])
    
    for ii in range(_num_dend):
        d_i = dendrite_list[ii]
    
        # applied flux
        ax[0].plot(d_i.output_data['time_vec'], d_i.output_data['phi_r'], color = color_list[ii], label = '{}'.format(d_i.name)) 
        
    
        # plot maximum and minimum input flux
        _max = np.max(d_i.output_data['phi_r'])
        if _max > _max_min_plotting_threshold: # only proceed if it is above noise
            _ind_max = ( np.abs( d_i.output_data['phi_r'] - _max ) ).argmin()
            ax[0].plot(d_i.output_data['time_vec'][_ind_max], _max, 'x', color = color_list[ii], label = 'max = {:5.3f}'.format(_max)) 
            
        _min = np.min(d_i.output_data['phi_r'])
        if _min < -_max_min_plotting_threshold: # only proceed if it is below noise
            _ind_min = ( np.abs( d_i.output_data['phi_r'] - _min ) ).argmin()
            ax[0].plot(d_i.output_data['time_vec'][_ind_min], _min, 'x', markersize = 4, color = color_list[ii], label = 'min = {:5.3f}'.format(_min)) 
        
        # plot dendrite flux thresholds at this value of ib
        if d_i.loops_present == 'ri':
            ib_vec = ib__list__ri
            phi_th_plus__vec = phi_th_plus__vec__ri
            phi_th_minus__vec = phi_th_minus__vec__ri
        elif d_i.loops_present == 'rtti':
            ib_vec = ib__list__rtti
            phi_th_plus__vec = phi_th_plus__vec__rtti
            phi_th_minus__vec = phi_th_minus__vec__rtti
        if _max > _max_min_plotting_threshold:
            _ind_ib = ( np.abs( ib_vec[:] - d_i.ib )).argmin()
            ax[0].plot([d_i.output_data['time_vec'][0],d_i.output_data['time_vec'][-1]], [phi_th_plus__vec[_ind_ib],phi_th_plus__vec[_ind_ib]], ':', color = color_list[ii], label = 'phi+ = {:5.3f}'.format(phi_th_plus__vec[_ind_ib])) 
            
        if _min < -_max_min_plotting_threshold:
            _ind_ib = ( np.abs( ib_vec[:] - d_i.ib )).argmin()
            ax[0].plot([d_i.output_data['time_vec'][0],d_i.output_data['time_vec'][-1]], [phi_th_minus__vec[_ind_ib],phi_th_minus__vec[_ind_ib]], ':', color = color_list[ii], label = 'phi- = {:5.3f}'.format(phi_th_minus__vec[_ind_ib])) 
    
        # accumulated signal
        ax[1].plot(d_i.output_data['time_vec'], d_i.s, color = color_list[ii], label = '{}'.format(d_i.name))
        
        # accumulated signal max
        _max = np.max(d_i.output_data['s'])
        _ind_max = ( np.abs( d_i.output_data['s'] - _max ) ).argmin()
        ax[1].plot(d_i.output_data['time_vec'][_ind_max], _max, 'x', color = color_list[ii], label = '{:5.3f}'.format(_max))
    
    ax[0].set_ylabel(r'$\phi_r$ [$\Phi_0$]')          
    ax[0].legend() # loc = 'lower right'
    
    ax[1].set_ylabel(r'$s$ [$I_c$]')
    ax[1].legend() # loc = 'lower right'
    # ax[1].set_ylim([-2,2])
    
    ax[1].set_xlabel(r'Time [ns]')
    # ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    # ax.grid(which='minor', linewidth = 0.15)
            
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # plt.show() 
    
    return

# =============================================================================
# end dendrite
# =============================================================================

# =============================================================================
# synapse
# =============================================================================

def plot_synapse(s_i):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 2*fig_size[1]) )
    plt.suptitle( 'soen synapse\ntau_rise = {:4.1f}ps, tau_fall = {:5.3f}ns, hotspot_duration = {:4.1f}ps'.format( 1e3*s_i.tau_rise, s_i.tau_fall, 1e3*s_i.hotspot_duration ) )
    
    _time_vec = s_i.time_vec
    ax.plot(_time_vec, s_i.output_data['phi_spd'], color = colors['blue3'])  
    ax.set_ylabel(r'$\phi_spd$ [$\Phi_0$]')  
    ax.set_xlabel(r'Time [ns]')
    # ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    # ax.grid(which='minor', linewidth = 0.15)
            
    # plt.subplots_adjust(wspace=0.3, hspace=0)
    # plt.show() 
    
    return
 
def plot_syn_dend(time_array, phi_spd, s_array, beta_di__vec, tau_di__vec, dt_soen__vec, fig_case):
    
    if fig_case == 'presentation':
        plt.rcParams['legend.fontsize'] = 8
    
    linestyle_list = ['solid','dashed'] # dotted
    color_array = [ ['blue3','red3','green3','yellow3'] , ['blue1','red1','green1','yellow1'] ]
    fig, ax = plt.subplots(nrows = len(beta_di__vec)+1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 2*fig_size[1]) )
    # plt.suptitle( 'soen dendrite, loops present = {}\nib = {:5.3f}\nbeta_c = {:5.3f}, beta_di/2pi = {:5.3e}, tau_di = {:3.1e}ns'.format( d_i.loops_present, d_i.bias_current, d_i.beta_c, d_i.circuit_betas[-1]/(2*np.pi), d_i.tau_di ) )
    
    ax[0].plot(time_array[0][0][0], phi_spd, color = colors['blue3'], label = 'phi_r')  
    ax[0].set_ylabel(r'$\phi_r$ [$\Phi_0$]')          
    ax[0].legend(loc = 'lower right')
    
    for ii in range(len(beta_di__vec)):
        _str1 = 'beta = {:3.1e}'.format(beta_di__vec[ii])
        for jj in range(len(tau_di__vec)):
            _str2 = 'tau = {:3.1e}ns'.format(tau_di__vec[jj])
            for kk in range(len(dt_soen__vec)):
                _str3 = 'dt = {:3.1f}ns'.format(dt_soen__vec[kk])
                
                if jj == 0 and kk == 0:
                    _str = '{}, {}, {}'.format(_str1,_str2,_str3)
                elif kk == 0:
                    _str = '{}, {}'.format(_str2,_str3)
                else:
                    _str = '{}'.format(_str3)
                
                ax[ii+1].plot(time_array[ii][jj][kk], s_array[ii][jj][kk], color = colors[color_array[kk][jj]], linestyle = linestyle_list[kk], label = _str) 
                ax[ii+1].set_ylabel(r'$s$ [$I_c$]')
                ax[ii+1].legend(loc = 'lower right')
    # ax[1].set_ylim([-2,2])
    
    ax[-1].set_xlabel(r'Time [ns]')
    # ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    # ax.grid(which='minor', linewidth = 0.15)
            
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # plt.show() 
    
    return

def plot_syn_dend_s_vs_ib(ib__vec, beta_di__vec, connection_strength, s_max__array):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 2*fig_size[1]) )
    plt.suptitle( 'soen synapse\nconnection strength = {:4.2f}'.format( connection_strength ) )
    
    color_list = ['blue3','green3','red3','yellow3']
    for ii in range(len(beta_di__vec)):
        ax.plot(ib__vec[:], s_max__array[ii,:], color = colors[color_list[ii]], label = 'beta_di = {:4.2e}'.format(beta_di__vec[ii]))  
    ax.set_ylabel(r'$s_{max}$ [$I_c$]')  
    ax.set_xlabel(r'ib [$I_c$]')
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
    ax.legend(loc = 'best')
            
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # plt.show() 
    
    return

def plot_syn_dend_burst_transfer_function(ib__vec, num_in_burst__vec, burst_frequency__vec, transfer_function__array, tau_di, beta_di):
    
    color_list = [['blue1','blue2','blue3','blue4','blue5'],['green1','green2','green3','green4','green5'],['yellow1','yellow2','yellow3','yellow4','yellow5'],['red1','red2','red3','red4','red5']]
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle('syn dend burst transfer function\ntau_di = {:4.2e}ns, beta_di = {:4.2e}'.format( tau_di, beta_di ) )
    for ii in range(len(ib__vec)):
        for jj in range(len(burst_frequency__vec)):
            ax.plot(num_in_burst__vec[:], transfer_function__array[ii,:,jj], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:4.2f}, freq = {:3.1f}/tau_di'.format(ib__vec[ii],burst_frequency__vec[jj]*tau_di))  
    ax.set_ylabel(r'$s_{max}$ [$I_c$]')  
    ax.set_xlabel(r'Num in burst')
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
    ax.legend(loc = 'best') 
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle('syn dend burst transfer function\ntau_di = {:4.2e}ns, beta_di = {:4.2e}'.format( tau_di, beta_di ) )
    for ii in range(len(ib__vec)):
        for jj in range(len(burst_frequency__vec)):
            ax.semilogx(num_in_burst__vec[:], transfer_function__array[ii,:,jj], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:4.2f}, freq = {:3.1f}/tau_di'.format(ib__vec[ii],burst_frequency__vec[jj]*tau_di))  
    ax.set_ylabel(r'$s_{max}$ [$I_c$]')  
    ax.set_xlabel(r'Num in burst')
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
    ax.legend(loc = 'best')
    
    return

def plot_syn_dend_rate_transfer_function(ib__vec, num_in_burst__vec, burst_frequency__vec, transfer_function__array, tau_di, beta_di):
    
    color_list = [['blue1','blue2','blue3','blue4','blue5'],['green1','green2','green3','green4','green5'],['yellow1','yellow2','yellow3','yellow4','yellow5'],['red1','red2','red3','red4','red5']]
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle('syn dend rate transfer function\ntau_di = {:4.2e}ns, beta_di = {:4.2e}'.format( tau_di, beta_di ) )
    for ii in range(len(ib__vec)):
        for jj in range(len(num_in_burst__vec)):
            ax.plot(tau_di*burst_frequency__vec[:], transfer_function__array[ii,:,jj], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:4.2f}, num = {:3.1f}'.format(ib__vec[ii],num_in_burst__vec[jj]))  
    ax.set_ylabel(r'$s_{max}$ [$I_c$]')  
    ax.set_xlabel(r'$f_{burst}$ [1/$\tau_{di}$]')
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
    ax.legend(loc = 'best') 
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle('syn dend rate transfer function\ntau_di = {:4.2e}ns, beta_di = {:4.2e}'.format( tau_di, beta_di ) )
    for ii in range(len(ib__vec)):
        for jj in range(len(num_in_burst__vec)):
            ax.semilogx(tau_di*burst_frequency__vec[:], transfer_function__array[ii,:,jj], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:4.2f}, num = {:3.1f}'.format(ib__vec[ii],num_in_burst__vec[jj]))  
    ax.set_ylabel(r'$s_{max}$ [$I_c$]')  
    ax.set_xlabel(r'$f_{burst}$ [1/$\tau_{di}$]')
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
    ax.legend(loc = 'best') 
    
    return

def plot_two_synapse_gate(synapse_1, synapse_2, dendrite_1, gate):
    
    fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , p['golden_ratio']*fig_size[1]) )
    plt.suptitle('two synapse gate: {}, loops present = {}\nib = {:5.3f}\nbeta_c = {:5.3f}, beta_di/2pi = {:5.3e}, tau_di = {:3.1e}ns'.format( gate, dendrite_1.loops_present, dendrite_1.bias_current, dendrite_1.beta_c, dendrite_1.circuit_betas[-1]/(2*np.pi), dendrite_1.tau_di ) )
    
    _time_vec = dendrite_1.output_data['tau_vec']/dendrite_1.time_params['t_tau_conversion']
     
    ax[0].plot(_time_vec, synapse_1.phi_spd, linestyle = 'solid', color = colors['yellow3'], label = '{}'.format(synapse_1.name))
    ax[0].plot(_time_vec, synapse_2.phi_spd, linestyle = 'dashed', color = colors['green3'], label = '{}'.format(synapse_2.name))
    ax[0].set_ylabel(r'$\phi_{spd}$ [$\Phi_0$]')          
    ax[0].legend() # loc = 'lower right'
    
    ax[1].plot(_time_vec, dendrite_1.output_data['phi_r'], color = colors['blue3'], label = '{}'.format(dendrite_1.name))  
    ax[1].set_ylabel(r'$\phi_r$ [$\Phi_0$]')          
    ax[1].legend() # loc = 'lower right'
    
    ax[2].plot(_time_vec, dendrite_1.output_data['s'], color = colors['blue3'], label = '{}'.format(dendrite_1.name)) 
    ax[2].set_ylabel(r'$s$ [$I_c$]')
    ax[2].legend() # loc = 'lower right'
    # ax[1].set_ylim([-2,2])
    
    ax[2].set_xlabel(r'Time [ns]')
    # ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    # ax.grid(which='minor', linewidth = 0.15)
            
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # plt.show() 
    
    return

def plot_two_synapse_gate__multi(s_max__mat, dt__vec, ib__vec, gate, dendrite_1):
    
    color_list = colors_gist(np.linspace(.1, 1,len(ib__vec)))
        
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , p['golden_ratio']*fig_size[1]) ) # 
    plt.suptitle('two synapse gate: {}, loops present = {}\nbeta_c = {:5.3f}, beta_di/2pi = {:5.3e}, tau_di = {:3.1e}ns'.format( gate, dendrite_1.loops_present, dendrite_1.beta_c, dendrite_1.circuit_betas[-1]/(2*np.pi), dendrite_1.tau_di ) )
    
    for ii in range(len(ib__vec)):    
        ax.plot(dt__vec[:], s_max__mat[ii,:], '-o', color = color_list[ii], label = 'ib = {:5.3f}'.format(ib__vec[ii]))
    ax.set_ylabel(r'$s$ [$I_c$]')          
    ax.legend() # loc = 'lower right'
    
    ax.set_xlabel(r'$\Delta t$ [ns]')
    # ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    # ax.grid(which='minor', linewidth = 0.15)
            
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # plt.show() 
    
    return

def plot_syn_two_dend_dend_burst_transfer_function_notch(ib__dend, ib__neu, num_in_burst__vec, burst_frequency__vec, transfer_function__array, tau_di__dend_1, beta_di__dend_1, tau_di__dend_2, beta_di__dend_2, tau_di__dend_3, beta_di__dend_3):
    
    color_list = ['blue1','blue2','blue3','blue4','blue5'] #,['green1','green2','green3','green4','green5'],['yellow1','yellow2','yellow3','yellow4','yellow5'],['red1','red2','red3','red4','red5']]
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    _str = 'syn dend burst transfer function'
    _str = '{}\ntau_di_dend_1 = {:4.2e}ns, beta_di_dend_1 = {:4.2e}'.format(_str, tau_di__dend_1, beta_di__dend_1)
    _str = '{}\ntau_di_dend_2 = {:4.2e}ns, beta_di_dend_2 = {:4.2e}'.format(_str, tau_di__dend_2, beta_di__dend_2)
    _str = '{}\ntau_di_dend_3 = {:4.2e}ns, beta_di_dend_3 = {:4.2e}'.format(_str, tau_di__dend_3, beta_di__dend_3)
    
    plt.suptitle(_str)
    for jj in range(len(burst_frequency__vec)):
        ax.plot(num_in_burst__vec[:], transfer_function__array[:,jj], '-o', color = colors[color_list[jj]], label = 'freq = {:3.1f}/tau_di_3'.format(burst_frequency__vec[jj]*tau_di__dend_3))  
    ax.set_ylabel(r'$s_{max}$ [$I_c$]')  
    ax.set_xlabel(r'Num in burst')
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
    ax.legend(loc = 'best') 
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle(_str)
    for jj in range(len(burst_frequency__vec)):
        ax.semilogx(num_in_burst__vec[:], transfer_function__array[:,jj], '-o', color = colors[color_list[jj]], label = 'freq = {:3.1f}/tau_di_3'.format(burst_frequency__vec[jj]*tau_di__dend_3))  
    ax.set_ylabel(r'$s_{max}$ [$I_c$]')  
    ax.set_xlabel(r'Num in burst')
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
    ax.legend(loc = 'best')
    
    return

# =============================================================================
# end synapse
# =============================================================================


# =============================================================================
# dendritic tree
# =============================================================================

def plot_dendritic_tree(dendrite):
    
    # determine depth of tree
    depth_of_tree = depth_of_dendritic_tree(dendrite)
    # print('depth_of_tree = {}'.format(depth_of_tree))
    
    fig, ax = plt.subplots(nrows = 1+2*depth_of_tree, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , (1+2*depth_of_tree)*fig_size[1]) )
    
    time_vec = dendrite.time_params['time_vec']
    tau_vec = dendrite.time_params['tau_vec']
    # plot synaptic flux
    def find_synapses_recursively_and_plot(_dendrite,ax):
        
        print('dendrite = {}'.format(_dendrite.name))
                
        for synapse_key in _dendrite.synaptic_inputs:
            
            if hasattr(_dendrite.synaptic_inputs[synapse_key],'phi_spd'):
                ax.plot(time_vec[:], _dendrite.synaptic_inputs[synapse_key].phi_spd[:], label = 'sy = {}'.format(_dendrite.synaptic_inputs[synapse_key].name)) 
                
            # find indices of input synapse spike times
            if hasattr(_dendrite.synaptic_inputs[synapse_key],'spike_times'):
                _ind_vec__syn_in = np.zeros([len(_dendrite.synaptic_inputs[synapse_key].spike_times)], dtype = int)
                for ii in range(len(_dendrite.synaptic_inputs[synapse_key].spike_times)):
                    _ind_vec__syn_in[ii] = ( np.abs( tau_vec[:] - _dendrite.synaptic_inputs[synapse_key].spike_times[ii] ) ).argmin()
                ax.plot(time_vec[_ind_vec__syn_in], _dendrite.synaptic_inputs[synapse_key].phi_spd[_ind_vec__syn_in], 'x')
            
        for dend in _dendrite.dendritic_inputs:
            find_synapses_recursively_and_plot(_dendrite.dendritic_inputs[dend],ax)
            
        return
    
    find_synapses_recursively_and_plot(dendrite,ax[0])
    
    # label axes
    ax[0].set_ylabel(r'$\phi_{spd}$ [$\Phi_0$]')  
    for ii in range(depth_of_tree):
        ax[2*ii+1].set_ylabel(r'$\phi_{dr}$ [$\Phi_0$]')
        ax[2*ii+2].set_ylabel(r'$s_{di}$ [$I_c$]')
    
    
    return

# =============================================================================
# end dendritic tree
# =============================================================================


# =============================================================================
# neuron
# =============================================================================

def plot_neuron(neuron_object):
    
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)
    
    # determine depth of tree
    depth_of_tree = depth_of_dendritic_tree(neuron_object)
    
    if hasattr(neuron_object,'plot_simple'):
        if neuron_object.plot_simple:
            _num = 1
        else:
            _num = 4
    else:
        neuron_object.plot_simple = False
        _num = 4
        
    neuron_object.bias_current = neuron_object.ib_n
    neuron_object.bias_current__refraction = neuron_object.ib_ref
    neuron_object.integration_loop_time_constant__refraction = neuron_object.tau_ref

    fig, ax = plt.subplots( nrows = _num+2*depth_of_tree, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , (3+2*depth_of_tree)*fig_size[1]) )
        
    _str = 'soen neuron: {}, loops present = {}'.format(neuron_object.name,neuron_object.loops_present)
    _str = '{}\nib = {:4.2f}, beta_ni/2pi = {:4.2e}, tau_ni = {:5.3e}ns'.format(_str,neuron_object.bias_current,neuron_object.circuit_betas[-1]/(2*np.pi),neuron_object.integration_loop_time_constant)
    _str = '{}\nib_refract = {:4.2f}, beta_refract/2pi = {:4.2e}, tau_refract = {:5.3e}ns'.format(_str,neuron_object.bias_current__refraction,neuron_object.circuit_betas__refraction[-1]/(2*np.pi),neuron_object.integration_loop_time_constant__refraction)
    plt.suptitle( _str )
    
    _time_vec = neuron_object.time_params['time_vec']
    _tau_vec = neuron_object.time_params['tau_vec']
    
    # find indices of neuron spike times
    _spike_ind_vec = np.zeros([len(neuron_object.spike_times)], dtype = int)
    for ii in range(len(neuron_object.spike_times)):
        _spike_ind_vec[ii] = ( np.abs( _tau_vec - neuron_object.spike_times[ii] ) ).argmin()
    
    # synapses
    def find_synapses_recursively_and_plot(dendrite_object,ax,already_plotted_synapses):
        
        if dendrite_object.name != '{}__dend_refraction'.format(neuron_object.name):
        
            for synapse_key in dendrite_object.synaptic_inputs:
                
                if dendrite_object.synaptic_inputs[synapse_key].name not in already_plotted_synapses:
                    
                    already_plotted_synapses.append(dendrite_object.synaptic_inputs[synapse_key].name)
                    
                    if len( dendrite_object.synaptic_inputs[synapse_key].spike_times_converted ) > 0: # only plot synapses that spiked
                        # print('synapses: name = {}'.format(dendrite_object.synaptic_inputs[synapse_key].name))
                        ax.plot(_time_vec[:], dendrite_object.synaptic_inputs[synapse_key].phi_spd[:], next(linecycler), label = '{}'.format(dendrite_object.synaptic_inputs[synapse_key].name)) 
                        
                        # find indices of input synapse spike times
                        _ind_vec__syn_in = np.zeros([len(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted)], dtype = int)
                        for ii in range(len(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted)):
                            _ind_vec__syn_in[ii] = ( np.abs( _tau_vec[:] - dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[ii] ) ).argmin()
                        # ax.plot(_time_vec[_ind_vec__syn_in], dendrite_object.synaptic_inputs[synapse_key].phi_spd[_ind_vec__syn_in], 'x')
                        ax.plot(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[:]/neuron_object.time_params['t_tau_conversion'], dendrite_object.synaptic_inputs[synapse_key].phi_spd[_ind_vec__syn_in[:]], 'x')
                
            for dendrite in dendrite_object.dendritic_inputs:
                find_synapses_recursively_and_plot(dendrite_object.dendritic_inputs[dendrite],ax,already_plotted_synapses)
            
        return
    
    already_plotted_synapses = []
    find_synapses_recursively_and_plot(neuron_object.dend_soma,ax[0],already_plotted_synapses)
    
    # dendrites
    def find_dendrites_recursively_and_plot(dendrite_object,ax,counter):
        
        counter += 1
        # print('dend_name = {}, counter = {}'.format(dendrite_object.name,counter))
        
        if dendrite_object.name != '{}__dend_refraction'.format(neuron_object.name):
        
            for dendrite_key in dendrite_object.dendritic_inputs:
                
                if np.max(dendrite_object.dendritic_inputs[dendrite_key].phi_r[:]) > 1e-6: # only plot dendrites that received appreciable flux
                    # print('dendrites: counter = {}, ax num = {}'.format(counter, len(ax)-1-3-2*counter-1))
                    _line = next(linecycler)
                    ax[ len(ax)-_num-2*counter-1 ].plot(_time_vec[:], dendrite_object.dendritic_inputs[dendrite_key].phi_r[:], linestyle = _line, label = '{}'.format(dendrite_object.dendritic_inputs[dendrite_key].name)) 
                    ax[ len(ax)-_num-2*counter ].plot(_time_vec[:], dendrite_object.dendritic_inputs[dendrite_key].s, linestyle = _line, label = '{}'.format(dendrite_object.dendritic_inputs[dendrite_key].name)) 
                        
            for dendrite in dendrite_object.dendritic_inputs:
                find_dendrites_recursively_and_plot(dendrite_object.dendritic_inputs[dendrite],ax,counter)
            
        return
    
    find_dendrites_recursively_and_plot(neuron_object.dend_soma,ax,0)
    
    ib__list, phi_r__array, i_di__array, r_fq__array, phi_th_plus__vec, phi_th_minus__vec, s_max_plus__vec, s_max_minus__vec, s_max_plus__array, s_max_minus__array = dend_load_arrays_thresholds_saturations('default_{}'.format(neuron_object.loops_present))
    _ind_ib = ( np.abs( ib__list[:] - neuron_object.dend_soma.ib ) ).argmin()

    ax[len(ax)-_num-1].plot(_time_vec[:], neuron_object.dend_soma.phi_r[:], color = colors['blue3'], label = 'total flux to NR')
    ax[len(ax)-_num-1].plot([_time_vec[0],_time_vec[-1]], [phi_th_plus__vec[_ind_ib],phi_th_plus__vec[_ind_ib]], ':', color = colors['red5'], label = 'flux threshold +')
    ax[len(ax)-_num-1].plot([_time_vec[0],_time_vec[-1]], [phi_th_minus__vec[_ind_ib],phi_th_minus__vec[_ind_ib]], ':', color = colors['red5'], label = 'flux threshold -')
    ax[len(ax)-_num-1].plot(_time_vec[_spike_ind_vec], neuron_object.dend_soma.phi_r[_spike_ind_vec], 'x', color = colors['blue5'])
    
    ax[len(ax)-_num].plot(_time_vec[:], neuron_object.dend_soma.s[:], color = colors['blue3'], label = 'current in NI')
    ax[len(ax)-_num].plot(_time_vec[_spike_ind_vec], neuron_object.dend_soma.s[_spike_ind_vec], 'x', color = colors['blue5'])
    ax[len(ax)-_num].plot([_time_vec[0],_time_vec[-1]] , [neuron_object.integrated_current_threshold,neuron_object.integrated_current_threshold], ':', color = colors['red5'], label = 'threshold')
    
    if not neuron_object.plot_simple:
        
        # refractory dendrite
        ax[len(ax)-3].plot(_time_vec[:], neuron_object.dend__ref.phi_r[:], color = colors['red3'], label = 'flux to refractory dendrite')
        ax[len(ax)-3].plot(_time_vec[_spike_ind_vec], neuron_object.dend__ref.phi_r[_spike_ind_vec], 'x', color = colors['red5'])
        
        ax[len(ax)-2].plot(_time_vec[:], neuron_object.dend__ref.s[:], color = colors['red3'], label = 'current in refractory dendrite')
        ax[len(ax)-2].plot(_time_vec[_spike_ind_vec], neuron_object.dend__ref.s[_spike_ind_vec], 'x', color = colors['red5'])
        
        # output synapses from neuron
        for synapse_key in neuron_object.synaptic_outputs:
            ax[-1].plot(_time_vec[:], neuron_object.synaptic_outputs[synapse_key].phi_spd[:], color = colors['green3'], label = '{}'.format(neuron_object.synaptic_outputs[synapse_key].name)) 
            ax[-1].plot(_time_vec[_spike_ind_vec], neuron_object.synaptic_outputs[synapse_key].phi_spd[_spike_ind_vec], 'x', color = colors['green5']) 
        
    # label axes
    ax[0].set_ylabel(r'$\phi_{spd}$ [$\Phi_0$]')
    for ii in range(depth_of_tree):
        ax[2*ii+1].set_ylabel(r'$\phi_{dr}$ [$\Phi_0$]')
        ax[2*ii+2].set_ylabel(r'$s_{di}$ [$I_c$]')
    ax[len(ax)-_num-1].set_ylabel(r'$\phi_{nr}$ [$\Phi_0$]')
    ax[len(ax)-_num].set_ylabel(r'$s_{ni}$ [$I_c$]')
    
    if not neuron_object.plot_simple:
        ax[len(ax)-3].set_ylabel(r'$\phi_{ref}$ [$\Phi_0$]')
        ax[len(ax)-2].set_ylabel(r'$s_{ref}$ [$I_c$]')
        ax[len(ax)-1].set_ylabel(r'$\phi_{spd}$ [$\Phi_0$]')
    
    ax[-1].set_xlabel(r'Time [ns]')
    
    loc = 'center left'
    for ii in range(len(ax)):
        if loc == 'center left':
            loc ='center right'
        elif loc == 'center right':
            loc = 'center left'
        ax[ii].legend(loc = loc) # loc = 'lower right'
    
    if 't_lims__plotting' in neuron_object.time_params:
        ax[-1].set_xlim( [ neuron_object.time_params['t_lims__plotting'][0] , neuron_object.time_params['t_lims__plotting'][1] ] )
            
    plt.subplots_adjust(wspace = 0.3, hspace = 0)
    # plt.show() 
    
    return

def plot_neuron_simple(neuron_object):
    neuron_object.bias_current = neuron_object.ib_n
    neuron_object.bias_current__refraction = neuron_object.ib_ref
    neuron_object.integration_loop_time_constant__refraction = neuron_object.tau_ref
    
    if fig_size[0] < 10:
        
        plt.rcParams['legend.fontsize'] = 6
        plt.rcParams['axes.labelsize'] = 8
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
    
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)
    
    # determine depth of tree
    depth_of_tree = depth_of_dendritic_tree(neuron_object)
    _num = 1

    fig, ax = plt.subplots( nrows = 5, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 3*fig_size[1]) )
        
    _str = 'soen neuron: {}, loops present = {}'.format(neuron_object.name,neuron_object.loops_present)
    _str = '{}\nib = {:4.2f}, beta_ni/2pi = {:4.2e}, tau_ni = {:5.3e}ns, J_rn = {:5.3f}'.format(_str,neuron_object.bias_current,neuron_object.circuit_betas[-1]/(2*np.pi),neuron_object.integration_loop_time_constant,neuron_object.refractory_dendrite_connection_strength)
    _str = '{}\nib_refract = {:4.2f}, beta_refract/2pi = {:4.2e}, tau_refract = {:5.3e}ns'.format(_str,neuron_object.bias_current__refraction,neuron_object.circuit_betas__refraction[-1]/(2*np.pi),neuron_object.integration_loop_time_constant__refraction)
    plt.suptitle( _str )
    
    _time_vec = neuron_object.time_params['time_vec']
    _tau_vec = neuron_object.time_params['tau_vec']
    
    # find indices of neuron spike times
    _spike_ind_vec = np.zeros([len(neuron_object.spike_times)], dtype = int)
    for ii in range(len(neuron_object.spike_times)):
        _spike_ind_vec[ii] = index_finder(_tau_vec,neuron_object.spike_times[ii])
    
    # synapses
    def find_synapses_recursively_and_plot(dendrite_object,ax,already_plotted_synapses):
        
        if dendrite_object.name != '{}__dend_refraction'.format(neuron_object.name):
        
            for synapse_key in dendrite_object.synaptic_inputs:
                
                if dendrite_object.synaptic_inputs[synapse_key].name not in already_plotted_synapses:
                    
                    already_plotted_synapses.append(dendrite_object.synaptic_inputs[synapse_key].name)
                    
                    if len( dendrite_object.synaptic_inputs[synapse_key].spike_times_converted ) > 0: # only plot synapses that spiked
                        # print('synapses: name = {}'.format(dendrite_object.synaptic_inputs[synapse_key].name))
                        ax.plot(_time_vec[:], dendrite_object.synaptic_inputs[synapse_key].phi_spd[:], next(linecycler), label = '{}'.format(dendrite_object.synaptic_inputs[synapse_key].name)) 
                        
                        # find indices of input synapse spike times
                        _ind_vec__syn_in = np.zeros([len(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted)], dtype = int)
                        for ii in range(len(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted)):
                            _ind_vec__syn_in[ii] = ( np.abs( _tau_vec[:] - dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[ii] ) ).argmin()
                        # ax.plot(_time_vec[_ind_vec__syn_in], dendrite_object.synaptic_inputs[synapse_key].phi_spd[_ind_vec__syn_in], 'x')
                        ax.plot(dendrite_object.synaptic_inputs[synapse_key].spike_times_converted[:]/neuron_object.time_params['t_tau_conversion'], dendrite_object.synaptic_inputs[synapse_key].phi_spd[_ind_vec__syn_in[:]], 'x')
                
            for dendrite in dendrite_object.dendritic_inputs:
                find_synapses_recursively_and_plot(dendrite_object.dendritic_inputs[dendrite],ax,already_plotted_synapses)
            
        return
    
    already_plotted_synapses = []
    find_synapses_recursively_and_plot(neuron_object.dend_soma,ax[0],already_plotted_synapses)
            
    # output synapses from neuron
    for synapse_key in neuron_object.synaptic_outputs:
        ax[0].plot(_time_vec[:], neuron_object.synaptic_outputs[synapse_key].phi_spd[:], color = colors['green3'], label = '{}'.format(neuron_object.synaptic_outputs[synapse_key].name)) 
        _spike_out_ind_vec = np.zeros([len(neuron_object.synaptic_outputs[synapse_key].spike_times_converted)], dtype = int)
        for ii in range(len(_spike_out_ind_vec)):
            _spike_out_ind_vec[ii] = index_finder(_tau_vec,neuron_object.synaptic_outputs[synapse_key].spike_times_converted[ii])
        ax[0].plot(_time_vec[_spike_out_ind_vec], neuron_object.synaptic_outputs[synapse_key].phi_spd[_spike_out_ind_vec], 'x', color = colors['green5'])
    
    # dendrites
    def find_dendrites_recursively_and_plot(dendrite_object,axs,counter):
        
        counter += 1
        # print('dend_name = {}, counter = {}'.format(dendrite_object.name,counter))
        
        if dendrite_object.name != '{}__dend_refraction'.format(neuron_object.name):
        
            for dendrite_key in dendrite_object.dendritic_inputs:
                
                if np.max(dendrite_object.dendritic_inputs[dendrite_key].phi_r[:]) > 1e-6: # only plot dendrites that received appreciable flux
                    # print('dendrites: counter = {}, ax num = {}'.format(counter, len(ax)-1-3-2*counter-1))
                    _line = next(linecycler)
                    axs[0].plot(_time_vec[:], dendrite_object.dendritic_inputs[dendrite_key].phi_r[:], linestyle = _line, label = '{}'.format(dendrite_object.dendritic_inputs[dendrite_key].name)) 
                    axs[1].plot(_time_vec[:], dendrite_object.dendritic_inputs[dendrite_key].s[:], linestyle = _line, label = '{}'.format(dendrite_object.dendritic_inputs[dendrite_key].name)) 
                    # ax[ len(ax)-2*counter ].plot(_time_vec[:], dendrite_object.dendritic_inputs[dendrite_key].s, linestyle = _line, label = '{}'.format(dendrite_object.dendritic_inputs[dendrite_key].name)) 
                        
            for dendrite in dendrite_object.dendritic_inputs:
                find_dendrites_recursively_and_plot(dendrite_object.dendritic_inputs[dendrite],ax,counter)
            
        return
    
    find_dendrites_recursively_and_plot(neuron_object.dend_soma,[ax[1],ax[2]],0)
    
    d_params = dend_load_arrays_thresholds_saturations(f'default_{neuron_object.dend_soma.loops_present}')
    ib__list = d_params["ib__list"]
    phi_r__array = d_params["phi_r__array"]
    i_di__array = d_params["i_di__array"]
    r_fq__array = d_params["r_fq__array"]
    phi_th_plus__vec = d_params["phi_th_plus__vec"]
    phi_th_minus__vec  = d_params["phi_th_minus__vec"]
    s_max_plus__vec  = d_params["s_max_plus__vec"]
    s_max_minus__vec = d_params["s_max_minus__vec"]
    s_max_plus__array = d_params["s_max_plus__array"]
    s_max_minus__array  = d_params["s_max_minus__array"]
    
    _ind_ib = ( np.abs( ib__list[:] - neuron_object.dend_soma.ib ) ).argmin()

    ax[3].plot(_time_vec[:], neuron_object.dend_soma.phi_r[:], color = colors['blue3'], label = 'total flux to NR')
    ax[3].plot([_time_vec[0],_time_vec[-1]], [phi_th_plus__vec[_ind_ib],phi_th_plus__vec[_ind_ib]], ':', color = colors['red5'], label = 'flux threshold +')
    ax[3].plot([_time_vec[0],_time_vec[-1]], [phi_th_minus__vec[_ind_ib],phi_th_minus__vec[_ind_ib]], ':', color = colors['red5'], label = 'flux threshold -')
    ax[3].plot(_time_vec[_spike_ind_vec], neuron_object.dend_soma.phi_r[_spike_ind_vec], 'x', color = colors['blue5'])
    
    ax[4].plot(_time_vec[:], neuron_object.dend_soma.s[:], color = colors['blue3'], label = 'current in NI')
    ax[4].plot(_time_vec[_spike_ind_vec], neuron_object.dend_soma.s[_spike_ind_vec], 'x', color = colors['blue5'])
    if (neuron_object.integrated_current_threshold-np.max(neuron_object.dend_soma.s[:]))/neuron_object.integrated_current_threshold < 0.5 or np.max(neuron_object.dend_soma.s[:]) > neuron_object.integrated_current_threshold:
        ax[len(ax)-_num].plot([_time_vec[0],_time_vec[-1]] , [neuron_object.integrated_current_threshold,neuron_object.integrated_current_threshold], ':', color = colors['red5'], label = 'threshold')
    
        
    # refractory dendrite
    # ax[1].plot(_time_vec[:], neuron_object.dend__ref.phi_r[:], color = colors['red3'], label = 'flux to refractory dendrite')
    # ax[1].plot(_time_vec[_spike_ind_vec], neuron_object.dend__ref.phi_r[_spike_ind_vec], 'x', color = colors['red5'])
    
    # ax[1].plot(_time_vec[:], neuron_object.dend__ref.s[:], color = colors['red3'], label = 'current in refractory dendrite')
    # ax[1].plot(_time_vec[_spike_ind_vec], neuron_object.dend__ref.s[_spike_ind_vec], 'x', color = colors['red5'])
        
    # label axes
    ax[0].set_ylabel(r'$\phi_{spd}$ [$\Phi_0$]')
    # for ii in range(depth_of_tree):
    #     ax[2*ii+1].set_ylabel(r'$\phi_{dr}$ [$\Phi_0$]')
    #     ax[2*ii+2].set_ylabel(r'$s_{di}$ [$I_c$]')
    ax[1].set_ylabel(r'$\phi_{dr}$ [$\Phi_0$]')
    ax[2].set_ylabel(r'$s_{di}$ [$I_c$]')
    ax[3].set_ylabel(r'$\phi_{nr}$ [$\Phi_0$]')
    ax[4].set_ylabel(r'$s_{ni}$ [$I_c$]')
    
    
    ax[-1].set_xlabel(r'Time [ns]')
    
    loc = 'center left'
    for ii in range(len(ax)):
        if loc == 'center left':
            loc = 'center right'
        elif loc == 'center right':
            loc = 'center left'
        ax[ii].legend(loc = loc) # loc = 'lower right'
    
    if 't_lims__plotting' in neuron_object.time_params:
        ax[-1].set_xlim( [ neuron_object.time_params['t_lims__plotting'][0] , neuron_object.time_params['t_lims__plotting'][1] ] )
            
    plt.subplots_adjust(wspace = 0.3, hspace = 0)
    # plt.show() 
    
    return

def plot_neuron_phase_portrait(neuron_object):
    
    _tau_vec = neuron_object.time_params['tau_vec']
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , 2*fig_size[1]) )
    plt.suptitle( 'soen neuron: {}; phase portrait'.format(neuron_object.name) )
    
    neuron_refractory_flux = neuron_object.dend__ref.s * neuron_object.dend_soma.dendritic_connection_strengths['{}__dend_{}'.format(neuron_object.name,'refraction')]
    neuron_applied_flux_without_refraction = neuron_object.dend_soma.phi_r - neuron_refractory_flux
    neuron_refractory_flux = -neuron_refractory_flux
    
    ax.plot(neuron_applied_flux_without_refraction, neuron_refractory_flux, color = colors['blue3'], label = 'non-refractory flux to NR')
    
    # put a green dot at input synapse spike times
    def find_synapses_recursively_and_plot(dendrite_object,ax):
        
        if dendrite_object.name != '{}__dend_refraction'.format(neuron_object.name):
        
            for synapse_key in dendrite_object.synaptic_inputs:
                
                # find indices of input synapse spike times
                _ind_vec__syn_in = np.zeros([len(dendrite_object.synaptic_inputs[synapse_key].spike_times)], dtype = int)
                for ii in range(len(dendrite_object.synaptic_inputs[synapse_key].spike_times)):
                    _ind_vec__syn_in[ii] = ( np.abs( _tau_vec[:] - dendrite_object.synaptic_inputs[synapse_key].spike_times[ii] ) ).argmin()
                ax.plot(neuron_applied_flux_without_refraction[_ind_vec__syn_in], neuron_refractory_flux[_ind_vec__syn_in], 'o', color = colors['green3'])
                
            for dendrite in dendrite_object.dendritic_inputs:
                find_synapses_recursively_and_plot(dendrite_object.dendritic_inputs[dendrite],ax)
                
            return
        
    ax.plot(neuron_applied_flux_without_refraction[0], neuron_refractory_flux[0], 'o', color = colors['green3'], label = 'starting point')

    find_synapses_recursively_and_plot(neuron_object.dend_soma,ax)     
    
    # put red crosses at neuron spike times
    _ind_vec = np.zeros([len(neuron_object.spike_times)], dtype = int)
    for ii in range(len(neuron_object.spike_times)):
        _ind_vec[ii] = ( np.abs( _tau_vec - neuron_object.spike_times[ii] ) ).argmin()
    ax.plot(neuron_applied_flux_without_refraction[_ind_vec], neuron_refractory_flux[_ind_vec], 'x', color = colors['red3'], label = 'spike_times')
    
    # set axes ranges
    phi_receiving__min = np.min(neuron_applied_flux_without_refraction)
    phi_receiving__max = np.max(neuron_applied_flux_without_refraction)
    phi_receiving__range = phi_receiving__max - phi_receiving__min
    phi_refraction__min = np.min(neuron_refractory_flux)
    phi_refraction__max = np.max(neuron_refractory_flux)
    phi_refraction__range = phi_refraction__max - phi_refraction__min
    ax.set_xlim([ phi_receiving__min - 0.05*phi_receiving__range , phi_receiving__max + 0.05*phi_receiving__range ])
    ax.set_ylim([ phi_refraction__min - 0.05*phi_refraction__range , phi_refraction__max + 0.05*phi_refraction__range ])
        
    # label axes
    ax.set_xlabel(r'$\phi_{nr}^+$ [$\Phi_0$]')
    ax.set_ylabel(r'$\phi_{ref}$ [$\Phi_0$]')
    
    # legend
    ax.legend()
                
    plt.subplots_adjust(wspace = 0.3, hspace = 0)
    
    return

def plot_neuron_rate_out_vs_constant_drive(ib_n__vec, beta_ni__vec, tau_ni__vec, applied_flux__array, rate_out__array, s_th__factor, neuron_object):
    
    color_list = [['blue1','blue2','blue3','blue4','blue5'],['green1','green2','green3','green4','green5'],['yellow1','yellow2','yellow3','yellow4','yellow5'],['red1','red2','red3','red4','red5']]
        
    for qq in range(len(tau_ni__vec)):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) ) # 
        plt.suptitle('neuron with constant drive: loops present = {}\ntau_ni = {:5.3e}ns, s_th__factor = {:3.1f}'.format( neuron_object.loops_present, tau_ni__vec[qq], s_th__factor ) )

        for ii in range(len(ib_n__vec)):
            for jj in range(len(beta_ni__vec)):            
                _inds = np.where( np.asarray(rate_out__array[qq][ii][jj][:]) > 0 )[0]
                # print(_inds)
                if len(_inds) > 0:
                    _inds = np.concatenate(([_inds[0]-1],_inds))
                    # print(_inds)
                    ax.plot(np.asarray(applied_flux__array[qq][ii][:])[_inds], np.asarray(rate_out__array[qq][ii][jj][:])[_inds], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:5.3f}, beta_ni/2pi = {:3.1e}'.format(ib_n__vec[ii],beta_ni__vec[jj]/2/np.pi))
            
        ax.set_ylim([0,20])
        ax.set_ylabel(r'rate out [MHz]')          
        ax.legend(loc = 'center left') # loc = 'lower right'
        ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
        ax.grid(which='minor', linewidth = 0.15)
        
        ax.set_xlabel(r'$\Phi_a$ [$\Phi_0$]')
                
        # plt.subplots_adjust(wspace=0.3, hspace=0)
        # plt.show() 
    
    return

def plot_neuron_rate_out_vs_rate_in(ib_n__vec, beta_ni__vec, rate_applied__vec, rate_out__array, s_th__factor, neuron_object):
    
    color_list = [['blue1','blue2','blue3','blue4','blue5'],['green1','green2','green3','green4','green5'],['yellow1','yellow2','yellow3','yellow4','yellow5'],['red1','red2','red3','red4','red5']]
        
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) ) # 
    plt.suptitle('neuron with constant drive: loops present = {}\ntau_ni = {:5.3e}ns, s_th__factor = {:3.1f}'.format( neuron_object.loops_present, neuron_object.dend_soma.tau_di, s_th__factor ) )
    
    for ii in range(len(ib_n__vec)):
        for jj in range(len(beta_ni__vec)):            
            # _inds = np.where( np.asarray(rate_out__array[ii,jj,:]) > 0 )[0]
            # # print(_inds)
            # if len(_inds) > 0:
            #     _inds = np.concatenate(([_inds[0]-1],_inds))
            #     # print(_inds)
            #     ax.plot(rate_applied__vec[_inds], rate_out__array[ii,jj,_inds], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:5.3f}, beta_ni = {:3.1e}'.format(ib_n__vec[ii],beta_ni__vec[jj]))
            # ax.plot(rate_applied__vec[:], rate_out__array[ii,jj,:], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:5.3f}, beta_ni = {:3.1e}'.format(ib_n__vec[ii],beta_ni__vec[jj]))
            ax.semilogx(rate_applied__vec[:], rate_out__array[ii,jj,:], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:5.3f}, beta_ni = {:3.1e}'.format(ib_n__vec[ii],beta_ni__vec[jj]))
        
    ax.set_ylim([0,20])
    ax.set_ylabel(r'rate out [MHz]')          
    ax.legend(loc = 'center left') # loc = 'lower right'
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
    
    ax.set_xlabel(r'rate in [MHz]')
    
    return

def plot_monosynaptic_point_neuron__rate_out_vs_rate_in(ib_n__vec, beta_ni__vec, rate_in__vec, rate_out__array, neuron_object):
    
    # color_list = [['blue1','blue2','blue3','blue4','blue5'],['green1','green2','green3','green4','green5'],['yellow1','yellow2','yellow3','yellow4','yellow5'],['red1','red2','red3','red4','red5']]
    color_list = [['blue1','blue3','blue5'],['green1','green3','green5'],['yellow1','yellow3','yellow5'],['red1','red3','red5']]
        
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) ) # 
    _str = 'neuron with constant rate in: loops present = {}'.format(neuron_object.loops_present)
    _str = '{}\ntau_ni = {:5.3f}ns, s_th = {:4.2f}'.format(_str, neuron_object.dend_soma.tau_di, neuron_object.integrated_current_threshold)
    _str = '{}\nib_ref = {:5.3f}, beta_ref = {:5.3e}, tau_ref = {:5.3f}ns'.format(_str, neuron_object.dend__ref.ib, neuron_object.dend__ref.beta, neuron_object.dend__ref.tau_di)
    plt.suptitle('{}'.format( _str ) )
    
    for ii in range(len(ib_n__vec)):
        for jj in range(len(beta_ni__vec)):            

            # ax.semilogx(1e-6*rate_in__vec[:], 1e-6*rate_out__array[ii,:], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:5.3f}, beta_ni = {:3.1e}'.format(ib_n__vec[ii],beta_ni__vec[jj]))
            ax.loglog(1e-6*rate_in__vec[:], 1e-6*rate_out__array[ii,jj,:], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:5.3f}, beta_ni = {:3.1e}'.format(ib_n__vec[ii],beta_ni__vec[jj]))
        
    # ax.set_ylim([0,20])
    ax.set_xlabel(r'rate in [MHz]')
    ax.set_ylabel(r'rate out [MHz]')          
    ax.legend(loc = 'center left') # loc = 'lower right'
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
    
    _str = 'tau_ni_{:5.3f}ns_s_th_{:4.2f}'.format(neuron_object.dend_soma.tau_di, neuron_object.integrated_current_threshold)
    _str = '{}_ib_ref_{:5.3f}_beta_ref_{:5.3e}_tau_ref_{:5.3f}ns'.format(_str, neuron_object.dend__ref.ib, neuron_object.dend__ref.beta, neuron_object.dend__ref.tau_di)
    plt.savefig('one_syn_one_neu__{}.pdf'.format(_str))
    
    return

def plot_monosynaptic_monodendritic_neuron__rate_out_vs_rate_in(rate_in__vec, rate_out__array, ib_dend__vec, tau_di__vec, beta_di__vec, neuron_object, dendrite_object):
    
    color_list = [['blue1','blue2','blue3','blue4','blue5'],['green1','green2','green3','green4','green5'],['yellow1','yellow2','yellow3','yellow4','yellow5'],['red1','red2','red3','red4','red5']]
    # color_list = [['blue1','blue3','blue5'],['green1','green3','green5'],['yellow1','yellow3','yellow5'],['red1','red3','red5']]

    
    for kk in range(len(beta_di__vec)):
                
        fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) ) # 
        _str = 'neuron with constant rate in: loops present = {}, beta_di = {:5.3e}'.format(neuron_object.loops_present, beta_di__vec[kk])
        _str = '{}\nib_n = {:5.3f}, beta_ni = {:5.3e}, tau_ni = {:5.3f}ns, s_th = {:4.2f}'.format(_str, neuron_object.dend_soma.ib, neuron_object.dend_soma.beta, neuron_object.dend_soma.tau_di, neuron_object.integrated_current_threshold)
        _str = '{}\nib_ref = {:5.3f}, beta_ref = {:5.3e}, tau_ref = {:5.3f}ns'.format(_str, neuron_object.dend__ref.ib, neuron_object.dend__ref.beta, neuron_object.dend__ref.tau_di)
        plt.suptitle('{}'.format( _str ) )
    
        for ii in range(len(ib_dend__vec)):
            for jj in range(len(tau_di__vec)):            
    
                # ax.semilogx(1e-6*rate_in__vec[:], 1e-6*rate_out__array[ii,:], '-o', color = colors[color_list[ii][jj]], label = 'ib = {:5.3f}, beta_ni = {:3.1e}'.format(ib_n__vec[ii],beta_ni__vec[jj]))
                ax.loglog(1e-6*rate_in__vec[:], 1e-6*rate_out__array[ii,kk,jj,:], '-o', color = colors[color_list[ii][jj]], label = 'ib_dend = {:5.3f}, tau_di = {:3.1e}'.format(ib_dend__vec[ii],tau_di__vec[jj]))
        
        # ax.set_ylim([0,20])
        ax.set_xlabel(r'rate in [MHz]')
        ax.set_ylabel(r'rate out [MHz]')          
        ax.legend(loc = 'center left') # loc = 'lower right'
        ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
        ax.grid(which='minor', linewidth = 0.15)
        
        _str = 'tau_ni_{:5.3f}ns_s_th_{:4.2f}'.format(neuron_object.dend_soma.tau_di, neuron_object.integrated_current_threshold)
        _str = '{}_ib_ref_{:5.3f}_beta_ref_{:5.3e}_tau_ref_{:5.3f}ns'.format(_str, neuron_object.dend__ref.ib, neuron_object.dend__ref.beta, neuron_object.dend__ref.tau_di)
        _str = '{}_beta_di_{:5.3e}'.format(_str, beta_di__vec[kk])
        plt.savefig('one_syn_one_dend_one_neu__{}.pdf'.format(_str))
    
    return


def plot_monosynaptic_monodendritic_neuron__single_synapse_event(spike_times__array, ib_neuron__vec, tau_di__vec, beta_ni__vec, neuron_object, dendrite_object, refractory_connection_strength):
    
    # color_list = [['blue1','blue2','blue3','blue4','blue5'],['green1','green2','green3','green4','green5'],['yellow1','yellow2','yellow3','yellow4','yellow5'],['red1','red2','red3','red4','red5']]
    # color_list = [['blue1','blue3','blue5'],['green1','green3','green5'],['yellow1','yellow3','yellow5'],['red1','red3','red5']]

    r_out__array = []
    num_burst__array = []
    ib_neuron__array_1 = []
    ib_neuron__array_2 = []            
    for ii in range(len(beta_ni__vec)):
        r_out__array.append([])
        num_burst__array.append([])
        ib_neuron__array_1.append([])
        ib_neuron__array_2.append([])
        
        for jj in range(len(tau_di__vec)):
            r_out__array[ii].append([])
            num_burst__array[ii].append([])
            ib_neuron__array_1[ii].append([])
            ib_neuron__array_2[ii].append([])
            
            for kk in range(len(ib_neuron__vec)):
                num_burst__array[ii][jj].append(len(spike_times__array[ii][jj][kk]))
                if len(spike_times__array[ii][jj][kk]) > 1:
                    r_out__array[ii][jj].append(np.max(1/np.diff(spike_times__array[ii][jj][kk])))
                else:
                    r_out__array[ii][jj].append(0)
                    
                 
    # for each beta_ni (separate plot for each tau_di)
    for jj in range(len(tau_di__vec)):
                
        fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) ) # 
        _str = 'neuron, one syn, one dend, one synapse event: loops present = {}'.format(neuron_object.loops_present)
        _str = '{}\nbeta_di = {:5.3e}, tau_di = {:5.3e}ns'.format(_str, dendrite_object.beta, tau_di__vec[jj])
        _str = '{}\nib_n = {:5.3f}, tau_ni = {:5.3f}ns, s_th = {:4.2f}'.format(_str, neuron_object.dend_soma.ib, neuron_object.dend_soma.tau_di, neuron_object.integrated_current_threshold)
        _str = '{}\nib_ref = {:5.3f}, beta_ref = {:5.3e}, tau_ref = {:5.3f}ns'.format(_str, neuron_object.dend__ref.ib, neuron_object.dend__ref.beta, neuron_object.dend__ref.tau_di)
        plt.suptitle('{}'.format( _str ) )
    
        color_list = colors_gist(np.linspace(.1, 1,len(beta_ni__vec)))
        for ii in range(len(beta_ni__vec)):            

            ax[0].plot(ib_neuron__vec[:], 1e-6*np.asarray(r_out__array[ii][jj][:]), '-o', color = color_list[ii], label = 'beta_ni = {:3.1e}'.format(beta_ni__vec[ii]))
            ax[1].plot(ib_neuron__vec[:], np.asarray(num_burst__array[ii][jj][:]), '-o', color = color_list[ii], label = 'beta_ni = {:3.1e}'.format(beta_ni__vec[ii]))
        
        # ax.set_ylim([0,20])
        ax[-1].set_xlabel('$i_b$ [$I_c$]')
        ax[0].set_ylabel('rate out [MHz]')          
        ax[1].set_ylabel('Num in burst')
        for qq in range(len(ax)):
            ax[qq].legend(loc = 'center left') # loc = 'lower right'
            ax[qq].grid(which = 'both', axis = 'both', color = colors['grey4'])
            ax[qq].grid(which='minor', linewidth = 0.15)
        plt.subplots_adjust(wspace=0.3, hspace=0)
        
        # _str = 'tau_ni_{:5.3f}ns_s_th_{:4.2f}'.format(neuron_object.dend_soma.tau_di, neuron_object.integrated_current_threshold)
        # _str = '{}_ib_ref_{:5.3f}_beta_ref_{:5.3e}_tau_ref_{:5.3f}ns'.format(_str, neuron_object.dend__ref.ib, neuron_object.dend__ref.beta, neuron_object.dend__ref.tau_di)
        # _str = '{}_beta_di_{:5.3e}'.format(_str, beta_di__vec[kk])
        # plt.savefig('one_syn_one_dend_one_neu__{}.pdf'.format(_str))
                  
    # for each tau_di (separate plot for each beta_ni)
    for ii in range(len(beta_ni__vec)):
                
        fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) ) # 
        _str = 'neuron, one syn, one dend, one synapse event: loops present = {}'.format(neuron_object.loops_present)
        _str = '{}\nbeta_di = {:5.3e}, beta_ni = {:5.3e}'.format(_str, dendrite_object.beta, beta_ni__vec[ii])
        _str = '{}\nib_n = {:5.3f}, tau_ni = {:5.3f}ns, s_th = {:4.2f}'.format(_str, neuron_object.dend_soma.ib, neuron_object.dend_soma.tau_di, neuron_object.integrated_current_threshold)
        _str = '{}\nib_ref = {:5.3f}, beta_ref = {:5.3e}, tau_ref = {:5.3f}ns'.format(_str, neuron_object.dend__ref.ib, neuron_object.dend__ref.beta, neuron_object.dend__ref.tau_di)
        plt.suptitle('{}'.format( _str ) )
    
        color_list = colors_gist(np.linspace(.1, 1,len(tau_di__vec)))
        for jj in range(len(tau_di__vec)):            
            
            ax[0].plot(ib_neuron__vec[:], 1e-6*np.asarray(r_out__array[ii][jj][:]), '-o', color = color_list[jj], label = 'tau_di = {:3.1e}'.format(tau_di__vec[jj]))
            ax[1].plot(ib_neuron__vec[:], np.asarray(num_burst__array[ii][jj][:]), '-o', color = color_list[jj], label = 'tau_di = {:3.1e}'.format(tau_di__vec[jj]))
        
        # ax.set_ylim([0,20])
        ax[-1].set_xlabel('$i_b$ [$I_c$]')
        ax[0].set_ylabel('rate out [MHz]')          
        ax[1].set_ylabel('Num in burst')
        for qq in range(len(ax)):
            ax[qq].legend(loc = 'center left') # loc = 'lower right'
            ax[qq].grid(which = 'both', axis = 'both', color = colors['grey4'])
            ax[qq].grid(which='minor', linewidth = 0.15)
        plt.subplots_adjust(wspace=0.3, hspace=0)
        
    return r_out__array


def plot_n_synapse_one_neuron(num_sy, Jij, s_max__mat, ib__vec, num_sy_fire__vec, neuron_1):
    
    dendrite_1 = neuron_1.dend_soma
    color_list = colors_gist(np.linspace(.1, 1,len(ib__vec)))
        
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , p['golden_ratio']*fig_size[1]) ) # 
    plt.suptitle('N-synapse dendrite: n_sy = {:d}, n_sy x Jij = {:4.2f}, loops present = {}\nbeta_di/2pi = {:5.3e}, tau_di = {:3.1e}ns'.format( int(num_sy), num_sy*Jij, dendrite_1.loops_present, dendrite_1.circuit_betas[-1]/(2*np.pi), dendrite_1.tau_di ) )
    
    for ii in range(len(ib__vec)):    
        ax.plot(num_sy_fire__vec[:]/num_sy, s_max__mat[ii,:], '-o', color = color_list[ii], label = 'ib = {:5.3f}'.format(ib__vec[ii]))
    [_min,_max] = ax.get_ylim()
    ax.plot(np.asarray([12,12])/num_sy, np.asarray([_min,_max]), ':', color = colors['grey10'])
    ax.plot(np.asarray([20,20])/num_sy, np.asarray([_min,_max]), ':', color = colors['grey10'])
    ax.set_ylim([_min,_max])
    ax.set_ylabel(r'$s$ [$I_c$]')          
    ax.legend() # loc = 'lower right'
    
    ax.set_xlabel(r'Active fraction')
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
            
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # plt.show() 
    
    return

# =============================================================================
# end neuron
# =============================================================================


# =============================================================================
# network
# =============================================================================

def plot_network(network_object):

    for neuron_key in network_object.neurons:    
        plot_neuron(network_object.neurons[neuron_key])
        
    return

# =============================================================================
# end network
# =============================================================================


# =============================================================================
# rate arrays
# =============================================================================
    
def dend_poly_plot_rate_array_alt_coloring(phi_r__array,s_di__array,r_fq__array,params, title = 'title not specified', axes_lims = ['auto','auto','auto'], axes_viewpoint = [30,-45]):

    # title can be 'from_params' or an actual title string
    # axes_lims is three-element list with each element either 'auto' or [min,max]; they're in this order: [[phi_r],[i_di],[r_fq]]
    phi_r_vec = np.asarray(phi_r__array)        
    if axes_lims[0] == 'auto':
        _ind_list = np.arange(0,len(phi_r__array),1)
    else:
        _ind_list = []        
        phi_r_min = axes_lims[0][0]
        phi_r_max = axes_lims[0][-1]
        for ii in range(len(phi_r_vec)):
            phi_r = phi_r_vec[ii]
            if phi_r >= phi_r_min and phi_r <= phi_r_max:
                _ind_list.append(ii)
        
    num_drives = len(_ind_list)
    
    cmap = colors_gist(np.linspace(0.05,1,num_drives)) # mp.cm.get_cmap('gist_earth')
    fig = plt.figure( figsize = ( fig_size[0] , fig_size[1] ) )
    if title == 'from_params':
        if params['loops_present'] == 'ri':
            fig.suptitle('{}, ib = {:07.5f}\nbeta_c = {:6.4f}, beta_1 = {:6.4f}, beta_2 = {:6.4f}'.format(params['loops_present'],params['ib'],params['beta_c'],params['beta_1'],params['beta_2']))
        if params['loops_present'] == 'pri':
            fig.suptitle('{}, ib = {:07.5f}, phi_p = {:6.4f}\nbeta_c = {:6.4f}, beta_1 = {:6.4f}, beta_2 = {:6.4f}, beta_L3 = {:6.4f}'.format(params['loops_present'],params['ib'],params['phi_p'],params['beta_c'],params['beta_1'],params['beta_2'],params['beta_L3']))
    else:
        fig.suptitle('{}'.format(title))
        
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(which = 'major', linewidth = 0.25)
    
    s_di_min = 1000
    s_di_max = -1000
    phi_r_min = 1e9
    phi_r_max = -1e9
    rate_min = 1000
    rate_max = -1000
    
    _s_di__start_ind = 3
    
    for ii in range(len(_ind_list)): 
        _ind = _ind_list[ii]
                
        x = np.insert(s_di__array[_ind][_s_di__start_ind:],0,0)
        z = phi_r__array[_ind]
        y = np.insert(r_fq__array[_ind][_s_di__start_ind:],0,0)
        
        verts = [(x[jj],y[jj]) for jj in range(len(x))]
        
        _clr = cmap[ii] # cmap(m*ii+c0)
                
        ax.add_collection3d(PolyCollection([verts], color = _clr, alpha = 0.1), zs = z, zdir = 'y')
        ax.plot(x, y, z, linewidth = 0.75, zdir = 'y', color = _clr, alpha = 1) 
        
        if np.min(s_di__array[_ind][:]) < s_di_min:
            s_di_min = np.min(s_di__array[_ind][:])
        if np.max(s_di__array[_ind][:]) > s_di_max:
            s_di_max = np.max(s_di__array[_ind][:])
                        
        if np.min(phi_r_vec[_ind]) < phi_r_min:
            phi_r_min = np.min(phi_r_vec[_ind])
        if np.max(phi_r_vec[_ind]) > phi_r_max:
            phi_r_max = np.max(phi_r_vec[_ind])
            
        if np.min(r_fq__array[_ind][:]) < rate_min:
            rate_min = np.min(r_fq__array[_ind][:])
        if np.max(r_fq__array[_ind][:]) > rate_max:
            rate_max = np.max(r_fq__array[_ind][:])
               
    ax.set_xlabel(r'$s_{di}$ [$I_c$]', labelpad = 0) # ,fontsize = 12
    D_s = s_di_max-s_di_min
    if axes_lims[1] == 'auto':
        ax.set_xlim3d(s_di_min-D_s/10,s_di_max+D_s/10)
    else:
        ax.set_xlim3d(axes_lims[1][0],axes_lims[1][1])
    
    ax.set_ylabel('$\phi_r$ [$\Phi_0$]', labelpad = 0) # , fontsize = 12
    if axes_lims[0] == 'auto':
        ax.set_ylim3d(phi_r_vec[0],phi_r_vec[-1])
    else:
        ax.set_ylim3d(axes_lims[0][0],axes_lims[0][1])
        
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r'$r_{fq}$ [fluxons per $\tau_0$]', rotation = 90, labelpad = 6) # ,fontsize = 12
    D_rate = rate_max-rate_min
    if axes_lims[2] == 'auto':
        ax.set_zlim3d(rate_min,rate_max+D_rate/10)
    else:
        ax.set_zlim3d(axes_lims[2][0],axes_lims[2][1])    
        
    ax.view_init(axes_viewpoint[0],axes_viewpoint[1])
    
    
    
    
    from matplotlib.collections import LineCollection
    
    x = np.linspace(0, 3 * np.pi, 500)
    y = np.sin(x)
    dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
    
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)
    fig.colorbar(line, ax=axs[0])

    return
    
def dend_poly_plot_rate_array(phi_r__array,s_di__array,r_fq__array,params, title = 'title not specified', axes_lims = ['auto','auto','auto'], axes_viewpoint = [30,-45]):

    # title can be 'from_params' or an actual title string
    # axes_lims is three-element list with each element either 'auto' or [min,max]; they're in this order: [[phi_r],[i_di],[r_fq]]
    phi_r_vec = np.asarray(phi_r__array)        
    if axes_lims[0] == 'auto':
        _ind_list = np.arange(0,len(phi_r__array),1)
    else:
        _ind_list = []        
        phi_r_min = axes_lims[0][0]
        phi_r_max = axes_lims[0][-1]
        for ii in range(len(phi_r_vec)):
            phi_r = phi_r_vec[ii]
            if phi_r >= phi_r_min and phi_r <= phi_r_max:
                _ind_list.append(ii)
        
    num_drives = len(_ind_list)
    
    cmap = colors_gist(np.linspace(0.05,1,num_drives)) # mp.cm.get_cmap('gist_earth')
    fig = plt.figure( figsize = ( fig_size[0] , fig_size[1] ) )
    if title == 'from_params':
        if params['loops_present'] == 'ri':
            fig.suptitle('{}, ib = {:07.5f}\nbeta_c = {:6.4f}, beta_1 = {:6.4f}, beta_2 = {:6.4f}'.format(params['loops_present'],params['ib'],params['beta_c'],params['beta_1'],params['beta_2']))
        if params['loops_present'] == 'pri':
            fig.suptitle('{}, ib = {:07.5f}, phi_p = {:6.4f}\nbeta_c = {:6.4f}, beta_1 = {:6.4f}, beta_2 = {:6.4f}, beta_L3 = {:6.4f}'.format(params['loops_present'],params['ib'],params['phi_p'],params['beta_c'],params['beta_1'],params['beta_2'],params['beta_L3']))
    else:
        fig.suptitle('{}'.format(title))
        
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(which = 'major', linewidth = 0.25)
    
    s_di_min = 1000
    s_di_max = -1000
    phi_r_min = 1e9
    phi_r_max = -1e9
    rate_min = 1000
    rate_max = -1000
    
    _s_di__start_ind = 3
    
    for ii in range(len(_ind_list)): 
        _ind = _ind_list[ii]
                
        x = np.insert(s_di__array[_ind][_s_di__start_ind:],0,0)
        z = phi_r__array[_ind]
        y = np.insert(r_fq__array[_ind][_s_di__start_ind:],0,0)
        
        verts = [(x[jj],y[jj]) for jj in range(len(x))]
        
        _clr = cmap[ii] # cmap(m*ii+c0)
                
        ax.add_collection3d(PolyCollection([verts], color = _clr, alpha = 0.1), zs = z, zdir = 'y')
        ax.plot(x, y, z, linewidth = 0.75, zdir = 'y', color = _clr, alpha = 1) 
        
        if np.min(s_di__array[_ind][:]) < s_di_min:
            s_di_min = np.min(s_di__array[_ind][:])
        if np.max(s_di__array[_ind][:]) > s_di_max:
            s_di_max = np.max(s_di__array[_ind][:])
                        
        if np.min(phi_r_vec[_ind]) < phi_r_min:
            phi_r_min = np.min(phi_r_vec[_ind])
        if np.max(phi_r_vec[_ind]) > phi_r_max:
            phi_r_max = np.max(phi_r_vec[_ind])
            
        if np.min(r_fq__array[_ind][:]) < rate_min:
            rate_min = np.min(r_fq__array[_ind][:])
        if np.max(r_fq__array[_ind][:]) > rate_max:
            rate_max = np.max(r_fq__array[_ind][:])
               
    ax.set_xlabel(r'$s_{di}$ [$I_c$]', labelpad = 0) # ,fontsize = 12
    D_s = s_di_max-s_di_min
    if axes_lims[1] == 'auto':
        ax.set_xlim3d(s_di_min-D_s/10,s_di_max+D_s/10)
    else:
        ax.set_xlim3d(axes_lims[1][0],axes_lims[1][1])
    
    ax.set_ylabel('$\phi_r$ [$\Phi_0$]', labelpad = 0) # , fontsize = 12
    if axes_lims[0] == 'auto':
        ax.set_ylim3d(phi_r_vec[0],phi_r_vec[-1])
    else:
        ax.set_ylim3d(axes_lims[0][0],axes_lims[0][1])
        
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r'$r_{fq}$ [fluxons per $\tau_0$]', rotation = 90, labelpad = 6) # ,fontsize = 12
    D_rate = rate_max-rate_min
    if axes_lims[2] == 'auto':
        ax.set_zlim3d(rate_min,rate_max+D_rate/10)
    else:
        ax.set_zlim3d(axes_lims[2][0],axes_lims[2][1])    
        
    ax.view_init(axes_viewpoint[0],axes_viewpoint[1])
    
    return

def dend_poly_plot_rate_array_wire_mesh(phi_r__array,s_di__array,r_fq__array,params, title = 'title not specified', axes_lims = ['auto','auto','auto'], axes_viewpoint = [30,-45]):    
    
    # title can be 'from_params' or an actual title string
    # axes_lims is three-element list with each element either 'auto' or [min,max]; they're in this order: [[phi_r],[i_di],[r_fq]]
    
    # from mpl_toolkits.mplot3d import Axes3D
    
    _s_di__start_ind = 0
    _s_di__end_ind = -1

    
    x = []
    y = []
    z = []
        
    if axes_lims[0] == 'auto':    
        s_di_min = 1000
        s_di_max = -1000
        phi_r_min = 1e9
        phi_r_max = -1e9
        rate_min = 1000
        rate_max = -1000
        
        for ii in range(len(phi_r__array)):
            phi_r = phi_r__array[ii] 
            
            s_di__vec = np.asarray( s_di__array[ii] )[_s_di__start_ind:_s_di__end_ind]
            r_fq__vec = np.asarray( r_fq__array[ii] )[_s_di__start_ind:_s_di__end_ind]
            
            for jj in range(len(s_di__vec)):
                x.append(s_di__vec[jj])
                y.append(phi_r)
                z.append(r_fq__vec[jj])
                
                if np.min(x) < s_di_min:
                    s_di_min = np.min(x)
                if np.max(x) > s_di_max:
                    s_di_max = np.max(x)
                                
                if np.min(y) < phi_r_min:
                    phi_r_min = np.min(y)
                if np.max(y) > phi_r_max:
                    phi_r_max = np.max(y)
                    
                if np.min(z) < rate_min:
                    rate_min = np.min(z)
                if np.max(z) > rate_max:
                    rate_max = np.max(z)
                
    elif axes_lims[0] != 'auto':
        phi_r_min = axes_lims[0][0]
        phi_r_max = axes_lims[0][-1]
        s_di_min = 1000
        s_di_max = -1000
        rate_min = 1000
        rate_max = -1000
            
        for ii in range(len(phi_r__array)):
            phi_r = phi_r__array[ii] 
            
            if phi_r >= phi_r_min and phi_r <= phi_r_max:
                s_di__vec = np.asarray( s_di__array[ii] )[_s_di__start_ind:_s_di__end_ind]
                r_fq__vec = np.asarray( r_fq__array[ii] )[_s_di__start_ind:_s_di__end_ind]
                
                for jj in range(len(s_di__vec)):
                    x.append(s_di__vec[jj])
                    y.append(phi_r)
                    z.append(r_fq__vec[jj])
                    
                    if np.min(x) < s_di_min:
                        s_di_min = np.min(x)
                    if np.max(x) > s_di_max:
                        s_di_max = np.max(x)
                                    
                    if np.min(y) < phi_r_min:
                        phi_r_min = np.min(y)
                    if np.max(y) > phi_r_max:
                        phi_r_max = np.max(y)
                        
                    if np.min(z) < rate_min:
                        rate_min = np.min(z)
                    if np.max(z) > rate_max:
                        rate_max = np.max(z)
            
    fig = plt.figure( figsize = ( fig_size[0] , fig_size[1] ) )
    if title == 'from_params':
        if params['loops_present'] == 'ri' or params['loops_present'] == 'rtti':
            fig.suptitle('{}, ib = {:07.5f}\nbeta_c = {:6.4f}, beta_1 = {:6.4f}, beta_2 = {:6.4f}'.format(params['loops_present'],params['ib'],params['beta_c'],params['beta_1'],params['beta_2']))
        if params['loops_present'] == 'pri':
            fig.suptitle('{}, ib = {:07.5f}, phi_p = {:6.4f}\nbeta_c = {:6.4f}, beta_1 = {:6.4f}, beta_2 = {:6.4f}, beta_L3 = {:6.4f}'.format(params['loops_present'],params['ib'],params['phi_p'],params['beta_c'],params['beta_1'],params['beta_2'],params['beta_L3']))
    else:
        fig.suptitle('{}'.format(title))
    ax = fig.gca(projection = '3d')
    ax.plot_trisurf(x, y, z, cmap = mp.cm.get_cmap('gist_earth'), norm = mp.colors.Normalize(vmin = 0, vmax = 1.1*np.max(z)), linewidth = 0.2, antialiased = True) # , alpha = 0.1    
    
    ax.set_xlabel(r'$s_{di}$ [$I_c$]', labelpad = 0) # ,fontsize = 12
    D_s = s_di_max-s_di_min
    if axes_lims[1] == 'auto':
        ax.set_xlim3d(s_di_min-D_s/10,s_di_max+D_s/10)
    else:
        ax.set_xlim3d(axes_lims[1][0],axes_lims[1][1])
    
    ax.set_ylabel('$\phi_r$ [$\Phi_0$]', labelpad = 0) # , fontsize = 12
    if axes_lims[0] == 'auto':
        ax.set_ylim3d(y[0],y[-1])
    else:
        ax.set_ylim3d(axes_lims[0][0],axes_lims[0][1])
        
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r'$r_{fq}$ [fluxons per $\tau_0$]', rotation = 90, labelpad = 6) # ,fontsize = 12
    D_rate = rate_max-rate_min
    if axes_lims[2] == 'auto':
        ax.set_zlim3d(rate_min,rate_max+D_rate/10)
    else:
        ax.set_zlim3d(axes_lims[2][0],axes_lims[2][1])    
        
    ax.view_init(axes_viewpoint[0],axes_viewpoint[1])

    return
    
# =============================================================================
# end rate arrays
# =============================================================================


# =============================================================================
# rate array thresholds and saturation points
# =============================================================================

def plot_thresholds_and_saturation_points(ib__list,phi_r__array,phi_th_plus__vec,phi_th_minus__vec,s_max_plus__vec,s_max_minus__vec,s_max_plus__array,s_max_minus__array,title):
    
    # phi_th
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle('{}'.format(title) )
         
    ax.plot(ib__list, phi_th_plus__vec, color = colors['blue3'], label = 'phi th +')
    ax.plot(ib__list, phi_th_minus__vec, color = colors['green3'], label = 'phi th -')
    ax.set_ylabel(r'$\phi_{th}^{+/-}$ [$\Phi_0$]')          
    ax.legend() # loc = 'lower right'
    
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
            
    ax.set_xlabel(r'ib [$I_c$]')
    plt.subplots_adjust(wspace=0.3, hspace=0)
        
    # s_max
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle('{}'.format(title) )
    
    ax.plot(ib__list, s_max_plus__vec, color = colors['blue3'], label = 's max +')
    ax.plot(ib__list, s_max_minus__vec, color = colors['green3'], label = 's max -')
    ax.set_ylabel(r'$s_{max}^{+/-}$ [$\Phi_0$]')          
    ax.legend() # loc = 'lower right'
    
    ax.grid(which = 'both', axis = 'both', color = colors['grey4'])
    ax.grid(which='minor', linewidth = 0.15)
            
    ax.set_xlabel(r'ib [$I_c$]')
    plt.subplots_adjust(wspace=0.3, hspace=0)    
    
    # s_max array
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle('{}'.format(title) )
    
    color_list = ['blue1','blue2','blue3','blue4','blue5','green1','green2','green3','green4','green5','yellow1','yellow2','yellow3','yellow4','yellow5','red1','red2','red3','red4','red5']
    for ii in range(len(ib__list)): 
        _phi_r__vec = np.asarray( phi_r__array[ii] )
        ax.plot(_phi_r__vec[ np.where( _phi_r__vec[:] > 0 ) ], s_max_plus__array[ii], color = colors[color_list[ii]], label = 'ib = {:5.3f}'.format(ib__list[ii]))
        ax.plot(_phi_r__vec[ np.where( _phi_r__vec[:] < 0 ) ], s_max_minus__array[ii], color = colors[color_list[ii]], label = 'ib = {:5.3f}'.format(ib__list[ii]))        
    
    ax.set_ylim([0,ax.get_ylim()[1]])
    ax.set_xlabel(r'$\phi_r$ [$\Phi_0$]')
    ax.set_ylabel(r'$s_{max}^{+/-}$ [$I_c$]')          
    ax.legend() # loc = 'lower right'  
    
    # same with restricted axes
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1] ) )
    plt.suptitle('{}'.format(title) )
    
    color_list = ['blue1','blue2','blue3','blue4','blue5','green1','green2','green3','green4','green5','yellow1','yellow2','yellow3','yellow4','yellow5','red1','red2','red3','red4','red5']
    for ii in range(len(ib__list)): 
        _phi_r__vec = np.asarray( phi_r__array[ii] )
        ax.plot(_phi_r__vec[ np.where( _phi_r__vec[:] > 0 ) ], s_max_plus__array[ii], color = colors[color_list[ii]], label = 'ib = {:5.3f}'.format(ib__list[ii]))
        ax.plot(_phi_r__vec[ np.where( _phi_r__vec[:] < 0 ) ], s_max_minus__array[ii], color = colors[color_list[ii]], label = 'ib = {:5.3f}'.format(ib__list[ii]))        
    
    ax.set_ylim([0,ax.get_ylim()[1]])
    ax.set_xlim([-1/2,1/2])
    ax.set_xlabel(r'$\phi_r$ [$\Phi_0$]')
    ax.set_ylabel(r'$s_{max}^{+/-}$ [$I_c$]')          
    ax.legend() # loc = 'lower right' 
    
    return


# =============================================================================
# rate array thresholds and saturation points
# =============================================================================


# =============================================================================
# nine-pixel classifier 
# =============================================================================

def plot_nine_pixel_num_spikes_vs_input(drive_dict,num_spikes_out__vec):
    
    index_list = np.linspace(1,len(drive_dict),len(drive_dict))
    label_list = []
    for _key in drive_dict:
        label_list.append(_key)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle('nine-pixel classifier')
    
    ax.plot(index_list, num_spikes_out__vec, color = colors['blue3'])
    ax.set_ylabel(r'number of spikes out')          
    ax.set_xlabel(r'input image')
    ax.legend() # loc = 'lower right'
    
    ax.set_xticks(index_list)
    ax.set_xticklabels(label_list)
    
    ax.grid(which = 'major', axis = 'both', color = colors['grey4'])
    # ax.grid(which='minor', linewidth = 0.15)
            
    ax.set_xlabel(r'input image')
    
    return

def plot_nine_pixel_activations_vs_input(drive_dict,s_z__vec,s_v__vec,s_n__vec):
    
    index_list = np.linspace(1,len(drive_dict),len(drive_dict))
    label_list = []
    for _key in drive_dict:
        label_list.append(_key)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False, figsize = ( fig_size[0] , fig_size[1]) )
    plt.suptitle('nine-pixel classifier')
    
    ax.plot(index_list, s_z__vec, color = colors['blue3'], label = 'z')
    ax.plot(index_list, s_v__vec, color = colors['red3'], label = 'v')
    ax.plot(index_list, s_n__vec, color = colors['green3'], label = 'n')
    ax.set_ylabel(r'$s_{max}$ [$I_c$]')          
    ax.set_xlabel(r'input image')
    ax.legend() # loc = 'lower right'
    
    ax.set_xticks(index_list)
    ax.set_xticklabels(label_list)
    
    ax.grid(which = 'major', axis = 'both', color = colors['grey4'])
    # ax.grid(which='minor', linewidth = 0.15)
            
    ax.set_xlabel(r'input image')
    
    return


# =============================================================================
# end nine-pixel classifier 
# =============================================================================