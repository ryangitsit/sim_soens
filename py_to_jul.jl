using PyCall

abstract type AbstractDendrite end

mutable struct Synapse
    name::String
    spike_times::Array
    phi_spd::Array
end


mutable struct ArborDendrite <: AbstractDendrite
    name      :: String
    s         :: Vector
    phir      :: Vector
    inputs    :: Dict
    synputs   :: Dict
    alpha     :: Float64
    beta      :: Float64
    phi_vec   :: Vector
    s_array   :: Array
    r_array   :: Array
end


mutable struct RefractoryDendrite <: AbstractDendrite
    name      :: String
    s         :: Vector
    phir      :: Vector
    inputs    :: Dict
    synputs   :: Dict
    alpha     :: Float64
    beta      :: Float64
    phi_vec   :: Vector
    s_array   :: Array
    r_array   :: Array
end

mutable struct SomaticDendrite <: AbstractDendrite
    name      :: String
    s         :: Vector
    phir      :: Vector
    inputs    :: Dict
    synputs   :: Dict
    alpha     :: Float64
    beta      :: Float64
    phi_vec   :: Vector
    s_array   :: Array
    r_array   :: Array

    spiked    :: Int
    threshold :: Float64
    abs_ref   :: Float64
    syn_ref   :: Synapse
end


function make_struct(obj,names,vals)
    params = Dict()
    for (i,name) in enumerate(names)
        params[names[i]] = vals[i]
    end
    obj_struct = WildCard(params)

    print(obj.nodes[1].neuron.ib)
    # ks = collect(keys(params))
    # # print(ks)
    print(obj_struct.params["nodes"][1].neuron.tau_ref)
    return obj_struct
end

function make_nodes(node,T,conversion)

    node_dict = Dict()

    dendrites = Dict()
    synapses = Dict()

    for syn in node.synapse_list
        if occursin("ref",syn.name)
            spike_times = []
            syn_ref = Synapse(syn.name,spike_times.+1,zeros(T))
            synapses[syn.name] = syn_ref
        else
            spike_times = [floor(Int,x) for x in syn.input_signal.spike_times.*conversion]
            synapses[syn.name] = Synapse(syn.name,spike_times.+1,zeros(T))
        end
    end

    for dend in node.dendrite_list
        inputs = Dict()
        for input in dend.dendritic_connection_strengths
            inputs[input[1]] = input[2]
        end
        synputs   = Dict()

        for synput in dend.synaptic_connection_strengths
            # spike_times = Int.(node.synapse_list[1].input_signal.spike_times.*conversion)
            synputs[synput[1]] = synput[2]
            # synspikes[synput[1]] = spike_times.+1
        end

        if occursin("soma",dend.name)
            new_dend = SomaticDendrite( 
                dend.name,                       # name      :: String
                zeros(T),                        # s         :: Vector
                zeros(T),                        # phir      :: Vector
                inputs,                          # inputs    :: Dict
                synputs,                         # synputs   :: Dict
                dend.alpha,
                dend.beta,
                dend.phi_r__vec,
                dend.i_di__subarray,
                dend.r_fq__subarray,
                0,                               # spiked    :: Int
                dend.s_th,                       # threshold :: Float64
                dend.absolute_refractory_period, # abs_ref   :: Float64
                synapses["rand_neuron_77132__syn_refraction"], # struct
                )

        elseif occursin("ref",dend.name)
            new_dend = RefractoryDendrite(
                dend.name,
                zeros(T),
                zeros(T),
                inputs,
                synputs,
                dend.alpha,
                dend.beta,
                dend.phi_r__vec,
                dend.i_di__subarray,
                dend.r_fq__subarray,
                )

        else
            new_dend = ArborDendrite(
                dend.name,
                zeros(T),
                zeros(T),
                inputs,
                synputs,
                dend.alpha,
                dend.beta,
                dend.phi_r__vec,
                dend.i_di__subarray,
                dend.r_fq__subarray,
                )
        end
            
        # push!(dendrites,new_dend)
        dendrites[dend.name] = new_dend
    end

    node_dict["synapses"] = synapses
    node_dict["dendrites"] = dendrites

    return node_dict
end

function obj_to_structs(net)
    tau_vec = net.time_params["tau_vec"]
    T = length(tau_vec)
    conversion = last(tau_vec)/(T/net.dt)

    net_dict  = Dict()
    node_dict = Dict()
    for node in net.nodes
        node_dict[node.name] = make_nodes(node,T+1,conversion)
    end
    net_dict["nodes"] = node_dict
    net_dict["dt"] = net.dt
    net_dict["tau_vec"] = tau_vec
    net_dict["d_tau"] = net.time_params["d_tau"]
    net_dict["conversion"] = conversion
    net_dict["T"] = T
    return net_dict

end