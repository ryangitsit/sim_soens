using PyCall

abstract type AbstractDendrite end

abstract type AbstractSynapse end

mutable struct Synapse <: AbstractSynapse
    name::String
    spike_times::Array
    phi_spd::Array
end

mutable struct RefractorySynapse <: AbstractSynapse
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

    phi_vec   :: Vector{Float64}
    s_array   :: Vector{Vector{Float64}}
    r_array   :: Vector{Vector{Float64}}

    ind_phi  :: Vector{Int64}
    ind_s    :: Vector{Int64}

    # phi_vec   :: Vector
    # s_array   :: Array
    # r_array   :: Array

    phi_min   :: Float64
    phi_max   :: Float64
    phi_len   :: Int64
end


mutable struct RefractoryDendrite <: AbstractDendrite
    name      :: String
    s         :: Vector
    phir      :: Vector
    inputs    :: Dict
    synputs   :: Dict
    alpha     :: Float64
    beta      :: Float64

    phi_vec   :: Vector{Float64}
    s_array   :: Vector{Vector{Float64}}
    r_array   :: Vector{Vector{Float64}}

    ind_phi  :: Vector{Int64}
    ind_s    :: Vector{Int64}

    # phi_vec   :: Vector
    # s_array   :: Array
    # r_array   :: Array

    phi_min   :: Float64
    phi_max   :: Float64
    phi_len   :: Int64
end

mutable struct SomaticDendrite <: AbstractDendrite
    name       :: String
    s          :: Vector
    phir       :: Vector
    inputs     :: Dict
    synputs    :: Dict
    alpha      :: Float64
    beta       :: Float64
 
    phi_vec    :: Vector{Float64}
    s_array    :: Vector{Vector{Float64}}
    r_array    :: Vector{Vector{Float64}}
 
    ind_phi  :: Vector{Int64}
    ind_s    :: Vector{Int64}
 
    phi_min    :: Float64
    phi_max    :: Float64
    phi_len    :: Int64
 
    last_spike :: Int64
    out_spikes :: Vector
    threshold  :: Float64
    abs_ref    :: Float64
    syn_ref    :: AbstractSynapse
    syn_outs   :: Array
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

function obj_to_vect(obj)
    vect = Vector{Float64}[]
    for arr in obj
        push!(vect,convert(Vector{Float64},arr))
    end
    return vect
end
    
function make_synapses(node,T,conversion,dt)
    synapses = Dict()

    for syn in node.synapse_list
        if occursin("ref",syn.name)
            spike_times = []
            syn_ref = RefractorySynapse(syn.name,spike_times.+1,zeros(T))
            synapses[syn.name] = syn_ref
            
        else
            spike_times = [floor(Int,x) for x in syn.input_signal.spike_times./dt]#.*conversion]
            synapses[syn.name] = Synapse(syn.name,spike_times.+1,zeros(T))
        end
    end
    return synapses
end

function  make_dendrites(node,T,conversion,dt,synapses,arr_list)
    dendrites = Dict()
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

        # phi_vec = dend.phi_r__vec
        # s_array = obj_to_vect(dend.i_di__subarray)
        # r_array = obj_to_vect(dend.r_fq__subarray)

        phi_vec = arr_list[1]
        s_array = obj_to_vect(arr_list[2])
        r_array = obj_to_vect(arr_list[3])

        # phi_vec = dend.phi_r__vec
        # s_array = dend.i_di__subarray
        # r_array = dend.r_fq__subarray

        if occursin("soma",dend.name)
            # @show last(collect(keys(synapses)))
            # @show collect(keys(synapses))
            # @show dend.absolute_refractory_period
            new_dend = SomaticDendrite( 
                dend.name,                              # name      :: String
                zeros(T),                               # s         :: Vector
                zeros(T),                               # phir      :: Vector
                inputs,                                 # inputs    :: Dict
                synputs,                                # synputs   :: Dict
                dend.alpha,
                dend.beta,
                phi_vec,
                s_array,
                r_array,
                Int64[],
                Int64[],
                findmin(phi_vec)[1],
                findmax(phi_vec)[1],
                length(phi_vec),
                0,                                      # last spike
                Int64[],                                # spiked    :: Int
                dend.s_th,                              # threshold :: Float64
                dend.absolute_refractory_period, #*conversion,     # abs_ref   :: Float64
                synapses[node.name*"__syn_refraction"], # struct
                dend.syn_outs
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
                phi_vec,
                s_array,
                r_array,
                Int64[],
                Int64[],
                findmin(phi_vec)[1],
                findmax(phi_vec)[1],
                length(phi_vec),
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
                phi_vec,
                s_array,
                r_array,
                Int64[],
                Int64[],
                findmin(phi_vec)[1],
                findmax(phi_vec)[1],
                length(phi_vec),
                )
        end
            
        # push!(dendrites,new_dend)
        dendrites[dend.name] = new_dend
    end
    return dendrites
end

function make_nodes(node,T,conversion,dt,arr_list)

    node_dict = Dict()
    node_dict["synapses"]  = make_synapses(node,T,conversion,dt)
    node_dict["dendrites"] = make_dendrites(node,T,conversion,dt,node_dict["synapses"],arr_list)

    return node_dict
end

function obj_to_structs(net)

    net_dict  = Dict()
    node_dict = Dict()
    
    arr_list = [net.phi_vec, net.s_array, net.r_array]

    tau_vec = net.time_params["tau_vec"]
    T = length(tau_vec)
    # conversion = last(tau_vec)/(T/net.dt)
    
    net_dict["nodes"] = node_dict
    net_dict["dt"] = net.dt
    net_dict["tau_vec"] = tau_vec
    net_dict["d_tau"] = net.time_params["d_tau"]
    net_dict["conversion"] = net.time_params["t_tau_conversion"] # conversion
    net_dict["T"] = T

    for node in net.nodes
        node_dict[node.name] = make_nodes(
            node,
            T,
            net_dict["conversion"], #conversion,
            net_dict["dt"],
            arr_list
            )
    end
    @show net_dict["conversion"]
    return net_dict

end