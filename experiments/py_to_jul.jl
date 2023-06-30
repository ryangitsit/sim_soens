# module NewMain end
# using REPL
# REPL.activate(NewMain)

using PyCall

abstract type AbstractDendrite end

abstract type AbstractSynapse end

mutable struct Synapse <: AbstractSynapse
    name        ::String
    spike_times ::Vector{Int64}
    phi_spd     ::Vector{Float64}
end

mutable struct RefractorySynapse <: AbstractSynapse
    name        ::String
    spike_times ::Vector{Int64}
    phi_spd     ::Vector{Float64}
end


mutable struct ArborDendrite <: AbstractDendrite
    name      :: String
    s         :: Vector{Float64}
    phir      :: Vector{Float64}
    inputs    :: Dict{String,Float64}
    synputs   :: Dict{String,Float64}
    alpha     :: Float64
    beta      :: Float64

    phi_vec   :: Vector{Float64}
    s_array   :: Vector{Vector{Float64}}
    r_array   :: Vector{Vector{Float64}}

    ind_phi  :: Vector{Int64}
    ind_s    :: Vector{Int64}

    phi_min   :: Float64
    phi_max   :: Float64
    phi_len   :: Int64

    flux_offset::Float64
end


mutable struct RefractoryDendrite <: AbstractDendrite
    name      :: String
    s         :: Vector{Float64}
    phir      :: Vector{Float64}
    inputs    :: Dict{String,Float64}
    synputs   :: Dict{String,Float64}
    alpha     :: Float64
    beta      :: Float64

    phi_vec   :: Vector{Float64}
    s_array   :: Vector{Vector{Float64}}
    r_array   :: Vector{Vector{Float64}}

    ind_phi  :: Vector{Int64}
    ind_s    :: Vector{Int64}

    phi_min   :: Float64
    phi_max   :: Float64
    phi_len   :: Int64

    flux_offset::Float64
end

mutable struct SomaticDendrite <: AbstractDendrite
    name       :: String
    s          :: Vector{Float64}
    phir       :: Vector{Float64}
    inputs     :: Dict{String,Float64}
    synputs    :: Dict{String,Float64}
    alpha      :: Float64
    beta       :: Float64
 
    phi_vec    :: Vector{Float64}
    s_array    :: Vector{Vector{Float64}}
    r_array    :: Vector{Vector{Float64}}
 
    ind_phi    :: Vector{Int64}
    ind_s      :: Vector{Int64}
 
    phi_min    :: Float64
    phi_max    :: Float64
    phi_len    :: Int64
 
    spiked     :: Int64
    out_spikes :: Vector{Int64}
    threshold  :: Float64
    abs_ref    :: Float64
    syn_ref    :: AbstractSynapse
    syn_outs   :: Dict{String,Int64}

    flux_offset::Float64
end

function obj_to_vect(obj)
    vect = Vector{Float64}[]
    for arr in obj
        push!(vect,convert(Vector{Float64},arr))
    end
    return vect
end
    
function make_synapses(node::PyObject,T::Int64,dt::Float64)
    synapses = Dict()

    for syn in node.synapse_list
        spike_times = [floor(Int,x) for x in syn.input_signal.spike_times./dt]
        synapses[syn.name] = Synapse(syn.name,spike_times.+1,zeros(T))

        # if occursin("ref",syn.name)
        #     spike_times = []
        #     syn_ref = RefractorySynapse(syn.name,spike_times.+1,zeros(T))
        #     synapses[syn.name] = syn_ref
            
        # else
        #     spike_times = [floor(Int,x) for x in syn.input_signal.spike_times./dt]
        #     synapses[syn.name] = Synapse(syn.name,spike_times.+1,zeros(T))
        # end
    end

    syn = node.refractory_synapse
    spike_times = []
    syn_ref = RefractorySynapse(syn.name,spike_times.+1,zeros(T))
    synapses[syn.name] = syn_ref

    return synapses
end

function  make_dendrites(
    node::PyObject,
    T::Int64,
    synapses,
    phi_vec::Vector{Float64},
    s_array::Vector{Vector{Float64}},
    r_array::Vector{Vector{Float64}},
    dt::Float64
    )
    dendrites = Dict{String,AbstractDendrite}() #Dict()

    for dend in node.dendrite_list

        inputs = Dict{String,Float64}()
        for input in dend.dendritic_connection_strengths
            inputs[input[1]] = input[2]
        end

        synputs   = Dict{String,Float64}()
        for synput in dend.synaptic_connection_strengths
            # @show synput[1], synput[2]
            synputs[synput[1]] = synput[2]
        end

        # phi_vec = arr_list[1]::Vector{Float64}
        # s_array = obj_to_vect(arr_list[2])::Vector{Vector{Float64}}
        # r_array = obj_to_vect(arr_list[3])::Vector{Vector{Float64}}

        if occursin("soma",dend.name)
            new_dend = SomaticDendrite( 
                dend.name,                              # name      :: String
                zeros(T),                               # s         :: Vector
                ones(T).*dend.offset_flux,                               # phir      :: Vector
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
                dend.absolute_refractory_period/(dt), #*conversion,     # abs_ref   :: Float64
                synapses[node.name*"__syn_refraction"], # struct
                dend.syn_outs,
                dend.offset_flux
                )
                
        elseif occursin("ref",dend.name)
            new_dend = RefractoryDendrite(
                dend.name,
                zeros(T),
                ones(T).*dend.offset_flux, 
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
                dend.offset_flux
                )

        else
            new_dend = ArborDendrite(
                dend.name,
                zeros(T),
                ones(T).*dend.offset_flux, 
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
                dend.offset_flux
                )
        end
            
        # push!(dendrites,new_dend)
        dendrites[dend.name] = new_dend
    end
    return dendrites
end

function make_nodes(
    node::PyObject,
    T::Int64,
    dt::Float64,
    p::Vector{Float64},
    s::Vector{Vector{Float64}},
    r::Vector{Vector{Float64}}
    )

    node_dict = Dict{String,Any}()
    node_dict["synapses"]  = make_synapses(node,T,dt)
    node_dict["dendrites"] = make_dendrites(node,T,node_dict["synapses"],p,s,r,dt)
    node_dict["outputs"] = node.neuron.dend_soma.syn_outs
    node_dict["soma"] = node.neuron.dend_soma.name

    return node_dict
end

function obj_to_structs(net::PyObject)

    net_dict  = Dict{String,Any}()
    node_dict = Dict{String,Any}()
    
    # arr_list = [net.phi_vec, net.s_array, net.r_array]

    p = net.phi_vec::Vector{Float64}
    s = obj_to_vect(net.s_array)::Vector{Vector{Float64}}
    r = obj_to_vect(net.r_array)::Vector{Vector{Float64}}
    
    net_dict["nodes"] = node_dict::Dict
    net_dict["dt"] = net.dt::Float64
    net_dict["d_tau"] = net.time_params["d_tau"]::Float64
    net_dict["conversion"] = net.time_params["t_tau_conversion"]::Float64
    net_dict["T"] = length(net.time_params["tau_vec"])::Int64

    for node in net.nodes
        node_dict[node.name] = make_nodes(
            node,
            net_dict["T"],
            net_dict["dt"],
            p,
            s,
            r
            )
    end

    return net_dict

end