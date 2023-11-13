# module NewMain end
# using REPL
# REPL.activate(NewMain)
# using Distributed
# addprocs(2)
using PyCall
# using julia_stepper
using JLD2, FileIO

abstract type AbstractDendrite end

abstract type AbstractSynapse end

mutable struct SynapseWindowed<: AbstractSynapse
    name        ::String
    spike_times ::Vector{Int32}
    phi_spd     ::Vector{Float32}
end

mutable struct RefractorySynapseWindowed <: AbstractSynapse
    name        ::String
    spike_times ::Vector{Int32}
    phi_spd     ::Vector{Float32}
end


mutable struct ArborDendriteWindowed <: AbstractDendrite
    name              :: String

    s_now             :: Float32
    phir_now          :: Float32
    s_next            :: Float32
    phir_next         :: Float32

    inputs            :: Dict{String,Float32}
    synputs           :: Dict{String,Float32}
    const alpha       :: Float32
    const beta        :: Float32

    const phi_vec     :: Vector{Float32}
    const s_array     :: Vector{Vector{Float32}}
    const r_array     :: Vector{Vector{Float32}}

    ind_phi           :: Vector{Int32}
    ind_s             :: Vector{Int32}

    const phi_min     :: Float32
    const phi_max     :: Float32
    const phi_len     :: Int32

    const abs_min_neg :: Float32
    const abs_min_pos :: Float32
    const abs_idx_neg :: Int32
    const abs_idx_pos :: Int32

    flux_offset::Float32
end


mutable struct RefractoryDendriteWindowed <: AbstractDendrite
    name              :: String

    s_now             :: Float32
    phir_now          :: Float32
    s_next            :: Float32
    phir_next         :: Float32

    inputs    :: Dict{String,Float32}
    synputs   :: Dict{String,Float32}
    const alpha     :: Float32
    const beta      :: Float32

    const phi_vec   :: Vector{Float32}
    const s_array   :: Vector{Vector{Float32}}
    const r_array   :: Vector{Vector{Float32}}

    ind_phi  :: Vector{Int32}
    ind_s    :: Vector{Int32}

    const phi_min   :: Float32
    const phi_max   :: Float32
    const phi_len   :: Int32

    const abs_min_neg :: Float32
    const abs_min_pos :: Float32
    const abs_idx_neg :: Int32
    const abs_idx_pos :: Int32

    flux_offset::Float32
end

mutable struct SomaticDendriteWindowed <: AbstractDendrite
    name              :: String

    s_now             :: Float32
    phir_now          :: Float32
    s_next            :: Float32
    phir_next         :: Float32

    inputs          :: Dict{String,Float32}
    synputs         :: Dict{String,Float32}
    const alpha     :: Float32
    const beta      :: Float32
 
    const phi_vec   :: Vector{Float32}
    const s_array   :: Vector{Vector{Float32}}
    const r_array   :: Vector{Vector{Float32}}

    ind_phi         :: Vector{Int32}
    ind_s           :: Vector{Int32}

    const phi_min   :: Float32
    const phi_max   :: Float32
    const phi_len   :: Int32

    spiked          :: Int32
    out_spikes      :: Vector{Int32}
    const threshold :: Float32
    const abs_ref   :: Float32
    const syn_ref   :: AbstractSynapse
    const syn_outs  :: Dict{String,Int32}

    const abs_min_neg :: Float32
    const abs_min_pos :: Float32
    const abs_idx_neg :: Int32
    const abs_idx_pos :: Int32

    flux_offset::Float32
end

function obj_to_vect(obj)
    vect = Vector{Float32}[]
    for arr in obj
        push!(vect,convert(Vector{Float32},Float32.(arr)))
    end
    return vect
end
    
function make_synapses(node::PyObject,T::Int32,dt::Float32)
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
    T::Int32,
    synapses,
    phi_vec::Vector{Float32},
    s_array::Vector{Vector{Float32}},
    r_array::Vector{Vector{Float32}},
    dt::Float32,
    abs_min_neg::Float32,
    abs_min_pos::Float32,
    abs_idx_neg::Int32,
    abs_idx_pos::Int32
    )
    dendrites = Dict{String,AbstractDendrite}() #Dict()

    for dend in node.dendrite_list

        inputs = Dict{String,Float32}()
        for input in dend.dendritic_connection_strengths
            inputs[input[1]] = input[2]
        end

        synputs   = Dict{String,Float32}()
        for synput in dend.synaptic_connection_strengths
            # @show synput[1], synput[2]
            synputs[synput[1]] = synput[2]
        end

        # phi_vec = arr_list[1]::Vector{Float32}
        # s_array = obj_to_vect(arr_list[2])::Vector{Vector{Float32}}
        # r_array = obj_to_vect(arr_list[3])::Vector{Vector{Float32}}

        offset = minimum( [(maximum([dend.offset_flux,-dend.phi_th])), dend.phi_th] )
        # println(dend.offset_flux," or ",dend.phi_th," --> ",offset)

        if occursin("soma",dend.name)
            new_dend = SomaticDendriteWindowed( 
                dend.name,                              # name      :: String
                0,          
                1.0*offset, 
                0,          
                1.0*offset, 
                inputs,                                 # inputs    :: Dict
                synputs,                                # synputs   :: Dict
                dend.alpha,
                dend.beta,
                phi_vec,
                s_array,
                r_array,
                Int32[],
                Int32[],
                findmin(phi_vec)[1],
                findmax(phi_vec)[1],
                length(phi_vec),
                0,                                      # last spike
                Int32[],                                # spiked    :: Int
                dend.s_th,                              # threshold :: Float32
                dend.absolute_refractory_period/(dt), #*conversion,     # abs_ref   :: Float32
                synapses[node.name*"__syn_refraction"], # struct
                dend.syn_outs,
                abs_min_neg,
                abs_min_pos,
                abs_idx_neg,
                abs_idx_pos,
                dend.offset_flux
                )

        elseif occursin("ref",dend.name)
            new_dend = RefractoryDendriteWindowed(
                dend.name,
                0,          
                1.0*offset, 
                0,          
                1.0*offset, 
                inputs,
                synputs,
                dend.alpha,
                dend.beta,
                phi_vec,
                s_array,
                r_array,
                Int32[],
                Int32[],
                findmin(phi_vec)[1],
                findmax(phi_vec)[1],
                length(phi_vec),
                abs_min_neg,
                abs_min_pos,
                abs_idx_neg,
                abs_idx_pos,
                dend.offset_flux
                )

        else
            new_dend = ArborDendriteWindowed(
                dend.name,
                0,          
                1.0*offset, 
                0,          
                1.0*offset, 
                inputs,
                synputs,
                dend.alpha,
                dend.beta,
                phi_vec,
                s_array,
                r_array,
                Int32[],
                Int32[],
                findmin(phi_vec)[1],
                findmax(phi_vec)[1],
                length(phi_vec),
                abs_min_neg,
                abs_min_pos,
                abs_idx_neg,
                abs_idx_pos,
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
    T::Int32,
    dt::Float32,
    p::Vector{Float32},
    s::Vector{Vector{Float32}},
    r::Vector{Vector{Float32}},
    abs_min_neg::Float32,
    abs_min_pos::Float32,
    abs_idx_neg::Int32,
    abs_idx_pos::Int32
    )

    node_dict = Dict{String,Any}()
    node_dict["synapses"]  = make_synapses(node,T,dt)
    node_dict["dendrites"] = make_dendrites(
        node,T,node_dict["synapses"],p,s,r,dt,abs_min_neg,abs_min_pos,abs_idx_neg,abs_idx_pos
        )
    node_dict["outputs"] = node.neuron.dend_soma.syn_outs
    node_dict["firing_targets"] = node.neuron.dend_soma.firing_targets
    node_dict["soma"] = node.neuron.dend_soma.name

    return node_dict
end

function save_dict(net::Dict{String,Any})
    save("net_temp.jld2", "data", net)
end

function load_net(name)
    net_dict = load(name)["data"]
    return net_dict
end

function obj_to_structs(net::PyObject)

    net_dict  = Dict{String,Any}()
    node_dict = Dict{String,Any}()
    
    # arr_list = [net.phi_vec, net.s_array, net.r_array]

    p = net.phi_vec::Vector{Float32}
    s = obj_to_vect(net.s_array)::Vector{Vector{Float32}}
    r = obj_to_vect(net.r_array)::Vector{Vector{Float32}}

    # p = p .|> Vector{Float32}
    # s = s .|> Vector{Float32}
    # r = r .|> Vector{Float32}

    abs_min_neg = Float32(net.phi_vals["neg_min"])
    abs_min_pos = Float32(net.phi_vals["pos_min"])
    abs_idx_neg = Int32(net.phi_vals["neg_idx"])
    abs_idx_pos = Int32(net.phi_vals["pos_idx"])
    
    

    net_dict["nodes"] = node_dict::Dict
    net_dict["dt"] = Float32(net.dt)::Float32
    net_dict["d_tau"] = Float32(net.time_params["d_tau"])::Float32
    net_dict["conversion"] = Float32(net.time_params["t_tau_conversion"])::Float32
    net_dict["T"] = Int32(length(net.time_params["tau_vec"]))::Int32

    for node in net.nodes
        node_dict[node.name] = make_nodes(
            node,
            net_dict["T"],
            net_dict["dt"],
            p,
            s,
            r,
            abs_min_neg,
            abs_min_pos,
            abs_idx_neg,
            abs_idx_pos
            )
    end

    # for (node_name,node) in net_dict["nodes"]
    #     # @show node_name

    #     for (out_node_name,syn_list) in node["firing_targets"]
    #         # @show out_node_name

    #         for syn_name in syn_list
    #             # @show syn_name
    #             syn = net_dict["nodes"][out_node_name]["synapses"][syn_name]
    #             # node["outputs"][syn_name] = syn
    #         end
    #     end
    # end

    # save("net_dict_2.jld2", "data", net_dict)
    # load("net_dict.jld2")["data"]
    # save_dict(net_dict)
    return net_dict

end

