using PyCall
# include("py_to_jul.jl")

using Distributed
# addprocs(3)

# module NewMain end
# using REPL
# REPL.activate(NewMain)
# using Distributed
# addprocs(2)
# using PyCall
# using julia_stepper


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
    const alpha     :: Float64
    const beta      :: Float64

    const phi_vec   :: Vector{Float64}
    const s_array   :: Vector{Vector{Float64}}
    const r_array   :: Vector{Vector{Float64}}

    ind_phi  :: Vector{Int64}
    ind_s    :: Vector{Int64}

    const phi_min   :: Float64
    const phi_max   :: Float64
    const phi_len   :: Int64

    flux_offset::Float64
end


mutable struct RefractoryDendrite <: AbstractDendrite
    name      :: String
    s         :: Vector{Float64}
    phir      :: Vector{Float64}
    inputs    :: Dict{String,Float64}
    synputs   :: Dict{String,Float64}
    const alpha     :: Float64
    const beta      :: Float64

    const phi_vec   :: Vector{Float64}
    const s_array   :: Vector{Vector{Float64}}
    const r_array   :: Vector{Vector{Float64}}

    ind_phi  :: Vector{Int64}
    ind_s    :: Vector{Int64}

    const phi_min   :: Float64
    const phi_max   :: Float64
    const phi_len   :: Int64

    flux_offset::Float64
end

mutable struct SomaticDendrite <: AbstractDendrite
    name       :: String
    s          :: Vector{Float64}
    phir       :: Vector{Float64}
    inputs     :: Dict{String,Float64}
    synputs    :: Dict{String,Float64}
    const alpha      :: Float64
    const beta       :: Float64

    const phi_vec    :: Vector{Float64}
    const s_array    :: Vector{Vector{Float64}}
    const r_array    :: Vector{Vector{Float64}}
 
    ind_phi    :: Vector{Int64}
    ind_s      :: Vector{Int64}
 
    const phi_min    :: Float64
    const phi_max    :: Float64
    const phi_len    :: Int64
 
    spiked     :: Int64
    out_spikes :: Vector{Int64}
    const threshold  :: Float64
    const abs_ref    :: Float64
    const syn_ref    :: AbstractSynapse
    const syn_outs   :: Dict{String,Int64}

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
    # save("net_dict_2.jld2", "data", net_dict)
    # load("net_dict.jld2")["data"]
    # save_dict(net_dict)
    return net_dict

end



function stepper(net_dict::Dict{Any,Any})
    # @show Threads.nthreads()
    """
    Plan:
     - Go over all nodes in network
        - create structs for synapses
        - create structs for dendrites
        - creat information table for connectivity
        - update in downstream direction
        - return dataframe of signals and fluxes
            - update py objects with new info
        - win
    """
    # net_dict["T"] = 10
    syn_names  = Dict{String,Vector{Any}}()
    dend_names = Dict{String,Vector{Any}}()
    node_names = []

    for (node_name,node) in net_dict["nodes"]
        push!(node_names,node_name)
        syns = collect(keys(node["synapses"]))
        dends = collect(keys(node["dendrites"]))
        syn_names[node_name] = syns
        dend_names[node_name] = dends
    end

    for t_idx in 1:net_dict["T"]-1
        for idx in 1:length(node_names)
            node = net_dict["nodes"][node_names[idx]]
            node_name = node_names[idx]
            loop_synapses(node,node_name,syn_names,net_dict["T"],net_dict["conversion"],t_idx)
            loop_dendrites(node,node_name,dend_names,net_dict["d_tau"],t_idx)

            if node["dendrites"][node["soma"]].spiked == 1
                for (syn_name,spks) in node["outputs"]
                    for (node_name,node) in net_dict["nodes"]
                        if occursin(node_name,syn_name)
                            push!(net_dict["nodes"][node_name]["synapses"][syn_name].spike_times,t_idx+100)
                        end
                    end
                end
                node["dendrites"][node["soma"]].spiked = 0
            end
            
        end
    end
    return net_dict
end

function loop_synapses(node::Dict{Any, Any},node_name::String,syn_names::Dict{String,Vector{Any}},T::Int64,conversion::Float64,t_idx::Int64)
    @distributed for iter in 1:length(node["synapses"])
        syn = node["synapses"][syn_names[node_name][iter]]
        synapse_input_update(
            syn,
            t_idx,
            T,
            conversion
            )
    end
end

function loop_dendrites(node::Dict{Any, Any},node_name::String,dend_names::Dict{String,Vector{Any}},d_tau::Float64,t_idx::Int64)
    @distributed for iter in 1:length(node["dendrites"])
        dend = node["dendrites"][dend_names[node_name][iter]]
        dend_update(
            node,
            dend,
            t_idx,
            d_tau
            )
    end
end

function stepper(net_dict::Dict{String,Any})
    # @show Threads.nthreads()
    """
    Plan:
     - Go over all nodes in network
        - create structs for synapses
        - create structs for dendrites
        - creat information table for connectivity
        - update in downstream direction
        - return dataframe of signals and fluxes
            - update py objects with new info
        - win
    """
    # net_dict["T"] = 10
    syn_names  = Dict{String,Vector{Any}}()
    dend_names = Dict{String,Vector{Any}}()
    node_names = []

    for (node_name,node) in net_dict["nodes"]
        push!(node_names,node_name)
        syns = collect(keys(node["synapses"]))
        dends = collect(keys(node["dendrites"]))
        syn_names[node_name] = syns
        dend_names[node_name] = dends
    end

    for t_idx in 1:net_dict["T"]-1
        for idx in 1:length(node_names)
            node = net_dict["nodes"][node_names[idx]]
            node_name = node_names[idx]
            loop_synapses(node,node_name,syn_names,net_dict["T"],net_dict["conversion"],t_idx)
            loop_dendrites(node,node_name,dend_names,net_dict["d_tau"],t_idx)

            if node["dendrites"][node["soma"]].spiked == 1
                for (syn_name,spks) in node["outputs"]
                    for (node_name,node) in net_dict["nodes"]
                        if occursin(node_name,syn_name)
                            push!(net_dict["nodes"][node_name]["synapses"][syn_name].spike_times,t_idx+100)
                        end
                    end
                end
                node["dendrites"][node["soma"]].spiked = 0
            end
            
        end
    end
    return net_dict
end

function loop_synapses(node::Dict{String, Any},node_name::String,syn_names::Dict{String,Vector{Any}},T::Int64,conversion::Float64,t_idx::Int64)
    @distributed for iter in 1:length(node["synapses"])
        syn = node["synapses"][syn_names[node_name][iter]]
        synapse_input_update(
            syn,
            t_idx,
            T,
            conversion
            )
    end
end

function loop_dendrites(node::Dict{String, Any},node_name::String,dend_names::Dict{String,Vector{Any}},d_tau::Float64,t_idx::Int64)
    @distributed for iter in 1:length(node["dendrites"])
        dend = node["dendrites"][dend_names[node_name][iter]]
        dend_update(
            node,
            dend,
            t_idx,
            d_tau
            )
    end
end


function synapse_input_update(syn::Synapse,t::Int64,T::Int64,conversion::Float64)
    if t in syn.spike_times
        duration = 1500
        hotspot = 3
        until = min(t+duration,T)
        syn.phi_spd[t:until] = max.(syn.phi_spd[t:until],SPD_response(conversion,hotspot)[1:until-t+1])
    end
    return syn
end


function synapse_input_update(syn::RefractorySynapse,t::Int64,T::Int64,conversion::Float64)
    if t in syn.spike_times
        duration = 1500
        hotspot = 2 
        # t = tau_vec[spk]
        until = min(t+duration,T)
        syn.phi_spd[t:until] = max.(syn.phi_spd[t:until],SPD_response(conversion,hotspot)[1:until-t+1])
    end
    return syn
end


function SPD_response(conversion::Float64,hs::Int64)
    """
    Move to before time stepper
    """
    conversion = conversion * .01 #.0155
    phi_peak = 0.5
    tau_rise = 0.02 *conversion
    tau_fall = 65   *conversion #50
    hotspot  = hs*.02    *conversion #- 22.925
    # @show phi_peak 
    # @show tau_rise 
    # @show tau_fall 
    # @show hotspot  
    coeff = phi_peak*(1-tau_rise/tau_fall)
    
    # duration = 500
    duration = floor(Int,8*50*conversion)

    phi_rise = [coeff * (1-exp(-t/tau_rise)) for t in 1:hotspot]
    phi_fall = [coeff * (1-exp(-hotspot/tau_rise))*exp(-(t-hotspot)/tau_fall) for t in hotspot+1:duration-1]

    return [0;phi_rise;phi_fall] #.-.0575
end


function dend_update(node::Dict,dend::ArborDendrite,t_idx::Int,d_tau::Float64)
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau::Float64)
end


function dend_update(node::Dict,dend::RefractoryDendrite,t_idx::Int,d_tau::Float64)
    # dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau::Float64)
end


# function dend_update(node::Dict,dend::SomaticDendrite,t_idx::Int,d_tau::Float64)
#     # if dend.s[t_idx] >= dend.threshold && t_idx .- dend.last_spike > dend.abs_ref
#     #     spike(dend,t_idx,dend.syn_ref)
#     if dend.s[t_idx] >= dend.threshold
#         # @show t_idx
#         if isempty(dend.out_spikes) != true
#             if t_idx .- last(dend.out_spikes) > dend.abs_ref
#                 spike(dend,t_idx,dend.syn_ref)
#             end
#         else
#             spike(dend,t_idx,dend.syn_ref)
#         end 
#     else
#         # if neuron has spiked, check if abs_ref cleared
#         if isempty(dend.out_spikes) != true 
#             # @show t_idx, t_idx .- last(dend.out_spikes)
#             if t_idx .- last(dend.out_spikes) > dend.abs_ref
#                 # @show t_idx
#                 dend_inputs(node,dend,t_idx)
#                 dend_synputs(node,dend,t_idx)
#                 dend_signal(dend,t_idx,d_tau::Float64)
#             else
#                 # dend_inputs(node,dend,t_idx)
#                 dend_synputs(node,dend,t_idx)
#             end
#         # else update
#         else
#             # @show t_idx
#             dend_inputs(node,dend,t_idx)
#             dend_synputs(node,dend,t_idx)
#             dend_signal(dend,t_idx,d_tau::Float64)
#         end
#     end

# end

function dend_update(node::Dict,dend::SomaticDendrite,t_idx::Int,d_tau::Float64)
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    if dend.s[t_idx] >= dend.threshold
        if isempty(dend.out_spikes) != true
            if t_idx .- last(dend.out_spikes) > dend.abs_ref
                spike(dend,t_idx,dend.syn_ref)
            end
        else
            spike(dend,t_idx,dend.syn_ref)
        end 
    else
        # if neuron has spiked, check if abs_ref cleared
        if isempty(dend.out_spikes) != true 
            if t_idx .- last(dend.out_spikes) > dend.abs_ref
                dend_signal(dend,t_idx,d_tau::Float64)
            end
        else
            dend_signal(dend,t_idx,d_tau::Float64)
        end
    end
end


function spike(dend::SomaticDendrite,t_idx::Int,syn_ref::AbstractSynapse) ## add spike to syn_ref
    dend.spiked = 1
    push!(dend.out_spikes,t_idx)
    # dend.s[t_idx+1:length(dend.s)] .= 0
    push!(syn_ref.spike_times,t_idx+1)
    for (name,syn) in dend.syn_outs
        dend.syn_outs[name]+=1
    end

    # syn_ref.phi_spd[t:until] = max.(syn.phi_spd[t:until],SPD_response(conversion,hotspot)[1:until-t+1])

end


function dend_inputs(node::Dict,dend::AbstractDendrite,t_idx::Int64)
    update = 0
    for input in dend.inputs
        update += node["dendrites"][input[1]].s[t_idx]*input[2]
    end
    dend.phir[t_idx+1] += update
end


function dend_synputs(node::Dict,dend::AbstractDendrite,t_idx::Int)
    update = 0
    for synput in dend.synputs 
        dend.phir[t_idx+1] += node["synapses"][synput[1]].phi_spd[t_idx]*synput[2] 
    end
end


function dend_signal(dend::AbstractDendrite,t_idx::Int,d_tau::Float64)

    # lst = dend.phi_vec
    val = dend.phir[t_idx+1]

    if val > dend.phi_max
        # print("High roll")
        val = val - dend.phi_max
    elseif val < dend.phi_min
        # print("Low roll")
        val = val - dend.phi_min
    end

    # ind_phi = closest_index(dend.phi_vec,val)
    ind_phi = index_approxer(val)
    # ind_phi = closest_index(dend.phi_vec,dend.phir[t_idx+1]) # +1 or not?

    # ind_phi = index_approxer(
    #     val,
    #     dend.phi_max,
    #     dend.phi_min,
    #     dend.phi_len
    #     )

    s_vec = dend.s_array[ind_phi]

    # ind_s = closest_index(s_vec,dend.s[t_idx])

    ind_s = s_index_approxer(s_vec,dend.s[t_idx])

    # @show ind_phi
    # @show s_vec
    # ind_s = index_approxer(
    #     dend.s[t_idx],
    #     first(s_vec),
    #     last(s_vec),
    #     length(s_vec)
    #     )

    # ind_s = index_approxer(dend.s[t_idx],first(s_vec),last(s_vec),length(s_vec))

    r_fq = dend.r_array[ind_phi][ind_s]
    # @show ind_phi,ind_s, r_fq
    # if occursin("soma",dend.name) #&& t_idx%10==0
    #     @show ind_phi,ind_s, r_fq
    # end
    dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
    
end


function closest_index(lst,val)
    return findmin(abs.(lst.-val))[2] #indexing!
end

function s_index_approxer(vec::Vector{Float64},val::Float64)
    if length(vec) > 1
    # s_idx = floor(Int, (val/(maximum(vec)-minimum(vec)))*length(vec) )
        slope = (last(vec) - vec[1])/length(vec)
        s_idx = maximum([floor(Int,((val-vec[1])/slope)),1])
        s_idx = minimum([s_idx,length(vec)])
    else
        s_idx = 1
    end
    return s_idx
end

function index_approxer(val::Float64)
    # ,maxval::Float64,minval::Float64,lenlst::Int
    if val <= -.1675
        _ind__phi_r = minimum([floor(Int,(333*(abs(val)-.1675)/(1-.1675))),667])
    elseif val >= .1675
        _ind__phi_r = minimum([floor(Int,(333*(abs(val)-.1675)/(1-.1675)))+335,667])
    elseif val < 0
        _ind__phi_r = 333
    else
        _ind__phi_r = 334
    end
    return _ind__phi_r + 1
end


function clear_all(strct)
    strct = nothing
    return strct
end

function unbindvariables()
    for name in names(Main)
        if !isconst(Main, name)
            Main.eval(:($name = nothing))
        end
    end
end

py"""
def jul_to_py(pynet,jul_net):
    for node in pynet.nodes:
        for i,dend in enumerate(node.dendrite_list):
            jul_dend = jul_net["nodes"][node.name]["dendrites"][dend.name]
            dend.s     = jul_dend.s #[:-1]
            dend.phi_r = jul_dend.phir #[:-1]

            dend.ind_phi = jul_dend.ind_phi #[:-1]
            dend.ind_s = jul_dend.ind_s #[:-1]
            dend.phi_vec = jul_dend.phi_vec #[:-1]

            if "soma" in dend.name:
                spike_times = (jul_dend.out_spikes-1)* pynet.dt * pynet.time_params["t_tau_conversion"]
                dend.spike_times        = spike_times
                node.neuron.spike_times = spike_times
            # # if net.print_times: print(sum(jul_net[node.name][i].s))/net.dt
        for i,syn in enumerate(node.synapse_list):
            jul_syn = jul_net["nodes"][node.name]["synapses"][syn.name]
            syn.phi_spd = jul_syn.phi_spd  

        import os
        import pickle
        pick = f'./temp_out.pickle'
        filehandler = open(pick, 'wb') 
        pickle.dump(pynet, filehandler)
        filehandler.close()
"""

py"""
import pickle
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
import sim_soens
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

picklin  = py"load_pickle"
picklit  = py"jul_to_py"

pynet = picklin("temp_net.pickle")
julnet = obj_to_structs(pynet)
stepper(julnet)
picklit(pynet,julnet)


