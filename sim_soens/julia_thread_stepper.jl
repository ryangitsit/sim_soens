# using Distributed
# addprocs(2)

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
        Threads.@threads for idx in 1:length(node_names)
            node = net_dict["nodes"][node_names[idx]]
            node_name = node_names[idx]
            loop_synapses(node,node_name,syn_names,net_dict["T"],net_dict["conversion"],t_idx,net_dict["dt"])
            loop_dendrites(node,node_name,dend_names,net_dict["d_tau"],t_idx)
            
            if node["dendrites"][node["soma"]].spiked == 1
                for (syn_name,spks) in node["outputs"]
                    for (node_name,node) in net_dict["nodes"]
                        # if occursin(node_name,syn_name)
                        if node_name*'_' == syn_name[begin:length(node_name)+1]
                            # @show node_name, syn_name 
                            push!(net_dict["nodes"][node_name]["synapses"][syn_name].spike_times,t_idx+Int(10/net_dict["dt"]))
                        end
                    end
                end
                node["dendrites"][node["soma"]].spiked = 0
            end
            
        end
    end
    return net_dict
end

function loop_synapses(node::Dict{Any, Any},node_name::String,syn_names::Dict{String,Vector{Any}},T::Int64,conversion::Float64,t_idx::Int64,dt::Float64)
    duration = 1500 #Int64(250/dt)
    Threads.@threads for iter in 1:length(node["synapses"])
        syn = node["synapses"][syn_names[node_name][iter]]
        synapse_input_update(
            syn,
            t_idx,
            T,
            conversion,
            dt,
            duration
            )
    end
end

function loop_dendrites(node::Dict{Any, Any},node_name::String,dend_names::Dict{String,Vector{Any}},d_tau::Float64,t_idx::Int64)
    Threads.@threads for iter in 1:length(node["dendrites"])
        dend = node["dendrites"][dend_names[node_name][iter]]
        dend_update(
            node,
            dend,
            t_idx,
            d_tau
            )
    end
end

# function stepper(net_dict::Dict{String,Any})
#     # @show Threads.nthreads()
#     """
#     Plan:
#      - Go over all nodes in network
#         - create structs for synapses
#         - create structs for dendrites
#         - creat information table for connectivity
#         - update in downstream direction
#         - return dataframe of signals and fluxes
#             - update py objects with new info
#         - win
#     """
#     # net_dict["T"] = 10
#     syn_names  = Dict{String,Vector{Any}}()
#     dend_names = Dict{String,Vector{Any}}()
#     node_names = []

#     for (node_name,node) in net_dict["nodes"]
#         push!(node_names,node_name)
#         syns = collect(keys(node["synapses"]))
#         dends = collect(keys(node["dendrites"]))
#         syn_names[node_name] = syns
#         dend_names[node_name] = dends
#     end

#     for t_idx in 1:net_dict["T"]-1
#         Threads.@threads for idx in 1:length(node_names)
#             node = net_dict["nodes"][node_names[idx]]
#             node_name = node_names[idx]
#             loop_synapses(node,node_name,syn_names,net_dict["T"],net_dict["conversion"],t_idx,net_dict["dt"])
#             loop_dendrites(node,node_name,dend_names,net_dict["d_tau"],t_idx)

#             if node["dendrites"][node["soma"]].spiked == 1
#                 for (syn_name,spks) in node["outputs"]
#                     for (node_name,node) in net_dict["nodes"]
#                         if occursin(node_name,syn_name)
#                             push!(net_dict["nodes"][node_name]["synapses"][syn_name].spike_times,t_idx+100)
#                         end
#                     end
#                 end
#                 node["dendrites"][node["soma"]].spiked = 0
#             end
            
#         end
#     end
#     return net_dict
# end

# function loop_synapses(node::Dict{String, Any},node_name::String,syn_names::Dict{String,Vector{Any}},T::Int64,conversion::Float64,t_idx::Int64,dt::Float64)
#     Threads.@threads for iter in 1:length(node["synapses"])
#         syn = node["synapses"][syn_names[node_name][iter]]
#         synapse_input_update(
#             syn,
#             t_idx,
#             T,
#             conversion,
#             dt
#             )
#     end
# end

# function loop_dendrites(node::Dict{String, Any},node_name::String,dend_names::Dict{String,Vector{Any}},d_tau::Float64,t_idx::Int64)
#     Threads.@threads for iter in 1:length(node["dendrites"])
#         dend = node["dendrites"][dend_names[node_name][iter]]
#         dend_update(
#             node,
#             dend,
#             t_idx,
#             d_tau
#             )
#     end
# end


function synapse_input_update(syn::Synapse,t::Int64,T::Int64,conversion::Float64,dt::Float64,duration::Int64)
    # hotspot = 3
    if t in syn.spike_times
        spk_idx = findfirst(item -> item == t, syn.spike_times)
        if spk_idx > 1
            if syn.spike_times[spk_idx] - syn.spike_times[spk_idx-1] > 35
                # @show syn.spike_times[spk_idx], syn.spike_times[spk_idx-1]
                until = min(t+duration,T)
                syn.phi_spd[t:until] = max.(syn.phi_spd[t:until],SPD_response(conversion,dt)[1:until-t+1])
            else
                deleteat!(syn.spike_times,spk_idx)
            end
        else
            until = min(t+duration,T)
            syn.phi_spd[t:until] = max.(syn.phi_spd[t:until],SPD_response(conversion,dt)[1:until-t+1])
        end
    end
    return syn
end


function synapse_input_update(syn::RefractorySynapse,t::Int64,T::Int64,conversion::Float64,dt::Float64,duration::Int64)
    # hotspot = 3
    if t in syn.spike_times
        # t = tau_vec[spk]
        until = min(t+duration,T)
        syn.phi_spd[t:until] = max.(syn.phi_spd[t:until],SPD_response(conversion,dt)[1:until-t+1])
    end
    return syn
end


function SPD_response(conversion::Float64,dt::Float64)
    """
    Move to before time stepper
    """
    hs = 3
    conversion = (conversion * .01) #.0155
    phi_peak = 0.5
    tau_rise = 0.02    *conversion
    tau_fall = 65      *conversion/(dt*10) #50
    hotspot  = hs*.02  *conversion #- 22.925
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

    val = dend.phir[t_idx+1]

    if val > dend.phi_max
        # print("High roll")
        val = val - dend.phi_max
    elseif val < dend.phi_min
        # print("Low roll")
        val = val - dend.phi_min
    end


    ind_phi = index_approxer(val)

    # ind_phi = general_index_approxer(
    #     val,
    #     dend.abs_min_neg,
    #     dend.abs_min_pos,
    #     dend.abs_idx_neg,
    #     dend.abs_idx_pos
    #     )

    # ind_phi = searchsortedfirst(dend.phi_vec,val)
    # ind_phi= minimum([ind_phi,length(dend.phi_vec)])



    s_vec = dend.s_array[ind_phi]


    ind_s = s_index_approxer(s_vec,dend.s[t_idx])
    # ind_s = searchsortedfirst(s_vec,dend.s[t_idx])
    # ind_s= minimum([ind_s,length(s_vec)])

    r_fq = dend.r_array[ind_phi][ind_s]
    
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

function general_index_approxer(
    val::Float64,
    abs_min_neg::Float64,
    abs_min_pos::Float64,
    abs_idx_neg::Int64,
    abs_idx_pos::Int64
    )
    # ,maxval::Float64,minval::Float64,lenlst::Int
    if val <= abs_min_neg
        _ind__phi_r = minimum([floor(Int,(abs_idx_neg*(abs(val)-abs_min_pos)/(1-abs_min_pos))),667])
    elseif val >= abs_min_pos
        _ind__phi_r = minimum([floor(Int,(abs_idx_neg*(abs(val)-abs_min_pos)/(1-abs_min_pos)))+abs_idx_pos+1,667])
    elseif val < 0
        _ind__phi_r =  abs_idx_neg
    else
        _ind__phi_r = abs_idx_pos
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