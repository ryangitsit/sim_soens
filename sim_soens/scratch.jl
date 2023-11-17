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

        add_input(node,node_name,syn_names,net_dict)

    end

    for t_idx in 1:net_dict["T"]-1
        Threads.@threads for idx in 1:length(node_names)
            node = net_dict["nodes"][node_names[idx]]
            node_name = node_names[idx]
            # loop_synapses(node,node_name,syn_names,net_dict["T"],net_dict["conversion"],t_idx,net_dict["dt"])
            # loop_dendrites(node,node_name,dend_names,net_dict["d_tau"],t_idx)
            loop_dendrites(node,node_name,dend_names,Float64(net_dict["d_tau"]),Int64(t_idx),net_dict)

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

function add_input(node::Dict{Any, Any},node_name::String,syn_names::Dict{String,Vector{Any}},net_dict::Dict{Any,Any})
    T          = net_dict["T"]
    conversion = net_dict["conversion"]
    dt         = net_dict["dt"]
    duration   = 1500
    hotspot    = 3
    gap        = Int64(floor(50/net_dict["dt"]))*.0000000000000001
    
    SPD = SPD_response(conversion,hotspot,dt)
    net_dict["SPD"] = SPD

    Threads.@threads for iter in 1:length(node["synapses"])
        syn = node["synapses"][syn_names[node_name][iter]]

        if length(syn.spike_times) > 0
            spike_time = syn.spike_times[1]

            until = min(spike_time+duration,T)
            syn.phi_spd[spike_time:until] = max.(syn.phi_spd[spike_time:until],SPD[1:until-spike_time+1])

            recent = spike_time

            counter = 0
            for (i,spike_time) in enumerate(syn.spike_times[2:end])
                
                if 1 == 1 #spike_time-recent > gap
                    until = min(spike_time+duration,T)
                    syn.phi_spd[spike_time:until] = max.(syn.phi_spd[spike_time:until],SPD[1:until-spike_time+1])
                    recent = spike_time
                    counter+=1
                else
                    print("Delete!")
                    deleteat!(syn.spike_times,counter+1)
                end

            end
        end

        # for (i,spike_time) in enumerate(syn.spike_times[2:end])
        #     if spike_time - syn.spike_times
        #     until = min(spike_time+duration,T)
        #     syn.phi_spd[spike_time:until] = max.(syn.phi_spd[spike_time:until],SPD[1:until-spike_time+1])
        # end
    end
end


function SPD_response(conversion::Float64,hs::Int64,dt::Float64)
    """
    Move to before time stepper
    """
    conversion = (conversion * .01) #.0155
    phi_peak = 0.5
    tau_rise = 0.02 *conversion
    tau_fall = 65   *conversion/(dt*10) #50
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


# function loop_synapses(node::Dict{Any, Any},node_name::String,syn_names::Dict{String,Vector{Any}},T::Int64,conversion::Float64,t_idx::Int64,dt::Float64)
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

# function loop_dendrites(node::Dict{Any, Any},node_name::String,dend_names::Dict{String,Vector{Any}},d_tau::Float64,t_idx::Int64)
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
        Threads.@threads for idx in 1:length(node_names)
            node = net_dict["nodes"][node_names[idx]]
            node_name = node_names[idx]
            loop_synapses(node,node_name,syn_names,net_dict["T"],net_dict["conversion"],t_idx,net_dict["dt"])
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

function loop_dendrites(node::Dict{Any, Any},node_name::String,dend_names::Dict{String,Vector{Any}},d_tau::Float64,t_idx::Int64,net_dict::Dict{Any, Any})
    Threads.@threads for iter in 1:length(node["dendrites"])
        dend = node["dendrites"][dend_names[node_name][iter]]
        # @show dend.name, dend.s_now, dend.phir_next
        # dend.s_now = dend.s_next
        # dend.phir_next = 0
        dend_update(
            node,
            dend,
            t_idx,
            d_tau,
            net_dict
            )
    end
end

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


function synapse_input_update(syn::Synapse,t::Int64,T::Int64,conversion::Float64,dt::Float64)
    if t in syn.spike_times
        duration = 1500
        hotspot = 3
        until = min(t+duration,T)
        syn.phi_spd[t:until] = max.(syn.phi_spd[t:until],SPD_response(conversion,hotspot,dt)[1:until-t+1])
    end
    return syn
end


function synapse_input_update(syn::RefractorySynapse,t::Int64,T::Int64,conversion::Float64,dt::Float64)
    if t in syn.spike_times
        duration = 1500
        hotspot = 2 
        # t = tau_vec[spk]
        until = min(t+duration,T)
        syn.phi_spd[t:until] = max.(syn.phi_spd[t:until],SPD_response(conversion,hotspot,dt)[1:until-t+1])
    end
    return syn
end


function SPD_response(conversion::Float64,hs::Int64,dt::Float64)
    """
    Move to before time stepper
    """
    conversion = (conversion * .01) #.0155
    phi_peak = 0.5
    tau_rise = 0.02 *conversion
    tau_fall = 65   *conversion/(dt*10) #50
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


# function dend_update(node::Dict,dend::ArborDendrite,t_idx::Int,d_tau::Float64)
#     dend_inputs(node,dend,t_idx)
#     dend_synputs(node,dend,t_idx)
#     dend_signal(dend,t_idx,d_tau::Float64)
# end
function dend_update(node::Dict,dend::ArborDendrite,t_idx::Int64,d_tau::Float64,net_dict::Dict{Any, Any}  )
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau::Float64)
end

# function dend_update(node::Dict,dend::RefractoryDendrite,t_idx::Int,d_tau::Float64)
#     # dend_inputs(node,dend,t_idx)
#     dend_synputs(node,dend,t_idx)
#     dend_signal(dend,t_idx,d_tau::Float64)
# end

function dend_update(node::Dict,dend::RefractoryDendrite,t_idx::Int64,d_tau::Float64,net_dict::Dict{Any, Any})
    # dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau::Float64)
end

function dend_update(node::Dict,dend::SomaticDendrite,t_idx::Int64,d_tau::Float64,net_dict::Dict{Any, Any})
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    if dend.s[t_idx] >= dend.threshold
        if isempty(dend.out_spikes) != true
            if t_idx .- last(dend.out_spikes) > dend.abs_ref
                spike(dend,t_idx,dend.syn_ref,node,net_dict)
            end
        else
            spike(dend,t_idx,dend.syn_ref,node,net_dict)
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

# function dend_update(node::Dict,dend::SomaticDendrite,t_idx::Int,d_tau::Float64)
#     dend_inputs(node,dend,t_idx)
#     dend_synputs(node,dend,t_idx)
#     if dend.s[t_idx] >= dend.threshold
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
#             if t_idx .- last(dend.out_spikes) > dend.abs_ref
#                 dend_signal(dend,t_idx,d_tau::Float64)
#             end
#         else
#             dend_signal(dend,t_idx,d_tau::Float64)
#         end
#     end
# end

function dend_update(node::Dict,dend::SomaticDendrite,t_idx::Int64,d_tau::Float64,net_dict::Dict{Any, Any})
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    if dend.s[t_idx] >= dend.threshold
        if isempty(dend.out_spikes) != true
            if t_idx .- last(dend.out_spikes) > dend.abs_ref
                spike(dend,t_idx,dend.syn_ref,node,net_dict)
            end
        else
            spike(dend,t_idx,dend.syn_ref,node,net_dict)
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


# function spike(dend::SomaticDendrite,t_idx::Int,syn_ref::AbstractSynapse) ## add spike to syn_ref
#     dend.spiked = 1
#     push!(dend.out_spikes,t_idx)
#     # dend.s[t_idx+1:length(dend.s)] .= 0
#     push!(syn_ref.spike_times,t_idx+1)
#     for (name,syn) in dend.syn_outs
#         dend.syn_outs[name]+=1
#     end

#     # syn_ref.phi_spd[t:until] = max.(syn.phi_spd[t:until],SPD_response(conversion,hotspot)[1:until-t+1])

# end


function spike(dend::SomaticDendrite,t_idx::Int64,syn_ref::AbstractSynapse,node::Dict{Any, Any},net_dict::Dict{Any, Any}) ## add spike to syn_ref
    # println("Spiked!")
    dend.spiked = 1
    push!(dend.out_spikes,t_idx)
    # dend.s[t_idx+1:length(dend.s)] .= 0
    push!(syn_ref.spike_times,t_idx+1)
    for (name,syn) in dend.syn_outs
        dend.syn_outs[name]+=1
    end
    
    T = net_dict["T"]
    SPD = net_dict["SPD"]
    # conversion = net_dict["conversion"]
    # dt = net_dict["dt"]
    spike_time = Int(floor(t_idx+(10/net_dict["dt"])))
    duration = Int(floor(200/net_dict["dt"]))
    # hotspot = 3
    until = Int(min(spike_time+duration,T))
    gap = Int64(floor(50/net_dict["dt"]))*.00000000001

    for (node_name,node) in net_dict["nodes"]
        for (out_node_name,syn_list) in node["firing_targets"]
            for syn_name in syn_list

                target_synapse = net_dict["nodes"][out_node_name]["synapses"][syn_name]
                past_spikes = filter(x -> x < spike_time, target_synapse.spike_times)
                if length(past_spikes) > 0
                    if 1 == 1 #spike_time - filter(x -> x < spike_time, target_synapse.spike_times)[end] > gap
                        insert!(target_synapse.spike_times,Int(length(past_spikes)),spike_time)
                        until = min(spike_time+duration,T)
                        target_synapse.phi_spd[spike_time:until] = max.(target_synapse.phi_spd[spike_time:until],SPD[1:until-spike_time+1])
                    end
                else
                    push!(target_synapse.spike_times,spike_time)
                    until = min(spike_time+duration,T)
                    target_synapse.phi_spd[spike_time:until] = max.(target_synapse.phi_spd[spike_time:until],SPD[1:until-spike_time+1])
                end

                syn_ref.phi_spd[spike_time:until] = max.(syn_ref.phi_spd[spike_time:until],SPD[1:until-spike_time+1])



            end
        end
    end

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