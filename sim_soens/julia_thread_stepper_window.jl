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
            loop_dendrites(node,node_name,dend_names,Float32(net_dict["d_tau"]),Int32(t_idx),net_dict)
        end
    end
    return net_dict
end




function add_input(node::Dict{Any, Any},node_name::String,syn_names::Dict{String,Vector{Any}},net_dict::Dict{Any,Any})
    T          = net_dict["T"]
    conversion = Float32(net_dict["conversion"])
    dt         = Float32(net_dict["dt"])
    duration   = Int32(1500)
    hotspot    = Int32(3)
    gap        = Int32(floor(50/net_dict["dt"]))
    
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
                
                if spike_time-recent > gap
                    until = min(spike_time+duration,T)
                    syn.phi_spd[spike_time:until] = max.(syn.phi_spd[spike_time:until],SPD[1:until-spike_time+1])
                    recent = spike_time
                    counter+=1
                else
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

function loop_dendrites(node::Dict{Any, Any},node_name::String,dend_names::Dict{String,Vector{Any}},d_tau::Float32,t_idx::Int32,net_dict::Dict{Any, Any})
    Threads.@threads for iter in 1:length(node["dendrites"])
        dend = node["dendrites"][dend_names[node_name][iter]]
        # @show dend.name, dend.s_now, dend.phir_next
        dend.s_now = dend.s_next
        dend.phir_next = 0
        dend_update(
            node,
            dend,
            t_idx,
            d_tau,
            net_dict
            )
    end
end



function SPD_response(conversion::Float32,hs::Int32,dt::Float32)
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


function dend_update(node::Dict,dend::ArborDendriteWindowed,t_idx::Int32,d_tau::Float32,net_dict::Dict{Any, Any}  )
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau::Float32)
end


function dend_update(node::Dict,dend::RefractoryDendriteWindowed,t_idx::Int32,d_tau::Float32,net_dict::Dict{Any, Any})
    # dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau::Float32)
end



function dend_update(node::Dict,dend::SomaticDendriteWindowed,t_idx::Int32,d_tau::Float32,net_dict::Dict{Any, Any})
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    if dend.s_now >= dend.threshold
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
                dend_signal(dend,t_idx,d_tau::Float32)
            end
        else
            dend_signal(dend,t_idx,d_tau::Float32)
        end
    end
end


function spike(dend::SomaticDendriteWindowed,t_idx::Int32,syn_ref::AbstractSynapse,node::Dict{Any, Any},net_dict::Dict{Any, Any}) ## add spike to syn_ref
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
    gap = Int32(floor(50/net_dict["dt"]))

    for (node_name,node) in net_dict["nodes"]
        for (out_node_name,syn_list) in node["firing_targets"]
            for syn_name in syn_list

                target_synapse = net_dict["nodes"][out_node_name]["synapses"][syn_name]
                past_spikes = filter(x -> x < spike_time, target_synapse.spike_times)
                if length(past_spikes) > 0
                    if spike_time - filter(x -> x < spike_time, target_synapse.spike_times)[end] > gap
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


function dend_inputs(node::Dict,dend::AbstractDendrite,t_idx::Int32)
    update = 0
    for input in dend.inputs
        update += node["dendrites"][input[1]].s_now*input[2]
    end
    dend.phir_next += update
end


function dend_synputs(node::Dict,dend::AbstractDendrite,t_idx::Int32)
    update = 0
    for synput in dend.synputs 
        dend.phir_next += node["synapses"][synput[1]].phi_spd[t_idx]*synput[2] 
    end
end


function dend_signal(dend::AbstractDendrite,t_idx::Int32,d_tau::Float32)

    val = dend.phir_next

    if val > dend.phi_max
        # print("High roll")
        val = val - dend.phi_max
    elseif val < dend.phi_min
        # print("Low roll")
        val = val - dend.phi_min
    end


    ind_phi = index_approxer(val)

    s_vec = dend.s_array[ind_phi]


    ind_s = s_index_approxer(s_vec,dend.s_now)
    # ind_s = searchsortedfirst(s_vec,dend.s_now)
    # ind_s= minimum([ind_s,length(s_vec)])

    r_fq = dend.r_array[ind_phi][ind_s]
    
    dend.s_next = dend.s_now*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
    
end


function s_index_approxer(vec::Vector{Float32},val::Float32)
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

function index_approxer(val::Float32)
    # ,maxval::Float32,minval::Float32,lenlst::Int
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