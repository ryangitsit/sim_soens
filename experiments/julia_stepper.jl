function stepper(net_dict)

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
    for t_idx in 1:net_dict["T"]
        for (node_name,node) in net_dict["nodes"]
            # @show node
            for (name,syn) in node["synapses"]
                synapse_input_update(
                    syn,
                    t_idx,
                    net_dict["T"],
                    net_dict["conversion"],
                    net_dict["tau_vec"],
                    )
            end

            for (name,dend) in node["dendrites"]
                dend_update(
                    node,
                    dend,
                    t_idx,
                    net_dict["tau_vec"][t_idx],
                    net_dict["d_tau"]
                    )
            end
        end
    end
    return net_dict
end


function synapse_input_update(syn,t,T,conversion,tau_vec)
    duration = 1500 #floor(Int,1500*conversion)
    if t in syn.spike_times
        # t = tau_vec[spk]
        until = min(t+duration,T)
        syn.phi_spd[t:until-2] = max.(syn.phi_spd[t:until-2],SPD_response(conversion)[1:until-t-1])
    end
    return syn
end


function SPD_response(conversion)
    phi_peak = 0.5
    tau_rise = 0.02 *conversion
    tau_fall = 50   *conversion #50
    hotspot  = 3    *conversion
    coeff = phi_peak*(1-tau_rise/tau_fall)
    
    # duration = 500
    duration = floor(Int,1500*conversion)
    e = ℯ
    # e = 2.71

    phi_rise = [coeff * (1-e^(-t/tau_rise)) for t in 1:hotspot]
    phi_fall = [coeff * (1-e^(-hotspot/tau_rise))*e^(-(t-hotspot)/tau_fall) for t in hotspot+1:duration-1]

    return [0;phi_rise;phi_fall]
end


function dend_update(node::Dict,dend::ArborDendrite,t_idx::Int,t_now,d_tau::Float64)
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau::Float64)
end


function dend_update(node::Dict,dend::RefractoryDendrite,t_idx::Int,t_now,d_tau::Float64)
    # dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau::Float64)
end


function dend_update(node::Dict,dend::SomaticDendrite,t_idx::Int,t_now,d_tau::Float64)
    # if dend.s[t_idx] >= dend.threshold && t_idx .- dend.last_spike > dend.abs_ref
    #     spike(dend,t_idx,dend.syn_ref)
    if dend.s[t_idx] >= dend.threshold
        if isempty(dend.out_spikes) != true
            if t_idx .- last(dend.out_spikes) > dend.abs_ref
                spike(dend,t_idx,dend.syn_ref)
            end
        else
            spike(dend,t_idx,dend.syn_ref)
        end 
    else
        dend_inputs(node,dend,t_idx)
        dend_synputs(node,dend,t_idx)
        dend_signal(dend,t_idx,d_tau::Float64)
    end
end


function spike(dend::SomaticDendrite,t_idx::Int,syn_ref::Synapse) ## add spike to syn_ref
    dend.last_spike = t_idx
    push!(dend.out_spikes,t_idx)
    dend.s[t_idx+1:length(dend.s)] .= 0
    push!(syn_ref.spike_times,t_idx+1)
end


function dend_inputs(node::Dict,dend::AbstractDendrite,t_idx)
    update = 0
    for input in dend.inputs

        # if t_idx == 10
        #     print("dend: ",dend.name,"\n   dendin = ",input[1],"  --  ",input[2],"\n")
        #     # print(node["synapses"][synput[1]].phi_spd[t_idx]*synput[2])
        # end

        update += node["dendrites"][input[1]].s[t_idx]*input[2]
    end
    dend.phir[t_idx+1] += update
end


function dend_synputs(node::Dict,dend::AbstractDendrite,t_idx::Int)
    update = 0
    for synput in dend.synputs

        if t_idx == 10
            print("synapse: ",synput[1],"  --  ",synput[2],"\n")
            # print(node["synapses"][synput[1]].phi_spd[t_idx]*synput[2])
        end

        update += node["synapses"][synput[1]].phi_spd[t_idx]*synput[2] # dend.s[t_idx]*input[2] + t_idx
    end
    dend.phir[t_idx+1] += update
end

function dend_signal(dend::AbstractDendrite,t_idx::Int,d_tau::Float64)

    # lst = dend.phi_vec
    # val = dend.phir[t_idx+1]
    ind_phi = closest_index(dend.phi_vec,dend.phir[t_idx+1]) # +1 or not?
    # ind_phi = index_approxer(
    #     dend.phir[t_idx+1],
    #     dend.phi_max,
    #     dend.phi_min,
    #     dend.phi_len
    #     )

    s_vec = dend.s_array[ind_phi]

    ind_s = closest_index(s_vec,dend.s[t_idx])
    # ind_s = index_approxer(dend.s[t_idx],first(s_vec),last(s_vec),length(s_vec))

    r_fq = dend.r_array[ind_phi][ind_s]
    dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
    
end

function closest_index(lst,val)
    return findmin(abs.(lst.-val))[2] #indexing!
end

function index_approxer(val::Float64,maxval::Float64,minval::Float64,lenlst::Int)
    range = maxval-minval
    ind = floor(Int,((val+range/2)/range)*lenlst)%lenlst+1
    return ind
end