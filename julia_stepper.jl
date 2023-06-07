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

    # T = length(tau_vec)

    # conversion = last(tau_vec)/(T/net_dict["dt"])


    # # set up julia structs
    # net_dict = Dict()
    # for node in net.nodes
    #     net_dict[node.name] = make_nodes(node,T+1,conversion)
    # end

    # @show T
    for t_idx in 1:net_dict["T"]
        for (node_name,node) in net_dict["nodes"]
            # @show node
            for (name,syn) in node["synapses"]
                synapse_input_update(
                    syn,
                    t_idx,
                    net_dict["T"],
                    net_dict["conversion"]
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

function synapse_input_update(syn,t,T,conversion)
    duration = floor(Int,1500*conversion)
    if t in syn.spike_times
        until = min(t+duration,T)
        syn.phi_spd[t:until-2] = max.(syn.phi_spd[t:until-2],SPD_response(conversion)[1:until-t-1])
    end
    return syn
end

function SPD_response(conversion)
    phi_peak = 0.5
    tau_rise = 0.02*conversion
    tau_fall = 50*conversion #50
    hotspot = 3*conversion
    coeff = phi_peak*(1-tau_rise/tau_fall)
    # duration = 500
    duration = floor(Int,1500*conversion)
    e = ℯ

    phi_rise = [coeff * (1-e^(-t/tau_rise)) for t in 1:hotspot]
    phi_fall = [coeff * (1-e^(-hotspot/tau_rise))*e^(-(t-hotspot)/tau_fall) for t in hotspot+1:duration-1]

    return [0;phi_rise;phi_fall]
end

function dend_update(node,dend::ArborDendrite,t_idx,t_now,d_tau)
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau)
end

function dend_update(node,dend::RefractoryDendrite,t_idx,t_now,d_tau)
    dend_inputs(node,dend,t_idx)
    dend_synputs(node,dend,t_idx)
    dend_signal(dend,t_idx,d_tau)
end

function dend_update(node,dend::SomaticDendrite,t_idx,t_now,d_tau)
    if t_idx - dend.spiked > dend.abs_ref && dend.s[t_idx] >= dend.threshold
        spike(dend,t_idx,dend.syn_ref)
    else
        dend_inputs(node,dend,t_idx)
        dend_synputs(node,dend,t_idx)
        dend_signal(dend,t_idx,d_tau)
    end
end

function spike(dend,t_idx,syn_ref) ## add spike to syn_ref
    dend.spiked = t_idx
    dend.s[t_idx:length(dend.s)] .= 0
    push!(syn_ref.spike_times,t_idx+1)
end


function dend_inputs(node,dend,t_idx)
    update = 0
    for input in dend.inputs
        update += node["dendrites"][input[1]].s[t_idx]*input[2]
    end
    dend.phir[t_idx+1] += update
end


function dend_synputs(node,dend,t_idx)
    update = 0

    for synput in dend.synputs
        update += node["synapses"][synput[1]].phi_spd[t_idx]*synput[2] # dend.s[t_idx]*input[2] + t_idx
    end
    dend.phir[t_idx+1] += update
end

function dend_signal(dend,t_idx,d_tau)

    lst = dend.phi_vec
    val = dend.phir[t_idx]
    _ind__phi_r = closest_index(lst,val)

    s_vec = dend.s_array[_ind__phi_r]

    lst = s_vec
    val = dend.s[t_idx]
    _ind__s = closest_index(lst,val)

    r_fq = dend.r_array[_ind__phi_r][_ind__s]
    dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
    
end

function closest_index(lst,val)
    return findmin(abs.(lst.-val))[2] #indexing!
end