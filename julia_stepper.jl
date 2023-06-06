using PyCall

function counter(n)
    for i in 1:n
        @show i
    end
end

mutable struct WildCard
    params::Dict{Any, Any}
end

mutable struct Dendrite
    name      :: String
    s         :: Vector
    phir      :: Vector
    inputs    :: Dict
    synputs   :: Dict
    synspikes :: Dict
end

mutable struct Synapse
    name::String
    spike_times::Array
    phi_spd::Array
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
        # spike_times = Int.(syn.input_signal.spike_times.*conversion)
        spike_times = [floor(Int,x) for x in syn.input_signal.spike_times.*conversion]
        synapses[syn.name] = Synapse(syn.name,spike_times.+1,zeros(T))
        # @show syn.name
        # @show syn.input_signal.spike_times
    end

    for dend in node.dendrite_list
        inputs = Dict()
        for input in dend.dendritic_connection_strengths
            inputs[input[1]] = input[2]
        end
        synputs   = Dict()
        synspikes = Dict()

        for synput in dend.synaptic_connection_strengths
            # spike_times = Int.(node.synapse_list[1].input_signal.spike_times.*conversion)
            synputs[synput[1]] = synput[2]
            # synspikes[synput[1]] = spike_times.+1
        end

        new_dend = Dendrite(dend.name,zeros(T),zeros(T),inputs,synputs,synspikes)
        # push!(dendrites,new_dend)
        dendrites[dend.name] = new_dend
    end

    node_dict["synapses"] = synapses
    node_dict["dendrites"] = dendrites

    return node_dict
end

function stepper(net,tau_vec,d_tau)

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

    T = length(tau_vec)
    conversion = last(tau_vec)/(T/net.dt)
    @show conversion*400
    # @show tau_vec
    # T = 3

    # set up julia structs
    net_dict = Dict()
    py_dict = Dict()
    for node in net.nodes
        py_dict[node.name] = node
        net_dict[node.name] = make_nodes(node,T+1,conversion)
    end

    

    # @show net_dict

    
    ops=0
    dend_ops = 0
    @show T
    # T = 2

    for t_idx in 1:T
        for (node_name,node) in net_dict
            # @show node

            for (name,syn) in node["synapses"]
                synapse_input_update(syn,t_idx,T,conversion)
            end

            for (name,dend) in node["dendrites"]
                py_dend = py_dict[node_name].dend_dict[name]
                dend = dend_update(py_dend,node,dend,t_idx,tau_vec[t_idx],d_tau)
                # dend_ops += 1
                # ops+=1
            end
        end
        # count+=1 
    end

    # @show dend_ops
    # @show(ops)
    return net_dict
end

function synapse_input_update(syn,t,T,conversion)
    duration = floor(Int,1500*conversion)
    if t in syn.spike_times
        @show t
        until = min(t+duration,T)
        syn.phi_spd[t:until-1] = max.(syn.phi_spd[t:until-1],SPD_response(conversion)[1:until-t])
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

function dend_update(py_dend,node,dend,t_idx,t_now,d_tau)
    soma = 0
    update = true

    # if occursin("soma", dend.name) # rewrite as unique method
    #     if dend.threshold_flag == true
    #         update = false
    #     end
    # end

    # skipping update skip
    dend = dend_inputs(node,dend,t_idx)
    dend = dend_synputs(node,dend,t_idx)
    dend = dend_signal(py_dend,dend,t_idx,d_tau)

    return soma

end

function dend_inputs(node,dend,t_idx)
    update = 0
    # @show collect(keys(node[1]))
    for input in dend.inputs
        # @show input
        # @show node[2][input[1]].name
        # @show node[2][input[1]].s[t_idx]
        update += node["dendrites"][input[1]].s[t_idx]*input[2]
    end
    # @show update
    # push!(dend.phir,update)
    dend.phir[t_idx+1] += update
    return dend
end


function dend_synputs(node,dend,t_idx)
    update = 0
    for synput in dend.synputs
        if occursin("refraction",synput[1]) == 0
            # @show synput
            # @show node["synapses"][synput[1]] #.name
            update += node["synapses"][synput[1]].phi_spd[t_idx]*synput[2] # dend.s[t_idx]*input[2] + t_idx
            # @show update
        end
    end
    # @show update
    # push!(dend.phir,update)
    dend.phir[t_idx+1] += update
    return dend
end

function dend_signal(py_dend,dend,t_idx,d_tau)

    lst = py_dend.phi_r__vec
    val = dend.phir[t_idx] 
    _ind__phi_r = closest_index(lst,val)

    i_di__vec = py_dend.i_di__subarray[_ind__phi_r]

    lst = i_di__vec
    val = dend.s[t_idx]
    _ind__s = closest_index(lst,val)

    r_fq = py_dend.r_fq__subarray[_ind__phi_r][_ind__s]
    # @show r_fq
    dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*py_dend.alpha/py_dend.beta) + (d_tau/py_dend.beta)*r_fq
    
    return dend
end

function closest_index(lst,val)
    return findmin(abs.(lst.-val))[2] #indexing!
end