using PyCall

function counter(n)
    for i in 1:n
        @show i
    end
end

mutable struct WildCard
    params::Dict{Any, Any}
end

mutable struct dendrite
    name::String
    s::Vector
    phir::Vector
    inputs::Dict
    synputs::Dict
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

function make_dendrites(dends)
    dendrites = []
    for dend in dends
        inputs = Dict()
        for input in dend.dendritic_connection_strengths
            inputs[input[1]] = input[2]
        end
        synputs = Dict()
        for synput in dend.synaptic_connection_strengths
            synputs[synput[1]] = synput[2]
        end
        push!(dendrites,dendrite(dend.name,zeros(length(dend.s)),zeros(length(dend.s)),inputs,synputs))
    end
    return dendrites
end

function stepper(net,tau_vec,d_tau)
    net_dict = Dict()
    py_dict = Dict()
    for node in net.nodes
        py_dict[node.name] = node
        net_dict[node.name] = make_dendrites(node.dendrite_list)
    end

    

    # @show net_dict

    T = length(tau_vec)-1
    ops=0
    dend_ops = 0
    @show T
    ### pythonic
    # for t_idx in 1:T
    #     for node in net.nodes
    #         for dend in node.dendrite_list
    #             dend_ops += dend_update(dend,t_idx,tau_vec[t_idx+1],d_tau)
    #             ops+=1
    #         end
    #     end
    #     # count+=1
    # end

    ### julianic
    # T = 2
    for t_idx in 1:T
        for node in net_dict
            # @show node
            for dend in node[2]
                # @show dend.name
                py_dend = py_dict[node[1]].dend_dict[dend.name]
                dend = dend_update(py_dend,dend,t_idx,tau_vec[t_idx+1],d_tau)
                # dend_ops += dend_update(dend,t_idx,tau_vec[t_idx+1],d_tau)
                # ops+=1
            end
        end
        # count+=1
    end

    # @show dend_ops
    # @show(ops)
    return net_dict
end

function dend_update(py_dend,dend,t_idx,t_now,d_tau)
    soma = 0
    update = true

    # if occursin("soma", dend.name) # rewrite as unique method
    #     if dend.threshold_flag == true
    #         update = false
    #     end
    # end

    if update == true
        dend = dend_inputs(dend,t_idx)
        dend = dend_synputs(dend,t_idx)
        dend = dend_signal(py_dend,dend,t_idx,d_tau)
    end


    return soma
end

function dend_inputs(dend,t_idx)
    update = 0
    for input in dend.inputs
        update += t_idx*input[2] # dend.s[t_idx]*input[2] + t_idx
    end
    # @show update
    # push!(dend.phir,update)
    dend.phir[t_idx+1] += update
    return dend
end


function dend_synputs(dend,t_idx)
    update = 0
    for synput in dend.synputs
        update += t_idx*synput[2] # dend.s[t_idx]*input[2] + t_idx
    end
    # @show update
    # push!(dend.phir,update)
    dend.phir[t_idx+1] += update
    return dend
end

function dend_signal(py_dend,dend,t_idx,d_tau)

    lst = py_dend.phi_r__vec
    val = dend.phir[t_idx+1] 
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