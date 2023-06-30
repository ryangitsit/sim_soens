# module NewMain end
# using REPL
# REPL.activate(NewMain)

function stepper(net_dict::Dict{Any,Any})

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
    for t_idx in 1:net_dict["T"]-1
        for (node_name,node) in net_dict["nodes"]
            # @show node
            for (name,syn) in node["synapses"]
                synapse_input_update(
                    syn,
                    t_idx,
                    net_dict["T"],
                    net_dict["conversion"],
                    )
            end

            for (name,dend) in node["dendrites"]
                dend_update(
                    node,
                    dend,
                    t_idx,
                    net_dict["d_tau"]
                    )
            end
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


function dend_update(node::Dict,dend::SomaticDendrite,t_idx::Int,d_tau::Float64)
    # if dend.s[t_idx] >= dend.threshold && t_idx .- dend.last_spike > dend.abs_ref
    #     spike(dend,t_idx,dend.syn_ref)
    if dend.s[t_idx] >= dend.threshold
        # @show t_idx
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
            # @show t_idx, t_idx .- last(dend.out_spikes)
            if t_idx .- last(dend.out_spikes) > dend.abs_ref
                # @show t_idx
                dend_inputs(node,dend,t_idx)
                dend_synputs(node,dend,t_idx)
                dend_signal(dend,t_idx,d_tau::Float64)
            else
                # dend_inputs(node,dend,t_idx)
                dend_synputs(node,dend,t_idx)
            end
        # else update
        else
            # @show t_idx
            dend_inputs(node,dend,t_idx)
            dend_synputs(node,dend,t_idx)
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

end


function dend_inputs(node::Dict,dend::AbstractDendrite,t_idx::Int64)
    update = 0
    for input in dend.inputs
        # if t_idx > 195 && t_idx < 205
        #     @show input, node["dendrites"][input[1]].phir[t_idx], node["dendrites"][input[1]].s[t_idx]*input[2]
        # end
        update += node["dendrites"][input[1]].s[t_idx]*input[2]
    end
    dend.phir[t_idx+1] += update
    # if t_idx > 195 && t_idx < 205
    #     println("---------------------------------------------------------------------")
    # end
end


function dend_synputs(node::Dict,dend::AbstractDendrite,t_idx::Int)
    update = 0
    for synput in dend.synputs 
        # if t_idx > 195 && t_idx < 205
        #     @show dend.name, synput, node["synapses"][synput[1]].phi_spd[t_idx]
        # end
        dend.phir[t_idx+1] += node["synapses"][synput[1]].phi_spd[t_idx]*synput[2] # dend.s[t_idx]*input[2] + t_idx
    end
    # if t_idx > 195 && t_idx < 205
    #     println("---------------------------------------------------------------------")
    # end
    # dend.phir[t_idx+1] += update
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
    ind_phi = closest_index(dend.phi_vec,dend.phir[t_idx+1]) # +1 or not?

    # ind_phi = index_approxer(
    #     val,
    #     dend.phi_max,
    #     dend.phi_min,
    #     dend.phi_len
    #     )

    s_vec = dend.s_array[ind_phi]

    ind_s = closest_index(s_vec,dend.s[t_idx])
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


function index_approxer(val::Float64,maxval::Float64,minval::Float64,lenlst::Int)
    if maxval == 0 && minval == 0
        return 1
    else
        range = abs(maxval-minval)::Float64
        ratio = abs(minval-val)/range::Float64
        ind = ratio*lenlst
        # ind = floor(Int,((val+range/2)/range)*lenlst)%lenlst+1
        # @show val
        # @show maxval
        # @show minval
        # @show range
        # @show ratio
        ind = min(floor(Int,ind)+1,lenlst-1) #%lenlst +1
        return ind + 1
    end
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