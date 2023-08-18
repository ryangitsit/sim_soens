# using PyCall
# using Statistics

include("py_to_jul.jl")
include("thread_stepper.jl")
# include("jul_MNIST.jl")
# py"setup"()

function load_net(name)
    net_dict = load(name)["data"]
    # @show typeof(net_dict)
    return net_dict
end

function run_net(net)
    stepper(net)
    # output = Int64[]
    # for (node_name,node) in net["nodes"]
    #     # @show node_name
    #     # @show (node["dendrites"][node["soma"]].out_spikes) * (net["dt"] / net["conversion"])
    #     push!(output,length(node["dendrites"][node["soma"]].out_spikes))
    # end
    # @show output
    return
end

function arbor_update(net_dict,desired)

    syn_names  = Dict{String,Vector{Any}}()
    dend_names = Dict{String,Vector{Any}}()
    T = net_dict["T"]
    for (node_name,node) in net_dict["nodes"]
        syns = collect(keys(node["synapses"]))
        dends = collect(keys(node["dendrites"]))
        syn_names[node_name] = syns
        dend_names[node_name] = dends
    end

    count = 1
    count2 = 1
    for (node_name,node) in net_dict["nodes"]
        error = desired[count] - length(node["dendrites"][node["soma"]].out_spikes)
        for iter in 1:length(node["dendrites"])
            dend = node["dendrites"][dend_names[node_name][iter]]
            if occursin("ref",dend.name) != 1
                # if count2 == 1
                #     @show dend.name
                #     for n in fieldnames(typeof(dend))
                #         println(n," -- ",sizeof(getfield(dend,n)))
                #     end
                # end
                step = error*mean(dend.s)*0.0005
                flux = mean(dend.phir) + step #dend.offset_flux
                if flux > 0.5 || flux < 0
                    step = -step
                end
                dend.flux_offset += step
                dend.phir = ones(T).*dend.flux_offset
                dend.s = zeros(T)
                
                # if count2 == 1
                #     @show dend.name
                #     for n in fieldnames(typeof(dend))
                #         println(n," -- ",sizeof(getfield(dend,n)))
                #     end
                # end
                count2+=1

            end
        end
        count+=1
        node["dendrites"][node["soma"]].out_spikes = Int64[]
        node["dendrites"][node["soma"]].spiked = 0
    end
end

function save_net(net::Dict{Any,Any})
    save("net_temp.jld2", "data", net)
end

@show Threads.nthreads()
name = "net_temp.jld2"
net = load_net(name)
run_net(net)
save_net(net)