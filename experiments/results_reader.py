import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_functions import *


def allow_ties(df,index):
    spike_str = df["spikes"][index][1:-1]
    spikes = np.fromstring(spike_str, dtype=int, sep=',')
    digit = int(df["digit"][index])
    sub = spikes - spikes[digit] 
    if sum(n > 0 for n in sub) == 0:
        return True
    else:
        return False
    
def no_ties(df,index):
    spike_str = df["spikes"][index][1:-1]
    spikes = np.fromstring(spike_str, dtype=int, sep=',')
    digit = int(df["digit"][index])
    sub = spikes - spikes[digit] 
    if sum(n > 0 for n in sub) == 0 and sum(n == 0 for n in sub) == 1:
        return True
    else:
        return False

def ongoing_performance(df):
    percents = []
    procents = []

    for start in range(len(df['sample'])-30):
        runs = 0
        run_wins = 0
        count = 0
        total = 0
        counts = np.array([0,0,0])
        totals = np.array([0,0,0])
        for index, row in df.iterrows():
            if index > start:
                total += 1
                totals[df["digit"][index]] += 1
                if df["digit"][index] == df["prediction"][index]:
                # if allow_ties(df,index) == True:
                # if no_ties(df,index) == True:
                    count+=1
                    counts[df["digit"][index]] += 1

        percents.append(count/total)
        procents.append(counts/totals)
    return percents, procents


def by_run_performance(df,decider,digits,samples,indivs):
    by_run = []
    by_dig_runs = [[] for _ in range(digits)]

    zrs = [0 for _ in range(digits)]

    dig_runs = zrs
    print(zrs)
    run_wins = 0
    if decider == 'ties':
        for index, row in df.iterrows():
            if allow_ties(df,index) == True:
                run_wins+=1
                dig_runs[df["digit"][index]]+=1
            if (index+1)%(digits*samples) == 0:
                by_run.append(run_wins/(digits*samples))
                for i,dig in enumerate(dig_runs):
                    by_dig_runs[i].append(dig/samples)
                dig_runs = [0 for _ in range(digits)]
                run_wins = 0

    if decider == 'winner':
        counter = 0
        for index, row in df.iterrows():
            
            if no_ties(df,index) == True:
                run_wins+=1
                # print(df["digit"][index])
                if indivs == True:
                    dig_runs[df["digit"][index]]+=1
                

            if (index+1)%(digits*samples) == 0:
                by_run.append(run_wins/(digits*samples))
                if indivs==True:
                    for i,dig in enumerate(dig_runs):
                        by_dig_runs[i].append(dig/samples)
                dig_runs = [0 for _ in range(digits)]
                run_wins = 0
                counter = 0

            # if index == len(df["digit"]) - 1 and (index+1)%(digits*samples) != 0:
            #     by_run.append(run_wins/(counter))
            #     for i,dig in enumerate(dig_runs):
            #         by_dig_runs[i].append(dig/(samples))

            counter +=1

    if decider == 'lucky':
        for index, row in df.iterrows():
            if df["digit"][index] == df["prediction"][index]:
                run_wins+=1
                dig_runs[df["digit"][index]]+=1
            if (index+1)%(digits*samples) == 0:
                by_run.append(run_wins/(digits*samples))
                for i,dig in enumerate(dig_runs):
                    by_dig_runs[i].append(dig/samples)
                dig_runs = [0 for _ in range(digits)]
                run_wins = 0

    return by_run, by_dig_runs






# df = pd.read_csv(
#     'MNIST_ongoing_julia.csv',
#     names=['sample','digit','spikes','error','prediction','time','init_time','run_time']
#     )




# # experiments = ['julia_inhibit_solver','MNIST_inelast','MNIST_unbounded','MNIST_eta']#,'MNIST_full']
# experiments = ['MNIST_unbounded','MNIST_deep_prime','MNIST_shallow_prime']
# until = 100000000

def plot_singles(experiments,until,digits,record='old'):
    for i,exp in enumerate(experiments):

        if record == 'old':
            df = pd.read_csv(
                f'results\MNIST\{exp}\learning_logger.csv',
                names=['sample','digit','spikes','error','prediction','time','init_time','run_time','offsets']
                )

            # percents, procents = ongoing_performance(df)
            by_run, digs = by_run_performance(df,'winner',10,50,True)

            pr = np.max(np.ceil(np.array(by_run)*30))
            print(f"Experiment {exp}, {len(by_run)} epochs, {np.round(pr*100/30,2)}% best run")

            print(f"Best run: {np.max(by_run)}% at {np.argmax(by_run)} out of {len(by_run)} runs")
            # plt.style.use('seaborn-muted')

        else:
            df = pd.read_csv(
                        f'results\MNIST\{exp}\performance_log.csv',
                        names=['all','digits']
                        )
            by_run = np.array(df["all"])
            digs = [[] for _ in range(digits)]
            for index, row in df.iterrows():
                for i,d in enumerate(digs):
                    str_arr = df["digits"][index][1:-1]
                    arr = []
                    num = ''
                    for ii in str_arr:
                        if ii != '.':
                            if ii != ' ':
                                num += ii
                        else:
                            arr.append(int(num))
                            num = ''
                    # if index == 10: print(arr)
                    digs[i].append(arr[i])



        plt.figure(figsize=(8,4))

        plt.title(f"MNIST Training Classification Performance - {exp}",fontsize=16)
        plt.xlabel("Epoch",fontsize=14)
        plt.ylabel("Accuracy",fontsize=14)
        plt.plot(by_run, linewidth = 4, label="Total")
        for ii, dig in enumerate(digs):
            plt.plot(dig, '--',label=ii)#, label=['0','1','2'])
        # plt.ylim(0,1.025)

        plt.legend()
        plt.show()

        
        # print(np.sum(df["run_time"]))
        # plt.plot(df["run_time"])
        # plt.show()

        # print("Average runtime = ",np.mean(df["run_time"]))

def plot_all(experiments,until,record='old'):
    plt.style.use('seaborn-muted')
    plt.figure(figsize=(8,4))

    plt.title(f"MNIST Training Classification Performance",fontsize=16)
    plt.xlabel("Epoch",fontsize=14)
    plt.ylabel("Accuracy",fontsize=14)

    for i,exp in enumerate(experiments):
        if 'full' in exp or 'large' in exp:
            print("FULL")
            digits = 10
            samples = 50
        else:
            digits = 3
            samples = 10

        if record == 'old':
            df = pd.read_csv(
                        f'results\MNIST\{exp}\learning_logger.csv',
                        names=['sample','digit','spikes','error','prediction','time','init_time','run_time','offsets']
                        )
            by_run, digs = by_run_performance(df,'winner',digits,samples,False)
            print(exp,' -- ', np.max(np.ceil(np.array(by_run)*(digits*samples))))
            x = np.arange(0,len(by_run),1) #+i*50
            plt.plot(x,by_run, linewidth = 4, label=exp)
            plt.ylim(0,1)
        else:

            df = pd.read_csv(
                        f'results\MNIST\{exp}\performance_log.csv',
                        names=['all','digits']
                        )
            by_run = np.array(df["all"])
            # print(by_run)
            plt.plot(by_run, linewidth = 4, label=exp)
            print(f'{exp} -> {len(by_run)} runs with max {max(by_run)}')

            # plt.plot(np.transpose(digs)[:until], '--', label=['0','1','2'])
            plt.ylim(0,100)
    # plt.ylim(0,1)
    plt.legend()
    plt.show()


experiments = [
    # 'julia_inhibit_solver',
    # 'MNIST_inelast',
    # 'MNIST_unbounded',
    # 'MNIST_eta',
    # 'MNIST_deep',
    # 'MNIST_unbounded_prime',
    # 'MNIST_deep_prime',
    # 'learning_decay',
    # 'MNIST_rich',
    # 'prob_update',
    # 'heidelearn',
    # 'MNIST_asymmetic',
    # 'M_binary',
    # 'MNIST_deep_prime',
    # 'MNIST_rich',
    # 'prob_update'
    # 'MNIST_large',
    # 'hebb_large',
    # 'hebb_test',
    # 'MNIST_asymm',
    # 'exin_test'
    # 'fixed_test',
    # 'modern_layers',
    # 'modern_inh_counter',
    # 'layers_heavy',
    # 'norm_test',
    # 'long',
    # 'long_deep'
    # 'simple_long',
    # 'simple_deep',
    # 'unbounded_deep',
    # 'unbounded_fan',
    # 'fanin_1.5',
    # 'fanin_1.5_full',
    # 'fanin_1.75_full',
    # 'fanin_1.75_nodec_full',
    # 'speed_testing_full',
    # 'speed_decay_full',
    # 'speed_bigeta_full',
    # 'speed_target15_full',
    # 'speed_target5_full',
    # 'spread_full',
    # 'target50_maxflux_full'
    "speed_target15_full2",
    'fanin_1.5_full',
    ]

until = 150*10000

# plot_singles(experiments,until)
# plot_all(experiments,until,record='old')
# 

experiments = {
    # "tiling_full",
    # "speed_target15_full3",
    # "tiling_deep_full",
    # "speed_target15_full_fan1",
    'thresh_full',
    # 'thresh_slow_full',
    # 'thresh_0.5_full',
    # 'thresh_0.5_noref_full',
    # 'thresh_0.5_noref_long_full',
    # 'long_slow_full',
    # # 'fanin_prime',

    # 'updates_inverse',
    # 'updates_fresh',
    # 'updates_coeff',
    'updates_cobuff',
    # 'updates_nocoeff',
    # 'updates_hicoeff_oldfan',
    # 'updates_half'
    'updates_cobuff_alternode_inh',
    'thresh_full_rerun',
    'steady_simple',
}


# experiments = []
# import os
# path = 'results/MNIST'
# if os.path.exists(path) == True:
#     dir_list = os.listdir(path)

#     for directory in dir_list:
#         try:
#             print(directory)
#             if 'arbor_sweep' in directory and os.path.isfile(path+'/'+directory+'/performance_log.csv'):
#                 experiments.append(directory)
#         except:
#             print(f"Dir {directory} does not exist.")

# plot_singles(experiments,until,10,record='new')
# plot_singles(experiments,until,10,record='new')
plot_all(experiments,until,record='new')

def res_rasters(path):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import os
    if os.path.exists(path) == True:
        dir_list = os.listdir(path)
        count = 0
        for directory in dir_list:
            try:
                # print(directory)
                exp_path = f"{path}{directory}"

                acc = picklin(exp_path,"performance")
                # print(acc)
                thresh = 61
                if np.max(acc)>thresh:
                    print(f"Above {thresh}: ",directory)
                    plt.figure(figsize=(16, 11))
                    img = mpimg.imread(f"{exp_path}/raster_all.png")
                    plt.imshow(img)
                    plt.show()

            except:
                print(f"Dir {directory} has no raster plot.")


def res_performances(path):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import os
    print("here")
    plt.figure(figsize=(16,6))
    # plt.tight_layout(rect=[0,0,.5,-.5]) 
    # plt.subplots_adjust(right=.7)
    plt.subplots_adjust(right=.7)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if os.path.exists(path) == True:
        dir_list = os.listdir(path)
        count = 0
        above = 0
        for directory in dir_list:
            try:
                exp_path = f"{path}{directory}"
                acc = picklin(exp_path,"performance")

                if np.max(acc)>60:
                    print(acc,np.max(acc),directory)
                    plt.plot(acc[:25],linewidth=3,label=exp_path[len("results/res_MNIST/res_"):])#,color=colors[above%len(colors)])
                    above+=1
                else:
                    plt.plot(acc[:25],'--',alpha=0.1)
            except:
                print(f"Dir {directory} has no performance measures yet.")

            count+=1
        plt.legend(
            title='eta_tau_c_tau_n_sth_n_sth_c_fans_n_fans_c_den_conn',
            loc='upper left', bbox_to_anchor=(1.04,1)
            )
        plt.xlabel("Epochs"  ,fontsize=18)
        plt.ylabel("Accuracy",fontsize=18)
        plt.title("MNIST Performance of Different Reservoir Configurations",fontsize=22)
        plt.show()

path = 'results/res_MNIST/'
# res_performances(path)
# res_rasters(path)

def evolve(path):
    import os
    if os.path.exists(path) == True:
        dir_list = os.listdir(path)
        accs = {}
        for directory in dir_list:
            try:
                exp_path = f"{path}{directory}"
                acc = picklin(exp_path,"performance")
                accs[directory] = np.max(acc)
            except:
                print(f"Dir {directory} has not been run.")
    print("\n")
    perf_rankings = dict(
        reversed(list({k: v for k, v in sorted(accs.items(), key=lambda item: item[1])}.items()))
        )
    for i,(k,v) in enumerate(perf_rankings.items()):
        if i < len(perf_rankings)*.05:
            print(k,v)
            exp_path = f"{path}{k}"
            if not os.path.isfile(f"{path}{k}/evolved/acc.pickle"):
                print(f"{k} has not undergone an evolution.")
                try:
                    res_spikes = picklin(exp_path,"res_spikes")
                except:
                    print(f"Dir {k} is not ready to evolve.")

                try:
                    with open(f"{path}{k}/config.txt") as f:
                        config = dict(eval(f.read()))
                except:
                    print(f"Dir {k} has no config file.")
                print(f"Running {k}")
                break
    config['path'] = f"results/res_MNIST/{k}/evolved"
    config['runs'] = 100
    config['res_spikes'] = res_spikes
    print(config)
    return config

path = 'results/res_MNIST/'
# config = evolve(path)