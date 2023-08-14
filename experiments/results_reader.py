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


def by_run_performance(df,decider,digits,samples):
    by_run = []
    by_dig_runs = [[] for _ in range(digits)]

    zrs = [0 for _ in range(digits)]

    dig_runs = zrs
    
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
        for index, row in df.iterrows():
            # print(df["sample"].astype(int)[1])
            # if df["sample"][(df["sample"]< samples)]:
            if df["sample"].astype(int)[1] == 0:#< samples:
                if no_ties(df,index) == True:
                    run_wins+=1
                    dig_runs[df["digit"][index]]+=1
                if (index+1)%(digits*samples) == 0:
                    by_run.append(run_wins/(digits*samples))
                    for i,dig in enumerate(dig_runs):
                        by_dig_runs[i].append(dig/samples)
                    dig_runs = [0 for _ in range(digits)]
                    run_wins = 0

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

def plot_singles(experiments,until):
    for i,exp in enumerate(experiments):
        df = pd.read_csv(
            f'results\MNIST\{exp}\learning_logger.csv',
            names=['sample','digit','spikes','error','prediction','time','init_time','run_time','offsets']
            )

        # percents, procents = ongoing_performance(df)
        by_run, digs = by_run_performance(df,'winner',3,1)

        pr = np.max(np.ceil(np.array(by_run)*30))
        print(f"Experiment {exp}, {len(by_run)} epochs, {np.round(pr*100/30,2)}% best run")


        plt.style.use('seaborn-v0_8-muted')
        plt.figure(figsize=(8,4))

        plt.title(f"MNIST Training Classification Performance - {exp}",fontsize=16)
        plt.xlabel("Epoch",fontsize=14)
        plt.ylabel("Accuracy",fontsize=14)
        plt.plot(by_run[:until], linewidth = 1, label="Total")
        # plt.plot(np.transpose(digs)[:until], '--', label=['0','1','2'])
        plt.ylim(0,1.025)

        plt.legend()
        plt.show()


        # plt.plot(df["run_time"])
        # plt.show()

        # print("Average runtime = ",np.mean(df["run_time"]))

def plot_all(experiments,until):
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(8,4))

    plt.title(f"MNIST Training Classification Performance",fontsize=16)
    plt.xlabel("Epoch",fontsize=14)
    plt.ylabel("Accuracy",fontsize=14)

    for i,exp in enumerate(experiments):
        df = pd.read_csv(
            f'results\MNIST\{exp}\learning_logger.csv',
            names=['sample','digit','spikes','error','prediction','time','init_time','run_time','offsets']
            )

        by_run, digs = by_run_performance(df,'winner',3,10)
        print(exp,' -- ', np.max(np.ceil(np.array(by_run)*30)))
        plt.plot(by_run, linewidth = 4, label=exp)
        plt.plot(np.transpose(digs)[:until], '--', label=['0','1','2'])
    plt.ylim(0,1)
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
    'MNIST_asymmetic',
    # 'M_binary',
    'MNIST_deep_prime',
    # 'MNIST_rich',
    # 'prob_update'
    ]

until = 150*10000

# plot_singles(experiments,until)
plot_all(experiments,until)