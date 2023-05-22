#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('MNIST_ongoing.csv',names=['sample','digit','spikes','error','prediction'])

# # print(df.to_string())   
# # spikes = np.array(np.array(df["spikes"][11]))
# spike_str = df["spikes"][11][1:-1]
# spikes = np.fromstring(spike_str, dtype=int, sep=',')
# print(spikes)
# for spk in spikes:
#     print(spk)
# digit = int(df["digit"][11])
# print(digit)
# print(spikes[digit])

# sub = spikes - spikes[0] 
# print(sub)
# print(sum(n > 0 for n in sub))


def allow_ties(df,index):
    spike_str = df["spikes"][index][1:-1]
    spikes = np.fromstring(spike_str, dtype=int, sep=',')
    digit = int(df["digit"][index])
    sub = spikes - spikes[digit] 
    if sum(n > 0 for n in sub) == 0:
        return True
    else:
        return False

# print(np.array(df["spikes"][11]) - np.array(df["spikes"])[df["digit"][11]]) 




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
                count+=1
                counts[df["digit"][index]] += 1

    percents.append(count/total)
    procents.append(counts/totals)

by_run = []
runs = 0
run_wins = 0
for index, row in df.iterrows():

    if df["digit"][index] == df["prediction"][index]: run_wins+=1

    if df["sample"][index] == 9 and df["digit"][index] == 2:
        runs += 1
        by_run.append(run_wins/30)
        run_wins = 0



plt.style.use('seaborn-v0_8-muted')
plt.figure(figsize=(8,4))
plt.title("MNIST Classification Performance")
plt.xlabel("Performance Measure Starting Point")
plt.ylabel("Classification Accuracy on Remaining Iterations")
plt.plot(percents, linewidth = 4,label='total')
# plt.plot(procents, label=['0','1','2'])
# plt.plot(by_run)
# plt.ylim(0,1)
plt.legend()
plt.show()
