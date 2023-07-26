import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_functions import *


def pixel_hist(backend,folder,under,version):
    names=[
            "regime",
            "validation",
            "ib",
            "tau",
            "beta",
            "s_th",
            "eta",
            "convergence",
            # "run"
            ]
    if version == 'new': names.append("run")
    df = pd.read_csv(
        f'results/{folder}/pixels.csv',
        names=names
            )
    print(f"Trials run: {len(df[names[0]])}")
    
    # print(len(df["regime"]))

    converge_dict = {
        "regime_Elastic":0,
        "regime_Inelastic":0,
        "regime_Unbounded":0,
        "validation_Intermittent":0,
        "validation_Update":0,
        "ib_1.8":0,
        "ib_2.0":0,
        "tau_50.0":0,
        "tau_150.0":0,
        "beta_2.0":0,
        "beta_3.0":0,
        "s_th_0.25":0,
        "s_th_0.5":0,
        "s_th_0.75":0,
        "eta_0.01":0,
        "eta_0.015":0,
        "eta_0.02":0,
        "convergence":0,
    }

    convs = []
    total = 0
    for index, row in df.iterrows():
        if df["convergence"][index] < under and df["ib"][index]==1.8 and df["validation"][index]=="Update": # and index < 70:
            end = 1
            if version == "new": end = 2
            for n in names[:-end]: 
                converge_dict[f"{n}_"+str(df[n][index])] += 1
            total += 1
            convs.append(row["convergence"])

    print(f"{total} total number of configurations converged under {under} with average of {np.mean(convs)}.")
    # print(converge_dict)
    # print(converge_dict)
    keys = list(converge_dict.keys())
    keys[0] = keys[0][7:]
    keys[1] = keys[1][7:]
    keys[2] = keys[2][7:]

    keys[3] = keys[3][11:]
    keys[4] = keys[4][11:]

    plt.style.use('seaborn-v0_8-muted')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(12,6))
    plt.bar(
        keys[:-1], 
        list(converge_dict.values())[:-1], 
        color = [
            colors[0],colors[0],colors[0],
            colors[1],colors[1],
            colors[2],colors[2],
            colors[3],colors[3],
            colors[4],colors[4],
            colors[5],colors[5],colors[5],
            colors[6%6],colors[6%6],colors[6%6],
            ]
        )
    plt.title(f"Parameter Occurences with Sub-{int(np.floor(under/30))} Epoch Convergence",fontsize=20)
    plt.xlabel("Parameter",fontsize=18)
    plt.ylabel("Count",fontsize=18)
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()

backend = 'jul'

# folder = 'pixels_random'
# version = 'new'

folder = 'jul_pixels_inh_prime'
version = 'old'

# folder = 'jul_testing'
under = 292

pixel_hist(backend,folder,under,version)