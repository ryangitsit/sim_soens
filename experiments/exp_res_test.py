import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../')
from sim_soens.super_net import SuperNet
from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.soen_components import network
from sim_soens.super_functions import *
from sim_soens.soen_plotting import raster_plot, activity_plot

# Saccade MNIST dataset
input = SuperInput(channels=36,type="saccade_MNIST",tile_time=50)
# raster_plot(input.spike_arrays)

# Random network of 72 neurons
from sim_soens.super_net import PointReservoir
params= {
    "N":72,
    "s_th":0.5,
    "beta":2*np.pi*10**2,
    "tau":100,
    "tau_ref":50,
    "tf":3600*5,
    "run":1,
    "laps":10,
    }

res = PointReservoir(**params)
# res.graph_net()

# Wire up input
res.connect_input(input)
# res.graph_input()

# Run the network and plot activity
s = time.perf_counter()
res.run_network()
f = time.perf_counter()
print("Runtime: ",(f-s)/60)
# raster_plot(res.net.spikes)

# Train and test linear classifier!
from sim_soens.super_functions import *
from sklearn.linear_model import LogisticRegression
spikes = res.net.spikes
# spikes = input.spike_arrays

N = 72
T = 3601*5
classes = 3
examples_per_class = 3
samples = classes*examples_per_class
window = 360*5
labels = [0,0,0,1,1,1,2,2,2]

# spikes = net.net.spikes
mat = spks_to_binmatrix(N,T,spikes)
# raster_plot(spikes)
model = LogisticRegression(max_iter=100000)
X = []
y = []
X_f = []
y_f = []
for i in range(samples):
    if  i%3 != 2:
        section = mat[:,i*window:i*window+window]
        x = np.concatenate(section).reshape(1, -1)[0]
        X.append(x)
        y.append(labels[i])


model.fit(X,y)

X_test = []
y_test = []
for i in range(samples):
    if i%3 == 2:
        section = mat[:,i*window:i*window+window]
        x = np.concatenate(section).reshape(1, -1)[0]
        X_test.append(x)


predictions=model.predict(X_test)

if np.array_equal(predictions, [0,1,2]):
    print(predictions, " --> Classified!")
    raster_plot(spikes)
else:
    print(predictions)