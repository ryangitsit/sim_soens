# Train and test linear classifier!
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_functions import *
from sklearn.linear_model import LogisticRegression


# x = np.random.rand(4,3)
# print(x)
# print(np.sum(x,axis=1)) 

dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
# print(len(dataset))
# print(len(dataset[0]))

digits = 3
samples = 10
T = 250
N = 784

def bin_to(bins):
    # return np.concatenate(bins)
    return np.sum(bins,axis=1)

mats = []
labels = []
for digit in range(digits):
    for sample in range(samples):
        # print(dataset[digit][sample])
        spikes = dataset[digit][sample]
        mats.append(bin_to(spks_to_binmatrix(N,T,spikes)))
        labels.append(digit)

# print(labels)

model = LogisticRegression(max_iter=100)
model.fit(mats,labels)

test = []
test_labels = []
for digit in range(digits):
    for sample in range(10):
        # print(dataset[digit][sample])
        spikes = dataset[digit][sample+samples]
        test.append(bin_to(spks_to_binmatrix(N,T,spikes)))
        test_labels.append(digit)

correct = 0
total = 0
predictions = model.predict(test)
for i,pred in enumerate(predictions):
    lab = test_labels[i]
    # print(lab,' --> ',pred)
    total += 1
    if lab==pred: correct +=1

print(f"Test accuracy = {np.round(100*correct/total,2)}%")


# for i,m in enumerate(mats):
#     model.fit(mats[i],labels[i])

# spikes = input.spike_arrays

# N = 72
# T = 3601*5
# classes = 3
# examples_per_class = 3
# samples = classes*examples_per_class
# window = 360*5
# labels = [0,0,0,1,1,1,2,2,2]

# # spikes = net.net.spikes
# mat = spks_to_binmatrix(N,T,spikes)
# # raster_plot(spikes)
# model = LogisticRegression(max_iter=100000)
# X = []
# y = []
# X_f = []
# y_f = []
# for i in range(samples):
#     if  i%3 != 2:
#         section = mat[:,i*window:i*window+window]
#         x = np.concatenate(section).reshape(1, -1)[0]
#         X.append(x)
#         y.append(labels[i])


# model.fit(X,y)

# X_test = []
# y_test = []
# for i in range(samples):
#     if i%3 == 2:
#         section = mat[:,i*window:i*window+window]
#         x = np.concatenate(section).reshape(1, -1)[0]
#         X_test.append(x)


# predictions=model.predict(X_test)

# if np.array_equal(predictions, [0,1,2]):
#     print(predictions, " --> Classified!")
#     # raster_plot(spikes)
# else:
#     print(predictions)