import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_functions import *
from sklearn.linear_model import LogisticRegression
import os
from sim_soens.soen_plotting import raster_plot

def main():

    directory = '../results/reservoirs_3/'
    count = 0
    correct = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            # net = picklin(directory,f[len(directory):-7])
            spikes = txt_to_spks(f)

            N = 72
            T = 3601
            classes = 3
            examples_per_class = 3
            samples = classes*examples_per_class
            window = 360   
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
            count +=1
    print(f"{correct} correct out of {count} --> {correct/count}% of configurations")

if __name__=='__main__':
    main()


