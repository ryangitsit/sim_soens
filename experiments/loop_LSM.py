
from src.super_functions import *
from sklearn.linear_model import LogisticRegression
import os
from soen_plotting import raster_plot

def main():

    directory = 'results/reservoirs_3/'
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
            model_fine = LogisticRegression(max_iter=100000)
            X = []
            y = []
            X_f = []
            y_f = []
            for i in range(samples):
                if  i%3 != 2:
                    section = mat[:,i*window:i*window+window]
                    x = np.concatenate(section).reshape(1, -1)[0]
                    # print(x)
                    X.append(x)
                    y.append(labels[i])
                    # print(len(section[:,330:331]))
                    for t in range(360):
                        slice = section[:,t].reshape(1, -1)[0]
                        # print(slice)
                        X_f.append(slice)
                        y_f.append(labels[i])
                        # print(len(t_slice.reshape(-1, 1)))
            model_fine.fit(X_f,y_f)
            # print(X_f[0])
            # print(len(y_f))

            model.fit(X,y)

            X_test = []
            y_test = []
            pred_fine = []
            for i in range(samples):
                if i%3 == 2:
                    section = mat[:,i*window:i*window+window]
                    x = np.concatenate(section).reshape(1, -1)[0]
                    X_test.append(x)
                    pred_slice = []
                    for t in range(360):
                        slice = section[:,t].reshape(1, -1)
                        # print(slice)
                        fit = model_fine.predict(slice)
                        # print(fit)
                        pred_slice.append(fit)
                        # print(np.concatenate(pred_slice))
                    pred_fine.append(np.bincount(np.concatenate(pred_slice)).argmax())


            predictions=model.predict(X_test)
            predictions_fine = pred_fine
            
            if np.array_equal(predictions, [0,1,2]) or np.array_equal(predictions_fine, [0,1,2]):
                print(predictions, predictions_fine, " --> Classified! --> ", f)
                correct+=1
                raster_plot(spikes)
            else:
                print(predictions, predictions_fine)
            count +=1
    print(f"{correct} correct out of {count} --> {correct/count}% of configurations")

if __name__=='__main__':
    main()


