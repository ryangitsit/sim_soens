
from super_functions import *
from sklearn.linear_model import LogisticRegression
import os


def main():

    directory = 'results/reservoirs/'
    
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            net = picklin(directory,f[len(directory):-7])

            N = 72
            T = 3601
            classes = 3
            examples_per_class = 3
            samples = classes*examples_per_class
            window = 360   
            labels = [0,0,0,1,1,1,2,2,2]

            spikes = net.net.spikes
            mat = spks_to_binmatrix(N,T,spikes)

            model = LogisticRegression()
            correct = 0
            X = []
            y = []
            for i in range(samples):
                if  i%3 != 2:
                    x = np.concatenate(mat[:,i*window:i*window+window]).reshape(1, -1)[0]
                    # print(x)
                    X.append(x)
                    y.append(labels[i])
            model.fit(X,y)

            X_test = []
            y_test = []
            for i in range(samples):
                if i%3 == 2:
                    x = np.concatenate(mat[:,i*window:i*window+window]).reshape(1, -1)[0]
                    X_test.append(x)

            predictions=model.predict(X_test)
            
            if np.array_equal(predictions, [0,1,2]):
                print(predictions, " --> Classified!")
            else:
                print(predictions)

if __name__=='__main__':
    main()


