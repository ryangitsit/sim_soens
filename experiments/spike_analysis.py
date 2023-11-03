# Train and test linear classifier!
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.super_functions import *
from sklearn.linear_model import LogisticRegression


'''
This file is for analysing the quality of spiking input and is not soens-specific
'''
def main():
    def make_dataset(digits,samples,slowdown,duration):
        '''
        Creates rate coded spiking MNIST dataset
            - digits   = number of classes (different handwritten digits)
            - samples  = number of examples from each class
            - slowdown = factor by which to reduce rate encoding
            - duration = how long each sample should be (nanoseconds)
        '''
        import matplotlib.pyplot as plt
        from .sim_soens import SuperInput
        dataset = [[] for _ in range(digits)]
        fig, axs = plt.subplots(digits,samples,figsize=(36,12))
        for i in range(digits):
            for j in range(samples):
                input_MNIST = SuperInput(
                    type='MNIST',
                    index=i,
                    sample=j,
                    slow_down=slowdown,
                    duration=duration
                    )
                spikes = input_MNIST.spike_arrays
                dataset[i].append([spikes[0],spikes[1]])

                axs[i][j].plot(spikes[1],spikes[0],'.k',ms=.5)
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
        # saves dataset
        picklit(
            dataset,
            "datasets/MNIST/",
            f"duration={duration}_slowdown={slowdown}"
            )
        # plots dataset
        plt.show()

    # make_dataset()

    dataset = picklin("datasets/MNIST/","duration=5000_slowdown=100")
    # print(len(dataset))
    # print(len(dataset[0]))

    digits = 10
    samples = 50
    T = 1000
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

    model = LogisticRegression(max_iter=10000)
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


if __name__=='__main__':
    main()
