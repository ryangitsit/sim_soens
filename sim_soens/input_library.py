from sim_soens.super_input import SuperInput


def make_dataset(digits,samples,slowdown,duration):
    '''
    Creates rate coded spiking MNIST dataset
        - digits   = number of classes (different handwritten digits)
        - samples  = number of examples from each class
        - slowdown = factor by which to reduce rate encoding
        - duration = how long each sample should be (nanoseconds)
    '''
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

def make_audio_dataset(patterns,replicas):
    '''
    For Heidelberg dataset only
    '''
    import tables
    file_path = f"datasets/Heidelberg/shd_train/shd_train.h5"
    fileh = tables.open_file(file_path, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels
    channels = np.max(np.concatenate(units))+1
    length = np.max(np.concatenate(times))
    classes = np.max(labels)

    names = ['ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN',
                'EIGHT','NINE','TEN','NULL','EINS','ZWEI','DREI','VIER',
                'FUNF','SECHS','SEBEN','ACHT','NEUN','ZEHN']
    dataset = [[] for _ in range(patterns)]
    for i in range(patterns):
        result = np.where(np.array(labels)==i)[0]
        for j in range(replicas):
            dataset[i].append([units[result[j]],times[result[j]]*1000])

    picklit(
        dataset,
        f"datasets/Heidelberg/",
        f"digits={patterns}_samples={replicas}"
        )
    return dataset