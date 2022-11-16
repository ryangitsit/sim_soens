#%%
import numpy as np
from soen_sim import input_signal

class SuperInput():
    def __init__(self,**entries):
        self.type ="MNIST", # random for rand gen
        self.duration = 100,
        self.channels = 28*28,
        self.slow_down = 10,
        self.total_spikes = 25,
        self.name = 'Super_Input'
        self.__dict__.update(entries)
        
        if self.type == "MNIST":
            self.channels = int(28*28)
            print("Generating MNNIST dataset...")
            mnist_indices, mnist_spikes = self.MNIST()
            self.spike_arrays = [mnist_indices,mnist_spikes]
            self.spike_rows = self.array_to_rows(self.spike_arrays)

        if self.type == "random":
            print("Generating random input...")
            self.spike_arrays = [np.random.randint(self.channels,size=self.total_spikes),np.random.rand(self.total_spikes)*self.duration]
            self.spike_rows = self.array_to_rows(self.spike_arrays)

        if self.type == "defined":
            print("Generating pre-defined input...")
            self.spike_arrays = self.defined_spikes
            self.spike_rows = self.array_to_rows(self.spike_arrays)
        
        self.signals = []
        for i in range(self.channels):
            self.signals.append(input_signal(name = 'input_synaptic_drive', 
                       input_temporal_form = 'arbitrary_spike_train', 
                       spike_times = self.spike_rows[i]))

    def gen_rand_input(self,spiking_indices,max_amounts):
        # self.input = np.random.randint(self.duration, size=())
        input = [ [] for _ in range(self.channels) ]
        spikers = np.random.randint(self.channels,size=spiking_indices)
        sum = []
        for n in spikers:
            input[n] = np.sort(np.random.randint(self.duration,size=np.random.randint(max_amounts)))
            sum.append(len(input[n]))
        print("Total number of spikes:", np.sum(sum))
        print("Spiking at neurons: ", spikers)
        return input


    def gen_ordered_input(self):
        pass

    def load_input(self):
        pass

    def rows_to_array(self,input):
        spikes = [ [] for _ in range(self.channels) ]
        count = 0
        for n in range(self.channels):
            if np.any(input[n]):
                # print(n)
                spikes[0].append(np.ones(len(input[n]))*n)
                spikes[1].append(input[n])
            count+=1
        spikes[0] =np.concatenate(spikes[0])
        spikes[1] = np.concatenate(spikes[1])
        return spikes

    def array_to_rows(self,array):
        rows = [ [] for _ in range(self.channels) ]
        for i in range(len(array[0])):
            rows[array[0][i]].append(array[1][i])
        return rows

    def MNIST(self):
        import brian2
        from keras.datasets import mnist
        print("load")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print("loaded")
        # simplified classification (0 1 and 8)
        X_train = X_train[(y_train == 1) | (y_train == 0) | (y_train == 2)]

        # pixel intensity to Hz (255 becoms ~63Hz)
        X_train = X_train / 4 
        numbers = [0,1,2]
        classes = ["zero", "one", "two"]

        # Generate spiking data
        brian2.start_scope()
        self.units = []
        self.times = []
        self.data = {}
        channels = 28*28
        X = X_train[self.index].reshape(channels)
        P = brian2.PoissonGroup(channels, rates=(X/self.slow_down)*brian2.Hz)
        MP = brian2.SpikeMonitor(P)
        net = brian2.Network(P, MP)
        net.run(self.duration*brian2.ms)
        spikes_i = np.array(MP.i[:])
        spikes_t = np.array(MP.t[:])
        self.indices = spikes_i
        self.times = spikes_t*1000

        return self.indices, self.times

# %%
