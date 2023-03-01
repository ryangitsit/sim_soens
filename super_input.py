#%%
import numpy as np
from soen_sim import input_signal
from super_functions import *

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

        elif self.type == "random":
            print("Generating random input...")
            self.spike_arrays = [np.random.randint(self.channels,size=self.total_spikes),np.random.rand(self.total_spikes)*self.duration]
            self.spike_rows = self.array_to_rows(self.spike_arrays)

        elif self.type == "defined":
            # print("Generating pre-defined input...")
            self.spike_arrays = self.defined_spikes
            self.spike_rows = self.array_to_rows(self.spike_arrays)

        elif self.type == "saccade_MNIST":
            self.spike_arrays = self.saccade_MNIST()
            self.spike_rows = self.array_to_rows(self.spike_arrays)

        elif self.type == "constant":
            self.constant()

        else:
            print("Please provide valid input type")
            pass
        
        if self.type != 'constant':
            self.signals = []
            for i in range(self.channels):
                array = np.sort(self.spike_rows[i])
                if array.any():
                    array = np.append(array,np.max(array)+.001)
                self.signals.append(input_signal(name = 'input_synaptic_drive', 
                                    input_temporal_form = 'arbitrary_spike_train', 
                                    spike_times = array) )
                # print(self.spike_rows[i])


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


    def constant(self):
        input = input_signal(name = 'constant_input', 
                             input_temporal_form = 'constant', 
                             applied_flux = self.phi_app)
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
        print(y_train[self.index])
        P = brian2.PoissonGroup(channels, rates=(X/self.slow_down)*brian2.Hz)
        MP = brian2.SpikeMonitor(P)
        net = brian2.Network(P, MP)
        net.run(self.duration*brian2.ms)
        spikes_i = np.array(MP.i[:])
        spikes_t = np.array(MP.t[:])
        self.indices = spikes_i
        self.times = spikes_t*1000

        return self.indices, self.times

    def saccade_MNIST(self):
        import matplotlib.pyplot as plt
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = X_train[(y_train == 0) | (y_train == 1) | (y_train == 2)]
        y = y_train[(y_train == 0) | (y_train == 1) | (y_train == 2)]
        
        dataset = [[] for i in range(3)]
        stream = [[],[]]
        count = 0
        for i in range(20):
            if len(dataset[y[i]]) < 3:
                # print(y[i])
                dataset[y[i]].append(aug_digit(X[i]))
                # X_aug = aug_digit(X[i])
                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(5, 5,figsize=(8,8))
                # tiles = []
                # for i in range(5):
                #     for j in range(5):
                #         x1=i*6
                #         x2=i*6+6
                #         y1=j*6
                #         y2=j*6+6
                #         print(x1,x2,y1,y2)
                #         img = X_aug[x1:x2,y1:y2]

                #         tiles.append(img)
                #         # fig = plt.figure
                #         axs[i,j].imshow(img, cmap='gray')
                #         axs[i,j].set_xticks([])
                #         axs[i,j].set_yticks([])
                #         # axs[i,j].set_title(f'{x1},{x2}-{y1},{y2}')
                # # for ax in axs:
                # #     ax.set_xticks([])
                # #     ax.set_yticks([])
                # plt.show()


        for data in dataset:
            for sample in data:
                tiles = tile_img(sample)
                tile_spikes = tiles_to_spikes(tiles,self.tile_time)
                stream[0].extend(tile_spikes[0])
                stream[1].extend(np.array(tile_spikes[1])+(self.tile_time*36)*count)
                # print(tile_spikes[1])
                # print(self.tile_time*36,count, self.tile_time*36*count)
                count+=1
        # print(len(dataset))
        # print(len(dataset[0]),len(dataset[1]),len(dataset[2]))

        # print(len(stream))
        # print(len(stream[0]))
        # from soen_plotting import raster_plot
        # raster_plot(stream)
        return stream