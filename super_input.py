#%%
import numpy as np

class SuperInput():
    def __init__(self,**entries):
        self.name = 'Super_Input'
        self.__dict__.update(entries)

    def gen_rand_input(self,spiking_indices,max_amounts):
        # self.input = np.random.randint(self.sim_in, size=())
        input = [ [] for _ in range(self.channels) ]
        spikers = np.random.randint(self.channels,size=spiking_indices)
        sum = []
        for n in spikers:
            input[n] = np.sort(np.random.randint(self.sim_in,size=np.random.randint(max_amounts)))
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
        import keras
        import brian2
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # simplified classification (0 1 and 8)
        X_train = X_train[(y_train == 1) | (y_train == 0) | (y_train == 2)]
        y_train = y_train[(y_train == 1) | (y_train == 0) | (y_train == 2)]
        X_test = X_test[(y_test == 1) | (y_test == 0) | (y_test == 2)]
        y_test = y_test[(y_test == 1) | (y_test == 0) | (y_test == 2)]

        # pixel intensity to Hz (255 becoms ~63Hz)
        X_train = X_train / 4 
        X_test = X_test / 4
        numbers = [0,1,2]
        classes = ["zero", "one", "two"]
        dataset = {}
        for i,n in enumerate(numbers):
            dataset[classes[i]] = []
            for j,ex in enumerate(y_train):
                if ex == n:
                    dataset[classes[i]].append(j)
                if len(dataset[classes[i]]) == 3:
                    break
        self.dataset = dataset

        # Generate spiking data
        brian2.start_scope()
        self.units = []
        self.times = []
        labels = []
        count = 0
        self.data = {}
        for n,inds in dataset.items():
            self.data[n] = []
            for i in inds:
                channels = 28*28
                X = X_train[i].reshape(channels)
                # print(y_train[i])
                labels.append(y_train[i])
                P = brian2.PoissonGroup(channels, rates=(X/self.slow_down)*brian2.Hz)
                MP = brian2.SpikeMonitor(P)
                net = brian2.Network(P, MP)
                net.run(self.sim_in*brian2.ms)
                spikes_i = np.array(MP.i[:])
                spikes_t = np.array(MP.t[:])
                # indices.append(spikes_i)
                # times.append(spikes_t)

                # print(f"Saving pattern {n}, replica {count}")
                loc = f'{"MNIST"}/inputs'
                item = f'pat{n}_rep{count}'
                self.units.append(spikes_i)
                self.times.append(spikes_t*1000)
                self.data[n].append(count)
                count+=1
                # if todo == "save":
                #     save_spikes(self.channels,self.sim_in,spikes_t*1000,spikes_i,loc,item,self.output_show)
        self.dataset = self.data

        return self.data, self.units, self.times


# input_args = {
#     "channels":28*28,
#     "rand_prob": .3,
#     "sim_in": 500,
#     "channels":28*28,
#     "slow_down": 100,
# }

# super_input = SuperInput(**input_args)
# # #%%
# import matplotlib.pyplot as plt
# def raster_plot(spikes):
#     plt.figure(figsize=(10, 6))
#     plt.plot(spikes[1], spikes[0], '.k')
#     plt.title('Spiking SOEN',fontsize=18)
#     plt.xlabel('Spike Time (ns)',fontsize=16)
#     plt.ylabel('Neuron index',fontsize=16)
#     # plt.xlim(0,800)

# mnist_data, mnist_indices, mnist_spikes = super_input.MNIST()
# for i in range(9):
#     spikes = [mnist_indices[i],mnist_spikes[i]]
#     spikes = super_input.array_to_rows(spikes)
#     print(np.sum(np.sum(spikes)))
#     raster_plot(super_input.rows_to_array(spikes))

# for s in spikes:
#     if np.any(s):
#         print(s)

#%%

# print(len(mnist_indices[1]))
#%%
# input = super_input.gen_rand_input(10,25)
# spikes = super_input.rows_to_array(input)

# print(spikes[0])

#     plt.show()
# raster_plot(spikes)
# # %%
