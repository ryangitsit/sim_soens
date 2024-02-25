import numpy as np

def dict_size(dct):
    from pympler import asizeof
    print("Surface Size: ", asizeof.asizeof(dct),"\n")
    print("Item sizes:")
    tot = 0
    for i,(k,v) in enumerate(dct.items()):
        size = asizeof.asizeof(v)
        print(f"  {k}"," "*(50-len(k)),f" -- {size}")
        tot+=size
        # print(tot)
    print("\nTotal Size: ",tot,"\n")

def array_to_rows(arrays,channels):
    rows = [ [] for _ in range(channels) ]
    for i in range(len(arrays[0])):
        rows[int(arrays[0][i])].append(arrays[1][i])
    return rows

def rows_to_array(rows):
    spikes = [ [] for _ in range(len(rows)) ]
    count = 0
    for n in range(len(rows)):
        if np.any(rows[n]):
            # print(n)
            spikes[0].append(np.ones(len(rows[n]))*n)
            spikes[1].append(rows[n])
        count+=1
    spikes[0] =np.concatenate(spikes[0])
    spikes[1] = np.concatenate(spikes[1])
    return spikes


def aug_digit(digit):
    X_aug = digit
    X_aug = np.append(X_aug,np.zeros([len(X_aug),2]),1)
    X_aug = np.append(X_aug,[np.zeros((30))],axis=0)
    X_aug = np.append(X_aug,[np.zeros((30))],axis=0)
    return X_aug

def tiles_to_spikes(tiles,tile_time):
    import brian2
    np.random.seed(10)
    indices = []
    times = []
    scarf = [[] for i in range(36)]
    for i,tile in enumerate(tiles):
        unraveled = np.concatenate(tile)
        # print(i,unraveled)
        P = brian2.PoissonGroup(len(unraveled), rates=unraveled*brian2.Hz)
        MP = brian2.SpikeMonitor(P)
        net = brian2.Network(P, MP)
        net.run(tile_time*brian2.ms)
        spikes_i = np.array(MP.i[:])
        spikes_t = np.array(MP.t[:])*tile_time+i*tile_time
        indices.extend(spikes_i)
        times.extend(spikes_t)
        spikes = [indices,times]
    return spikes

def tile_img(digit):
    tiles = []
    for i in range(5):
        for j in range(5):
            x1=i*6
            x2=i*6+6
            y1=j*6
            y2=j*6+6
            img = digit[x1:x2,y1:y2]
            tiles.append(img)
    return tiles


def spks_to_txt(spikes,N,prec,dir,name):
    """
    Convert Brain spikes to txt file
    - Each line is a neuron index
    - Firing times are recorded at at their appropriate neuron row
    """
    import os
    dirName = f"results/{dir}"
    try:
        os.makedirs(dirName)    
    except FileExistsError:
        pass

    indices = spikes[0]
    times = spikes[1]
    with open(f'{dirName}/{name}.txt', 'w') as f:
        for row in range(N):
            for i in range(len(indices)):
                if row == indices[i]:
                    if row == 0:
                        f.write(str(np.round(times[i],prec)))
                        f.write(" ")
                    else:
                        f.write(str(np.round(times[i],prec)))
                        f.write(" ")
            f.write('\n')

def np_save(exp,name,**kwargs):
    import os
    dirName = f"results/{dir}"
    try:
        os.makedirs(dirName)    
    except FileExistsError:
        pass

def txt_to_spks(file):
    """"
    Convert txt file back to Brian style spikes
     - Two parallel arrays of spike times and associated neuron indices
    """
    mat = []
    with open(file) as f:
        for line in f:
            arr = line.split(' ')
            mat.append(np.array(arr))
    indices = []
    times = []
    for i in range(len(mat)):
        row = []
        for j in range(len(mat[i])):
            if mat[i][j] != '\n':
                indices.append(i)
                row.append(float(mat[i][j]))
                times.append(float(mat[i][j]))
    indices = np.array(indices)
    times = np.array(times)
    return [indices,times]

def picklit(obj,path,name):
    import os
    import pickle
    try:
        os.makedirs(path)    
    except FileExistsError:
        pass
    pick = f'{path}/{name}.pickle'
    filehandler = open(pick, 'wb') 
    pickle.dump(obj, filehandler)
    filehandler.close()

def picklin(path,name):
    import os
    import pickle
    file = os.path.join(path, name)
    if '.pickle' in file:
        file = file
    else:
        file = file + '.pickle'
    # print(file)
    file_to_read = open(file, "rb")
    obj = pickle.load(file_to_read)
    file_to_read.close()
    return obj

def save_dict(dict,path,name):
    import os
    import json
    try:
        os.makedirs(path)    
    except FileExistsError:
        pass
    json = json.dumps(dict)
    f = open(path+name+".json","w")
    f.write(json)
    f.close()

def save_fig(plt,path,name):
    import os
    try:
        os.makedirs(path)    
    except FileExistsError:
        pass
    plt.savefig(f"{path}/{name}")

def spks_to_binmatrix(N,T,spikes):
    binned = np.zeros((N,T))
    for i in range(len(spikes[0])):
        if spikes[1][i] < T:
            binned[int(spikes[0][i])][int(np.floor(spikes[1][i]))] += 1
    return binned

def bin_matrix_to_spks(mat):
    indices = []
    times = []
    for i,row in enumerate(mat):
        for t,val in enumerate(row):
            if val!=0:
                indices.append(i)
                times.append(t)
    spikes = [indices,times]
    return spikes

def make_letters(patterns='zvn'):

    # non-noisy nine-pixel letters
    letters = {
        'z': [1,1,0,
              0,1,0,
              0,1,1],

        'v': [1,0,1,
              1,0,1,
              0,1,0],

        'n': [0,1,0,
              1,0,1,
              1,0,1]
    }

    if patterns == 'zvnx+':
        letters.update({
            'x': [1,0,1,
                  0,1,0,
                  1,0,1],
            '+': [0,1,0,
                  1,1,1,
                  0,1,0]
        })

    if patterns == 'all':
        letters.update({
            'x': [1,0,1,
                  0,1,0,
                  1,0,1],
            '+': [0,1,0,
                  1,1,1,
                  0,1,0],
            "|": 
                [0,1,0,
                 0,1,0,
                 0,1,0],
            "|  ": 
                [1,0,0,
                 1,0,0,
                 1,0,0,],
            "  |": 
                [0,0,1,
                 0,0,1,
                 0,0,1,],
            "-": 
                [0,0,0,
                 1,1,1,
                 0,0,0],
            "_": 
                [0,0,0,
                 0,0,0,
                 1,1,1],
            "\\": 
                [1,0,0,
                 0,1,0,
                 0,0,1],
            "/": 
                [0,0,1,
                 0,1,0,
                 1,0,0],
            "[]": 
                [1,1,1,
                 1,1,1,
                 1,1,1],
        })

    return letters

def plot_letters(letters,letter=None):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(letters),figsize=(8,6))
    for  j,(name,pixels) in enumerate(letters.items()):
        arrays = [[] for i in range(3)]
        count = 0
        for col in range(3):
            for row in range(3):
                arrays[col].append(pixels[count])
                count+=1
        pixels = np.array(arrays).reshape(3,3)

        axs[j].set_xticks([])
        axs[j].set_yticks([])
        axs[j].set_title(name,fontsize=14)

        if letter==name:
            axs[j].imshow(
                pixels,
                interpolation='nearest',
                cmap=cm.Oranges
                )
        else:
            axs[j].imshow(
                pixels,
                interpolation='nearest',
                cmap=cm.Blues
                )
    plt.show()

def make_inputs(letters,spike_time):
    from sim_soens.super_input import SuperInput
    # make the input spikes for different letters
    inputs = {}
    for name, pixels in letters.items():
        idx = np.where(np.array(letters[name])==1)[0]
        spike_times = np.ones(len(idx))*spike_time
        defined_spikes=[idx,spike_times]
        inputs[name]=SuperInput(type='defined',defined_spikes=defined_spikes)
    return inputs

def make_spikes(letter,spike_time):
    '''
    Converts a letter-array into spikes = [indices,times]
    '''
    # convert to indices of nine-channel input
    idx = np.where(np.array(letter)==1)[0]

    # all non-zero indices will spike at spike_time
    times = (np.ones(len(idx))*spike_time).astype(int)
    spikes = [idx,times]

    return spikes

def pixels_to_spikes(letter,spike_times):
    '''
    Converts a letter-array into spikes = [indices,times]
    '''
    indices = []
    times   = []

    # convert to indices of nine-channel input
    idx = np.where(np.array(letter)==1)[0]

    # all non-zero indices will spike at spike_times
    for time in spike_times:
        times.append((np.ones(len(idx))*time).astype(int))
        indices.append(idx)
    times = np.concatenate(times)
    indices = np.concatenate(indices)

    assert len(times)==len(indices)
    spikes = [list(indices),list(times)]

    return spikes

    
def timer_func(func): 
    from time import time 
    # This function shows the execution time of  
    # the function object passed 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func 


def tile():
    print("Tiling")
    idx_groups = {}

    for i in range(7):
        for j in range(7):
            idx_groups[f"{i}-{j}"] = []

    count = 0
    modsi = 0
    modsj = 0
    for i in range(28):
        if i%4 == 0 and i != 0 and i != 28:
            modsi+=1
        for j in range(28):
            if j%4 == 0 and j != 0 and j != 28:
                modsj+=1
            idx_groups[f"{modsi}-{modsj}"].append((i,j,count))
            count+=1
        modsj=0

    idx_list = []        
    for tile,pixels in idx_groups.items():
        for (i,j,count) in pixels:
            idx_list.append(count)

    return idx_groups, idx_list