import numpy as np

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

def tiles_to_spikes(tiles):
    import brian2
    indices = []
    times = []
    scarf = [[] for i in range(36)]
    for i,tile in enumerate(tiles):
        unraveled = np.concatenate(tile)
        # print(i,unraveled)
        P = brian2.PoissonGroup(len(unraveled), rates=unraveled*brian2.Hz/10)
        MP = brian2.SpikeMonitor(P)
        net = brian2.Network(P, MP)
        net.run(1000*brian2.ms)
        spikes_i = np.array(MP.i[:])
        spikes_t = np.array(MP.t[:])*1000+i*1000
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

