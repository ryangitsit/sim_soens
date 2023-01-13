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