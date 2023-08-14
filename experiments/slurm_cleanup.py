import os
path = os.getcwd()

if os.path.exists(path) == True:
    file_list = os.listdir(path)
    for file in file_list:
        if 'gpu-stats' in file or 'mpi_test' in file:
            os.remove(file)
            print (f"The file {file} has been removed")