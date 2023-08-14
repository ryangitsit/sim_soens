import os
from csv import writer


exp_config_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

path = "slurm_test"

List = [exp_config_index]

try:
    os.makedirs(path)    
except FileExistsError:
    pass

with open(f'{path}/learning_logger.csv', 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(List)
    f_object.close()