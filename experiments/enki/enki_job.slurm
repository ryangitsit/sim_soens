#!/bin/bash
#SBATCH --job-name=MNIST_sweep
#SBATCH --nodes=1 # Number of nodes (how many compute boxes)
#SBATCH --ntasks=1 # Number of MPI ranks (nodes*ntasks-per-node)
#SBATCH --ntasks-per-node=1 # How many tasks on each box (which has two socket)
#SBATCH --ntasks-per-socket=5 # How many tasks on each socket
#SBATCH --ntasks-per-core=1 # How many tasks on each core
#SBATCH --cpus-per-task=4 # Number of CPUs per MPI rank 
#SBATCH --gres=gpu:1 # Number of GPUs per
#SBATCH --mem-per-cpu=2G\\ # Memory per processor
#SBATCH --time=160:00:00 # Time limit hrs:min:sec
#SBATCH --array=1 # job array with index values 1, 2, 3
#SBATCH --partition=gpu # Selected partition
#SBATCH --mail-type=begin # send email when job begins
#SBATCH --mail-type=end # send email when job ends
#SBATCH --mail-user=rmo2@nist.gov

module purge 
module load python/3.10.9/anaconda
module load julia/1.9.0
module load cuda
conda activate /home/rmo2/envs/testenv

python-jl exp_MNIST_full.py --name test_series2 --digits 3 --run 5 --samples 10 --low_bound 0 --eta 0.0005 --decay True