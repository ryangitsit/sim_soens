ECHO OFF

FOR /L %%i IN (1,1,100000) DO (
    python exp_MNIST_full.py --run %%i --jul_threading 4 --digits 3 --samples 10 --name MNIST_asymmetric --low_bound 0 --eta 0.001 --decay True
)

PAUSE
