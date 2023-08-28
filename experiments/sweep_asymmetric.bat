ECHO OFF

FOR /L %%i IN (1895,1,100000) DO (
    python exp_MNIST_full.py --run %%i --jul_threading 4 --digits 3 --samples 10 --name MNIST_asymmetic --low_bound 0 --eta 0.0005 --decay True
)

PAUSE
