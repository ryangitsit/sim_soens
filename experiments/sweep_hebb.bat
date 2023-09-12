ECHO OFF

FOR /L %%i IN (168,1,100000) DO (
    python exp_MNIST_full.py --run %%i --jul_threading 4 --digits 3 --samples 10 --name hebb_test --low_bound 0 --eta 0.001 --hebbian True
)

PAUSE
