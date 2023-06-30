ECHO OFF

FOR /L %%i IN (4,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 3 --samples 25 --name MNIST_full
)

PAUSE
