ECHO OFF

FOR /L %%i IN (4,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 4 --samples 5 --name MNIST_full
)

PAUSE
