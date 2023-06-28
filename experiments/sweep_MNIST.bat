ECHO OFF

FOR /L %%i IN (1,1,10000) DO (
    python exp_MNIST_prime.py --run %%i
)

PAUSE
