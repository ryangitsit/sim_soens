ECHO OFF

FOR /L %%i IN (101,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 3 --samples 10 --name MNIST_deep_prime --eta 0.0005 --layers 4
)

PAUSE
