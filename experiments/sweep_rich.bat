ECHO OFF

FOR /L %%i IN (101,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 3 --samples 10 --name MNIST_rich --eta 0.0001 --layers 5 --elasticity inelastic
)

PAUSE
