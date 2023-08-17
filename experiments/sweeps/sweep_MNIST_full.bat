ECHO OFF

FOR /L %%i IN (1,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 10 --samples 100 --name MNIST_full --elasticity inelastic --eta 0.001
)

PAUSE
