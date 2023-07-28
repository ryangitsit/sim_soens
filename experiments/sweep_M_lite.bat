ECHO OFF

FOR /L %%i IN (1,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 3 --samples 1 --name M_lite --eta 0.0005 --elasticity inelastic --dataset MNIST
)

PAUSE   