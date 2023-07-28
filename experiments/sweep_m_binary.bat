ECHO OFF

FOR /L %%i IN (1,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 2 --samples 10 --name M_binary --eta 0.0005 --elasticity inelastic --dataset MNIST
)

PAUSE   