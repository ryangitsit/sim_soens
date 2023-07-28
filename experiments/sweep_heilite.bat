ECHO OFF

FOR /L %%i IN (1,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 3 --samples 1 --name heilite --eta 0.0005 --elasticity inelastic --dataset Heidelberg --duration 700
)

PAUSE   