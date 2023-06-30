ECHO OFF

FOR /L %%i IN (1,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 3 --samples 10 --name MNIST_inelast --elasticity inelastic
)

PAUSE
