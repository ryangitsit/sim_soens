ECHO OFF

FOR /L %%i IN (193,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 3 --samples 10 --name prob_update --eta 0.0005 --elasticity inelastic --decay True --probabilistic 0.5
)

PAUSE   