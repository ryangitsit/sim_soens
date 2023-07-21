ECHO OFF

FOR /L %%i IN (149,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 3 --samples 10 --name learning_decay --eta 0.0005 --elasticity unbounded --decay True
)

PAUSE   