ECHO OFF

FOR /L %%i IN (1,1,100000) DO (
    python exp_MNIST_full.py --run %%i --digits 3 --samples 1 --name M_lite_hi_eta_rerun --eta 0.005 --elasticity inelastic --dataset MNIST --plotting full
)

PAUSE   