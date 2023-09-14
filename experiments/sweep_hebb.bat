ECHO OFF

FOR /L %%i IN (676,1,100000) DO (
    python exp_MNIST_full.py --run %%i --jul_threading 4 --digits 3 --samples 10 --name hebb_test --low_bound 0 --eta 0.0001 --hebbian True
)
@REM python exp_MNIST_full.py --jul_threading 4 --digits 3 --samples 10 --name exin_test --low_bound 0 --eta 0.001 --hebbian True --exin [-.2,0,.8]
PAUSE
