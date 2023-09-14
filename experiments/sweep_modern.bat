ECHO OFF

FOR /L %%i IN (1,1,100000) DO (
    python exp_MNIST_full.py^
      --run           %%i^
      --jul_threading 4^
      --digits        3^
      --samples       10^
      --name          modern_layers^
      --low_bound     0^
      --eta           0.001^
      --hebbian       True^
      --exin          10,0,90^
      --fixed         .5^
      --rand_flux     0.1^
      --layers        5
)

PAUSE
