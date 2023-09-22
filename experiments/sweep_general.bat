ECHO OFF

FOR /L %%i IN (1,1,100000) DO (
    python exp_MNIST_full.py^
      --run           %%i^
      --s_th          0.25^
      --duration      1000^
      --jul_threading 4^
      --digits        3^
      --samples       10^
      --name          simple_long^
      --dataset       MNIST^
      --elast         elastic^
      --low_bound     0^
      --eta           0.001^
      --fixed         .5^
      --rand_flux     0.1^
      --layers        3^
      --decay         True
)

PAUSE

@REM --lay_weighting 1,1,1,4,8,10^
@REM --norm_fanin    True
@REM --hebbian       True^
@REM --exin          10,0,90^
@REM --inh_counter   True