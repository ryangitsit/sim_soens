ECHO OFF

@REM FOR /L %%i IN (1,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.25^
@REM       --duration      1000^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --name          fanin_1.75_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --eta           0.001^
@REM       --fixed         .5^
@REM       --rand_flux     0.1^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.75^
@REM       --decay         True
@REM )


@REM -----------------------------
@REM           no decay
@REM ------------------------------
@REM FOR /L %%i IN (1,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.25^
@REM       --duration      1000^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.005^
@REM       --name          fanin_1.75_nodec_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --fixed         .5^
@REM       --rand_flux     0.1^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.75
@REM )

@REM FOR /L %%i IN (511,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.25^
@REM       --duration      2000^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.005^
@REM       --name          spread_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.5^
@REM       --target        50
@REM )

@REM FOR /L %%i IN (1060,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.25^
@REM       --duration      2500^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.005^
@REM       --exp_name      target50_maxflux_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --max_offset    0.5^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.5^
@REM       --target        50
@REM )

FOR /L %%i IN (1,1,10) DO (
    python exp_MNIST_full.py^
      --run           %%i^
      --s_th          0.25^
      --duration      2500^
      --beta          3^
      --dt            1.0^
      --jul_threading 4^
      --digits        10^
      --samples       50^
      --eta           0.005^
      --exp_name      target5_maxflux_full^
      --backend       julia^
      --dataset       MNIST^
      --max_offset    0.5^
      --fixed         .5^
      --rand_flux     0.005^
      --layers        6^
      --lay_weighting 1,1,1,4,8,10^
      --norm_fanin    True^
      --fan_coeff     1.5^
      --target        5
)



@REM --lay_weighting 1,1,1,4,8,10^
@REM --norm_fanin    True
@REM --hebbian       True^
@REM --exin          10,0,90^
@REM --inh_counter   True
@REM --elast         elastic^ 
@REM --low_bound     0^



@REM FOR /L %%i IN (1849,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.25^
@REM       --duration      1000^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --name          fanin_1.5_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --eta           0.001^
@REM       --fixed         .5^
@REM       --rand_flux     0.1^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --decay         True
@REM )

PAUSE