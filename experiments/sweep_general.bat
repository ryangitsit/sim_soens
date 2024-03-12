ECHO OFF

@REM @REM @REM  low target, slow learning rate, long duration
@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          2000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.005^
@REM       --exp_name          targets_high_long_python^
@REM       --backend           python^
@REM       --dataset           MNIST^
@REM       --max_offset        phi_off^
@REM       --fixed             .5^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         3^
@REM       --fan_buffer        0.0^
@REM       --multi             True^
@REM       --offset_transfer   W_symmetric_relu_nobias_1000^
@REM       --target            42^
@REM       --off_target        21
@REM )

@REM @REM  low target, slow learning rate, long duration
FOR /L %%i IN (0,1,100000) DO (
    python exp_MNIST_full.py^
      --run               %%i^
      --s_th              0.1^
      --duration          1000^
      --beta              3^
      --dt                1.0^
      --jul_threading     4^
      --digits            10^
      --samples           50^
      --eta               0.00025^
      --exp_name          targets_high_slow^
      --backend           julia^
      --dataset           MNIST^
      --max_offset        phi_off^
      --fixed             .5^
      --norm_fanin_prime  True^
      --fan_coeff         3^
      --fan_buffer        0.0^
      --multi             True^
      --offset_transfer   W_symmetric_relu_nobias_1000^
      --target            42^
      --off_target        21
)



@REM @REM @REM  low target, slow learning rate, long duration
@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          1000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.0025^
@REM       --exp_name          low_slow_long^
@REM       --backend           julia^
@REM       --dataset           MNIST^
@REM       --max_offset        phi_off^
@REM       --fixed             .5^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         3^
@REM       --fan_buffer        0.0^
@REM       --multi             True^
@REM       --offset_transfer   W_symmetric_relu_nobias_1000^
@REM       --target            10
@REM )


@REM @REM  --target 10 for _not_all
@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          1000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.005^
@REM       --exp_name          weight_transfer_asymmetric^
@REM       --no_negative_jij   True^
@REM       --backend           julia^
@REM       --dataset           MNIST^
@REM       --max_offset        phi_off^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         3^
@REM       --fan_buffer        0.0^
@REM       --multi             True^
@REM       --weight_transfer   W_symmetric_relu_nobias_1000^
@REM       --target            25
@REM )



@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          1000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.005^
@REM       --exp_name          weight_transfer_inh_counting^
@REM       --inh_counter       True^
@REM       --backend           julia^
@REM       --dataset           MNIST^
@REM       --max_offset        phi_off^
@REM       --fixed             .5^
@REM       --rand_flux         0.0^
@REM       --layers            6^
@REM       --lay_weighting     1,1,1,4,8,10^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         3^
@REM       --fan_buffer        0.0^
@REM       --multi             True^
@REM       --weight_transfer   W_symmetric_relu_nobias_1000^
@REM       --target            10
@REM )

@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          1000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.005^
@REM       --exp_name          weight_transfer^
@REM       --backend           julia^
@REM       --dataset           MNIST^
@REM       --max_offset        phi_off^
@REM       --fixed             .5^
@REM       --layers            6^
@REM       --lay_weighting     1,1,1,4,8,10^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         3^
@REM       --fan_buffer        0.0^
@REM       --multi             True^
@REM       --weight_transfer   W_symmetric_relu_nobias_1000^
@REM       --target            10
@REM )


@REM @REM  --target 10 for _not_all
@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          1000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.005^
@REM       --exp_name          offset_transfer_all_4.5^
@REM       --backend           julia^
@REM       --dataset           MNIST^
@REM       --max_offset        phi_off^
@REM       --fixed             .5^
@REM       --layers            6^
@REM       --lay_weighting     1,1,1,4,8,10^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         4.5^
@REM       --fan_buffer        0.0^
@REM       --multi             True^
@REM       --offset_transfer   W_symmetric_relu_nobias_1000^
@REM       --target            25
@REM )











@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.1^
@REM       --duration      1000^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.005^
@REM       --exp_name      steady_inp_threshfull^
@REM       --backend       julia^
@REM       --dataset       keras^
@REM       --max_offset    half^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.5^
@REM       --multi         True^
@REM       --target        10
@REM )


@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          1000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.005^
@REM       --exp_name          steady_simple_boosted^
@REM       --backend           julia^
@REM       --dataset           keras^
@REM       --max_offset        phi_off^
@REM       --fixed             .5^
@REM       --rand_flux         0.005^
@REM       --layers            6^
@REM       --lay_weighting     1,1,1,4,8,10^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         1.5^
@REM       --fan_buffer        0.0^
@REM       --multi             True^
@REM       --target            10
@REM )

@REM python alternode.py
@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --exp_name          updates_cobuff^
@REM       --alternode         updates_cobuff_alternode_inh_renormed
@REM )

@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --exp_name          updates_cobuff^
@REM       --alternode         updates_cobuff_alternode_inh
@REM )

@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          1000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.001^
@REM       --exp_name          updates_cobuff^
@REM       --backend           julia^
@REM       --dataset           MNIST^
@REM       --max_offset        inverse^
@REM       --fixed             .5^
@REM       --rand_flux         0.005^
@REM       --layers            6^
@REM       --lay_weighting     1,1,1,4,8,10^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         2.25^
@REM       --fan_buffer        0.5^
@REM       --multi             True^
@REM       --target            25
@REM )

@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          1000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.005^
@REM       --exp_name          fanin_inhibitory^
@REM       --backend           julia^
@REM       --dataset           MNIST^
@REM       --max_offset        True^
@REM       --fixed             .5^
@REM       --rand_flux         0.005^
@REM       --layers            6^
@REM       --lay_weighting     1,1,1,4,8,10^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         0^
@REM       --multi             True^
@REM       --target            25
@REM )

@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run               %%i^
@REM       --s_th              0.1^
@REM       --duration          1000^
@REM       --beta              3^
@REM       --dt                1.0^
@REM       --jul_threading     4^
@REM       --digits            10^
@REM       --samples           50^
@REM       --eta               0.005^
@REM       --exp_name          fanin_symm_py^
@REM       --backend           python^
@REM       --dataset           MNIST^
@REM       --max_offset        True^
@REM       --fixed             .5^
@REM       --rand_flux         0.005^
@REM       --layers            6^
@REM       --lay_weighting     1,1,1,4,8,10^
@REM       --norm_fanin_prime  True^
@REM       --fan_coeff         0^
@REM       --multi             True^
@REM       --target            25
@REM )


@REM -----------------------------
@REM        fanin prime
@REM ------------------------------
@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.1^
@REM       --duration      1000^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.001^
@REM       --exp_name      fanin_prime^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --max_offset    0.5^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --fan_coeff     1.5^
@REM       --target        25
@REM )

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

@REM FOR /L %%i IN (1,1,10) DO (
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
@REM       --exp_name      target5_maxflux_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --max_offset    0.5^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.5^
@REM       --target        5
@REM )

@REM @REM changed to 32 bit at 5504 -- 6062
@REM FOR /L %%i IN (6706,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.25^
@REM       --duration      1000^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.005^
@REM       --exp_name      speed_target15_full2^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --max_offset    0.5^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.5^
@REM       --target        15
@REM )

@REM changed to 32 bit at 1607 -- 1907 
@REM FOR /L %%i IN (1799,1,100000) DO (
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
@REM       --exp_name      target5_maxflux_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --max_offset    0.5^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.5^
@REM       --target        5
@REM )


@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.25^
@REM       --duration      1000^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.005^
@REM       --exp_name      speed_target15_full_fan1^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --max_offset    0.5^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1^
@REM       --target        15
@REM )


@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.05^
@REM       --duration      5000^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.0001^
@REM       --exp_name      long_slow_full^
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


@REM FOR /L %%i IN (36,1,10000000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.25^
@REM       --duration      1000^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.005^
@REM       --exp_name      tiling_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --max_offset    0.5^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        5^
@REM       --lay_weighting 1,1,1,4,8^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.5^
@REM       --target        15^
@REM       --tiling        True
@REM )

@REM FOR /L %%i IN (0,1,10000000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.25^
@REM       --duration      1000^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.005^
@REM       --exp_name      tiling_deep_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --max_offset    0.5^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        9^
@REM       --lay_weighting 1,1,1,4,8,10,12,12,12^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.5^
@REM       --target        15
@REM )


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