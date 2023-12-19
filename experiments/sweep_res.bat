
@REM FOR /L %%i IN (0,1,100000) DO (
@REM     python exp_MNIST_full.py^
@REM       --run           %%i^
@REM       --s_th          0.05^
@REM       --duration      1000^
@REM       --beta          3^
@REM       --dt            1.0^
@REM       --jul_threading 4^
@REM       --digits        10^
@REM       --samples       50^
@REM       --eta           0.001^
@REM       --exp_name      thresh_0.5_noref_full^
@REM       --backend       julia^
@REM       --dataset       MNIST^
@REM       --max_offset    0.5^
@REM       --fixed         .5^
@REM       --rand_flux     0.005^
@REM       --layers        6^
@REM       --lay_weighting 1,1,1,4,8,10^
@REM       --norm_fanin    True^
@REM       --fan_coeff     1.5^
@REM       --target        25
@REM )


ECHO OFF

FOR %%a IN (0.005) DO (

    FOR %%b IN (50,100) DO (
        FOR %%c IN (250,500) DO (

            FOR %%d IN (0.1,.25) DO (
                FOR %%e IN (0.1,.25) DO (

                    FOR %%f IN (2.25,2.5) DO (
                        FOR %%g IN (1.5,1.75) DO (

                            FOR %%h IN (0.1,0.25,.5) DO (
                                FOR %%i IN (0.05,0.1,0.15) DO (

                                    python LSM_allspikes.py^
                                        --eta %%a^
                                        --nodes_tau %%b^
                                        --codes_tau %%c^
                                        --nodes_s_th %%d^
                                        --codes_s_th %%e^
                                        --fan_coeff_nodes %%f^
                                        --fan_coeff_codes %%g^
                                        --density %%h^
                                        --res_connect_coeff %%i
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

PAUSE