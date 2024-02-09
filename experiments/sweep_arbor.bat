ECHO OFF

FOR %%a IN (0.05, 0.1, 0.25) DO (
    FOR %%b IN (100, 150, 500) DO (
        FOR %%c IN (1.0, 1.5, 2) DO (
            FOR %%d IN (10,25,50) DO (
                FOR %%e IN (.005, 0.05, 0.5) DO (
                    FOR %%f IN (.4, .5, .6) DO (

                        FOR /L %%z IN (0,1,1501) DO (
                            python exp_MNIST_full.py^
                            --run           %%z^
                            --s_th          %%a^
                            --tau           %%b^
                            --fan_coeff     %%c^
                            --target        %%d^
                            --rand_flux     %%e^
                            --max_offset    %%f^
                            --exin          20,0,80^
                            --inh_counter   True^
                            --duration      1000^
                            --beta          3^
                            --dt            1.0^
                            --jul_threading 4^
                            --digits        10^
                            --samples       50^
                            --eta           0.001^
                            --exp_name      arbor_sweep^
                            --backend       julia^
                            --dataset       MNIST^
                            --fixed         .5^
                            --layers        6^
                            --lay_weighting 1,1,1,4,8,10^
                            --multi         True^
                            --norm_fanin    True
                        )

                        FOR /L %%z IN (0,1,1501) DO (
                            python exp_MNIST_full.py^
                            --run           %%z^
                            --s_th          %%a^
                            --tau           %%b^
                            --fan_coeff     %%c^
                            --target        %%d^
                            --rand_flux     %%e^
                            --max_offset    %%f^
                            --duration      1000^
                            --beta          3^
                            --dt            1.0^
                            --jul_threading 4^
                            --digits        10^
                            --samples       50^
                            --eta           0.001^
                            --exp_name      arbor_sweep^
                            --backend       julia^
                            --dataset       MNIST^
                            --fixed         .5^
                            --layers        6^
                            --lay_weighting 1,1,1,4,8,10^
                            --multi         True^
                            --norm_fanin    True
                        )

                    )
                )
            )
        )
    )
)

PAUSE