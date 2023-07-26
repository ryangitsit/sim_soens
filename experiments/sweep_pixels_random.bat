ECHO OFF


FOR /L %%i IN (31,1,40) DO (
    FOR %%a IN (1.8) DO (
        FOR %%b IN (50) DO (
            FOR %%c IN (2) DO (
                FOR %%d IN (0.5) DO (
                    FOR %%e IN (0.01,0.02) DO (
                        FOR %%f IN ("True", "False", "None") DO (
                            FOR %%g IN ("True", "False") DO (
                                python ./exp_learn_prime.py --exp_name "pixels_random" --backend "julia" --ib %%a  --tau %%b  --beta %%c  --s_th %%d --eta  %%e  --elast %%f  --valid %%g --run %%i --weights random
                            )
                        )
                    )
                )
            )
        )
    )
)

PAUSE