ECHO OFF

FOR %%a IN (1.8) DO (
    FOR %%b IN (50) DO (
        FOR %%c IN (2) DO (
            FOR %%d IN (0.5) DO (
                FOR %%e IN (0.02) DO (
                    FOR %%f IN ("True", "False", "None") DO (
                        FOR %%g IN ("True", "False") DO (
                            python ./exp_learn_prime.py --exp_name "pixels_rerun_python" --backend "python" --ib %%a  --tau %%b  --beta %%c  --s_th %%d --eta  %%e  --elast %%f  --valid %%g
                        )
                    )
                )
            )
        )
    )
)

PAUSE