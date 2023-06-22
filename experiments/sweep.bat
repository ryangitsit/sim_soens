ECHO OFF

FOR %%a IN (1.8,2.0) DO (
    FOR %%b IN (50,150) DO (
        FOR %%c IN (2,3) DO (
            FOR %%d IN (0.25,0.5,0.75) DO (
                FOR %%e IN (0.01,0.015,0.02) DO (
                    FOR %%f IN ("True", "False", "None") DO (
                        FOR %%g IN ("True", "False") DO (
                            python ./exp_learning.py --ib %%a  --tau %%b  --beta %%c  --s_th %%d --eta  %%e  --elast %%f  --valid %%g
                        )
                    )
                )
            )
        )
    )
)

PAUSE