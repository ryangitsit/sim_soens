ECHO OFF

FOR %%a IN (1.7,1.8,1.9,2.0) DO (
    FOR %%b IN (50,150,250) DO (
        FOR %%c IN (2,3,4) DO (
            FOR %%d IN (0.25,0.5,0.75) DO (
                FOR %%e IN (.005,0.01,0.015,0.02,0.025) DO (
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