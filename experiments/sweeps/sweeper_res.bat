ECHO OFF

FOR /L %%i IN (1,1,100) DO (
    FOR %%j IN (2,3,4) DO (
        FOR %%k IN (50,100,200) DO (
            FOR %%l IN (50,100,200) DO (
                python exp_point_res.py --run %%i --runs 100 --beta %%j --tau %%k --tau_ref %%l
            )
        )
    )
)

PAUSE
