ECHO OFF

FOR /L %%i IN (1,1,10) DO (
    FOR %%j IN (2,3,4) DO (
        FOR %%k IN (50,100,200) DO (
            FOR %%l IN (50,100,200) DO (
                FOR %%m IN (10,5,2) DO (
                    python exp_single_layer_rand.py --run %%i --runs 10 --form WTA --dir WTA_%%j_%%k_%%l_%%m  --beta %%j --tau %%k --tau_ref %%l --inhibit %%m
                )
            )
        )
    )
)

PAUSE