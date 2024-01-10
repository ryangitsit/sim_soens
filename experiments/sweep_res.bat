ECHO OFF

@REM FOR %%a IN (0.005) DO (

@REM     FOR %%b IN (50,100) DO (
@REM         FOR %%c IN (250,500) DO (

@REM             FOR %%d IN (0.1,.25) DO (
@REM                 FOR %%e IN (0.1,.25) DO (

@REM                     FOR %%f IN (2.25,2.5) DO (
@REM                         FOR %%g IN (1.5,1.75) DO (

@REM                             FOR %%h IN (0.1,0.25,.5) DO (
@REM                                 FOR %%i IN (0.05,0.1,0.15) DO (

@REM                                     python LSM_allspikes.py^
@REM                                         --N 98^
@REM                                         --eta %%a^
@REM                                         --nodes_tau %%b^
@REM                                         --codes_tau %%c^
@REM                                         --nodes_s_th %%d^
@REM                                         --codes_s_th %%e^
@REM                                         --fan_coeff_nodes %%f^
@REM                                         --fan_coeff_codes %%g^
@REM                                         --density %%h^
@REM                                         --res_connect_coeff %%i
@REM                                 )
@REM                             )
@REM                         )
@REM                     )
@REM                 )
@REM             )
@REM         )
@REM     )
@REM )

@REM FOR /L %%i IN (0,1,10000) DO (
@REM     python LSM_allspikes.py^
@REM         --N 98^
@REM         --evolve True
@REM )

@REM N=490
FOR %%a IN (0.005) DO (

    FOR %%b IN (50,100) DO (
        FOR %%c IN (250,500) DO (

            FOR %%d IN (0.25,.5) DO (
                FOR %%e IN (0.25,.5) DO (

                    FOR %%f IN (1.5,2.25) DO (
                        FOR %%g IN (1,1.5) DO (

                            FOR %%h IN (0.01,0.1,.25) DO (
                                FOR %%i IN (0.05,0.1,0.15) DO (

                                    python LSM_allspikes.py^
                                        --N 490^
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

FOR /L %%i IN (0,1,10000) DO (
    python LSM_allspikes.py^
        --N 490^
        --evolve True
)

PAUSE