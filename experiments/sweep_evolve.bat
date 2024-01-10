ECHO
FOR /L %%i IN (0,1,600) DO (
    python LSM_allspikes.py^
        --N 98^
        --evolve True
)
PAUSE