ECHO
FOR /L %%i IN (0,1,10000) DO (
    python LSM_allspikes.py^
        --N 98^
        --evolve True
)
PAUSE