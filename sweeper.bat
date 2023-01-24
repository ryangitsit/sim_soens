ECHO OFF

FOR /L %%i IN (1,1,100) DO (
    python exp_single_layer_rand.py --run %%i --runs 100 --form WTA --dir WTA
)