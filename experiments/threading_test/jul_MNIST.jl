using PyCall

py"""
def setup():
    import sys
    sys.path.append('../../sim_soens')
    sys.path.append('../../')

    # from sim_soens.super_input import SuperInput
    # from sim_soens.super_node import SuperNode
    from sim_soens.super_functions import picklit
    # from sim_soens.soen_sim import network, synapse
    print("Imports complete")
"""