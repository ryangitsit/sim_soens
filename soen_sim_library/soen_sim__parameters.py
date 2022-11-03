def common_parameters():
    
    jj_params = {'Ic': 100, # uA
                 'beta_c': 0.3, # dimensionless
                 }
    
    syn_params = {'syn_tau_rise': 0.02, # ns
                  'syn_tau_fall': 50, # ns
                  'syn_hotspot_duration': 3, # number of tau rise time constants
                  'syn_spd_duration': 8, # number of tau fall time constants
                  'syn_spd_phi_peak': 0.5, # Phi_0
                  }
    
    params = {**jj_params, **syn_params}
    
    return params