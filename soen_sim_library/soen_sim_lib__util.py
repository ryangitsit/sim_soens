def arg_helper(params,parameter_name,default_value):
    
    if parameter_name in params.keys(): 
        value = params[parameter_name] 
    else: 
        value = default_value
    
    return value
