
# import sys
# print(sys.version)

import pip
import sys, os
print(os.path.dirname(sys.executable))
def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

# Example
if __name__ == '__main__':
    install('PyCall')
    install('julia')

    import julia
    # import PyCall

    '''
    Before running this file, download Julia via the comman line using from 
        - https://julialang.org/downloads/platform/
    Then use this file to install julia via the julia python package
    '''

    julia.install()

