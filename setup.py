from setuptools import setup, find_packages

setup(name='sim_soens',
        version='0.1',
        description='simulator for superconducting optoelectronic network',
        url='https://github.com/ryangitsit/sim_soens',
        author='Ryan O\'Loughlin',
        author_email='rmoloughlin11@gmail.com',
        license='NIST',
        zip_safe=False,
        packages=find_packages(where="sim_soens"),
        # package_dir={"": "soen_sim_data"},
        include_package_data=True
)
