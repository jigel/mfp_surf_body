from setuptools import setup, find_packages

setup(
    name = 'mfp_stationary_phases',
    version = '0.0.0a0',
    description = 'Matched Field Processing for stationary phases',
    #long_description =
    # url = 
    author = 'J.Igel, D.Bowden, K.Sager, A.Fichtner',
    author_email  = 'jonas.igel@erdw.ethz.ch',
    license = 'MIT',
    # license
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Topic :: Seismology',
        'Programming Language :: Python :: 3',
    ],
    keywords = 'matched field processing',
    packages = find_packages(),
    #package_data = ,
    install_requires = [
        "numpy",
        "scipy",
        "cartopy",
        "pandas",
        "pyyaml",
        "matplotlib",
        "pandas",
        "obspy",
        "mpi4py",
        "cmasher"],
)