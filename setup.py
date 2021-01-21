from setuptools import setup, find_packages

setup(
    name = 'mfp_stationary_phases',
    version = '0.0.0a0',
    description = 'Match Field Processing for stationary phases',
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
    keywords = 'match field processing',
    packages = find_packages(),
    #package_data = ,
    install_requires = [
        "numpy",
        "scipy",
        "cartopy",
        "pandas",
        "pyyaml",
        "h5py",
        "pandas",
        "obspy"],
)