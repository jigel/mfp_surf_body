# mfp for avalanche detection

This is a slimmed down version of the stationary phase code. 
Match Field Processing for avalanche detection

## Packages
- numpy
- scipy
- pyyaml
- matplotlib
- pandas
- cartopy
- psutil
- cmasher
- mpi4py
- obspy

Note that obspy doesn't seem to work with python 3.9

## Install
Clone this repository, install packages above and go to the folder to run:

pip install -v -e .

## Run Example

To run a test case for either simple MFP or stationary phase MFP run:

python run_mfp.py mfp_example/mfp_config_simple.yml

or

python run_mfp.py mfp_example/mfp_config_stationary_phase.yml


## Change parameters

To set your own parameters, edit the mfp_config.yml file. The different parameters are explained in the config file. 

python run_mfp.py mfp_config.yml

The code can also be run with mpi:

mpirun -np {number of cores} python run_mfp.py mfp_config.yml


