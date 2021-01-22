# mfp_stationary_phases
Match Field Processing for stationary phases

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
Clone this repository and go to the folder. Run:

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


