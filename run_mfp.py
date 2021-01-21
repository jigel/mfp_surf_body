import numpy as np
import yaml 
import os
import sys
import shutil
from glob import glob
from pandas import read_csv
import matplotlib
matplotlib.use('Agg')
import time
import psutil
process = psutil.Process(os.getpid())
import functools
print = functools.partial(print, flush=True)

# mfp codes
from mfp_code.scripts.create_sourcegrid import create_sourcegrid
from mfp_code.util.plot import plot_grid
from mfp_code.scripts.create_stat_phase_synthetics import create_synth
#from mfp_code.scripts.mfp_main import run_mfp

from mpi4py import MPI
# simple embarrassingly parallel run:
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

t_0 = time.time()



# use a class args to give values to different codes
class args(object):
    
    def __init__(self):
        
        self.project_name = None
        self.output_path = None
        self.project_path = None
        self.correlation_path = None
        self.corr_format = None
        self.stationlist_path = None
        self.station_distance_min = None
        self.station_distance_max = None
        self.sourcegrid_path = None
        self.svp_grid_config = None
        self.method = None
        #self.t_window = None
        self.stationary_phases = None
        self.stat_phase_input = None 
        self.bandpass_filter = [0.1,0.2,5]
        self.taup_model = 'iasp91'
        self.phases = ['3kmps','P']
        #self.smoothing = 2
        self.plot = True
        
        
# Read in the mfp_config.yml file which should be given as input after mfp code
mfp_args = args()

with open(sys.argv[1]) as f:
    mfp_config = yaml.safe_load(f)
    
    
for attr in mfp_config:
    setattr(mfp_args,attr,mfp_config[attr])
    
    
# Make project folder and copy files there
if rank == 0:
    print("===="*20)
    print(f"Starting project: {mfp_args.project_name}")
    print("===="*20)

    
# boolean used to check grid validity for all ranks
grid_val = True


# make project folder
if rank == 0:
    if not os.path.isdir(os.path.abspath(os.path.join(mfp_args.output_path,mfp_args.project_name))):
        os.makedirs(os.path.abspath(os.path.join(mfp_args.output_path,mfp_args.project_name)))
        mfp_args.project_path = os.path.abspath(os.path.join(mfp_args.output_path,mfp_args.project_name))
    else:
        mfp_args.project_path = os.path.abspath(os.path.join(mfp_args.output_path,mfp_args.project_name))

    print("Project path: ", mfp_args.project_path)
    
    # copy sourcegrid and stationlist
    shutil.copy(mfp_args.stationlist_path,os.path.join(mfp_args.project_path,'stationlist.csv'))
    mfp_args.stationlist_path = os.path.join(mfp_args.project_path,'stationlist.csv')
    
    # copy mfp_config
    shutil.copy(sys.argv[1],os.path.join(mfp_args.project_path,'mfp_config.yml'))
    
    # copy sourcegrid file or create new grid
    if mfp_args.sourcegrid_path.endswith('.npy'):

        shutil.copy(mfp_args.sourcegrid_path,os.path.join(mfp_args.project_path,'sourcegrid.npy'))
        mfp_args.sourcegrid_path = os.path.join(mfp_args.project_path,'sourcegrid.npy')
        
        sourcegrid = np.load(mfp_args.sourcegrid_path)
        
        plot_grid(file_path=mfp_args.sourcegrid_path,data_incl=False,
                  output_file=os.path.join(mfp_args.project_path,'sourcegrid.png'),
                  only_ocean=False,title=f'Sourcegrid with {np.size(sourcegrid[0])} gridpoints',
                  stationlist_path=mfp_args.stationlist_path)
        
    elif mfp_args.sourcegrid_path.lower() in ['svp_grid','svp','grid','svp_grids']:
        print("Creating new sourcegrid using the given parameters.")
        
        # create new svp grid
        sourcegrid = create_sourcegrid(mfp_args.svp_grid_config)
        # number of gridpoints
        print(f"Number of gridpoints: {np.size(sourcegrid[0])}")
        
        # save grid as npy file
        np.save(os.path.join(mfp_args.project_path,'sourcegrid.npy'),sourcegrid)
        
        mfp_args.sourcegrid_path = os.path.join(mfp_args.project_path,'sourcegrid.npy')
        
        # plot sourcegrid
        plot_grid(file_path=mfp_args.sourcegrid_path,data_incl=False,
                  output_file=os.path.join(mfp_args.project_path,'sourcegrid.png'),
                  only_ocean=False,title=f'Sourcegrid with {np.size(sourcegrid[0])} gridpoints',
                  stationlist_path=mfp_args.stationlist_path)
        
        
    else: 
        grid_val = False
        

comm.Barrier()

# exit all ranks if grid not valid
grid_val = comm.bcast(grid_val,root=0)
if not grid_val:
    raise Exception("No valid sourcegrid input given. Has to be either .npy file or svp_grid.")


if rank == 0:
    run_time = open(os.path.join(mfp_args.project_path,'runtime.txt'),'w+')
    run_time.write(f"Number of cores: {size} \n")
    t_1 = time.time()
    run_time.write(f"Project setup: {np.around((t_1-t_0)/60,4)} min \n")
    
    
    
    #### Check memory usage
    print("Memory usage in Gb at start ", process.memory_info().rss / 1.e9)
    run_time.write(f"Memory usage in Gb at start {process.memory_info().rss / 1.e9} \n")

    
comm.Barrier()

# If stationary phases are to be calculated, create a synthetic correlation data with Ricker/Gauss wavelet at arrival time
if mfp_args.stationary_phases:
    
    # check which phases are wanted
    phase_pair_list = []
    
    if mfp_args.phase_pairs == 'all':
        phase_pair_list = [[i,j] for i in mfp_args.phase_pairs for j in mfp_args.phase_pairs]
    elif isinstance(mfp_args.phase_pairs,list):
        phase_pair_list = []

        for pair in mfp_args.phase_pairs:
            list_1 = [i for i in pair.split('-')] # one way
            list_2 = [i for i in pair.split('-')[::-1]] # and the other way too

            phase_pair_list.append(list_1)
            phase_pair_list.append(list_2)
            
            
    # get the main phases to create synthetic data, i.e. the ones with only one letter
    mfp_args.main_phases = [i for i in mfp_args.phases if len(i) == 1]

    if rank == 0:
        print(f"Creating synthetic correlations for stationary phase analysis: {mfp_args.main_phases}")
        # create folder for these correlations
        for phase in mfp_args.main_phases:
            mfp_args.corr_stat_phase_path = os.path.join(mfp_args.project_path,f'corr_stat_phase_{phase}_{mfp_args.stat_phase_input.lower()}')

            if not os.path.isdir(mfp_args.corr_stat_phase_path):
                os.makedirs(mfp_args.corr_stat_phase_path)
            
    comm.Barrier()
    
    create_synth(mfp_args,comm,size,rank)

    if rank == 0:
        print("All synthetic correlations created")
    comm.Barrier()
    
    

    
        
    
### DO MFP
## Instead of iterating over the grid, iterate over the correlations

#mfp = run_mfp(mfp_args)