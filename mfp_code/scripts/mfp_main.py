import numpy as np
import os
from glob import glob
from pandas import read_csv
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy.signal.filter import envelope
import obspy
from obspy.taup import TauPyModel

import sys

import functools
print = functools.partial(print, flush=True)

import psutil
process = psutil.Process(os.getpid())


def mfp(args,comm,size,rank):
    """
    Matched Field Processing
    """
    
    # set taup model
    model = TauPyModel(model=args.taup_model)

    
    # first get all possible station pairs and distances
    stationlist = read_csv(args.stationlist_path,keep_default_na = False)
    stat_lat = stationlist['lat']
    stat_lon = stationlist['lon']

    stat_dict = dict()
    stat_pair = dict()

    for i in stationlist.iterrows():

        net_1,sta_1,lat_1,lon_1 = i[1]['net'],i[1]['sta'],i[1]['lat'],i[1]['lon']
        stat_dict.update({f"{net_1}.{sta_1}":[lat_1,lon_1]})

        for j in stationlist.iterrows():
    
            net_2,sta_2,lat_2,lon_2 = j[1]['net'],j[1]['sta'],j[1]['lat'],j[1]['lon']
            dist_var,az_var, baz_var = gps2dist_azimuth(lat_1,lon_1,lat_2,lon_2)
        
            stat_pair.update({f"{net_1}.{sta_1}--{net_2}.{sta_2}":[dist_var,az_var,baz_var]})
    
    # find all correlation files with corr_format
    if not isinstance(args.corr_format,list):
        args.corr_format = [args.corr_format]
        
        
    if rank == 0:
        print("Looking for cross-correlations..")
    
    corr_files = [os.path.join(args.correlation_path,file) for file in os.listdir(args.correlation_path) if file.split(".")[-1] in args.corr_format]

    if corr_files == []:
        if rank == 0:
            print("Found no cross-correlations. Exiting..")
        sys.exit()
    
        
    sourcegrid = np.load(args.sourcegrid_path)
    
    if rank == 0:
        print(f"Found {np.size(corr_files)} cross-correlations.")
        print(f"Grid has {np.size(sourcegrid[0])} gridpoints.")
        
    comm.Barrier()
    
        
    # Don't load all correlations into dictionary, takes too long
    # split up correlations instead
    # need three loops: phase, correlation, grid
    # In Memory: grid, array for each phase pair

    # split up correlation files
    if rank == 0:
        corr_files_split = np.array_split(corr_files,size)
        corr_files_split = [k.tolist() for k in corr_files_split]
    else:
        corr_files_split = None
    
    corr_files_split = comm.scatter(corr_files_split,root=0) 
    
    if rank == 0:
        print(f"On average {np.around(np.size(corr_files)/size,2)} correlations per core.")
        
    comm.barrier()
    
    # if maximum distance is 0, set it to 360 so it's ignored
    if args.station_distance_max in [0, None]:
        args.station_distance_max = 360
    
    
    # initiate dictionary for MFP phase maps
    # content is grid[0],grid[1],basic,envelope
    if args.stationary_phases:
        MFP_phases_dict = dict()

        for phases in args.phase_list:
            MFP_phases_dict[f'{phases[0]}-{phases[1]}'] = np.asarray([sourcegrid[0],sourcegrid[1],np.zeros(np.shape(sourcegrid[0])),np.zeros(np.shape(sourcegrid[0]))])

    # different dictionary if we're just doing a simple MFP for individual phases separately
    else: 
        MFP_dict = dict()
        
        for phase in args.phases:
            MFP_dict[f'{phase}'] = np.asarray([sourcegrid[0],sourcegrid[1],np.zeros(np.shape(sourcegrid[0])),np.zeros(np.shape(sourcegrid[0]))])

    
    for i,corr in enumerate(corr_files_split):
        
        if i%1 == 0 and rank == 0:
            print(f"At {i} of {np.size(corr_files_split)} correlations on rank {rank}")

        file_name = os.path.basename(corr)

        net_1,sta_1 = file_name.split('.')[0],file_name.split('.')[1]
        net_2,sta_2 = file_name.split('--')[1].split('.')[0],file_name.split('--')[1].split('.')[1]

        stat_1 = f"{net_1}.{sta_1}"
        stat_2 = f"{net_2}.{sta_2}"
        
        stat_pair_name = f"{net_1}.{sta_1}--{net_2}.{sta_2}"
    
        dist_var = kilometers2degrees(stat_pair[stat_pair_name][0]/1000) # change distance from metres to degrees

        if args.station_distance_max > dist_var > args.station_distance_min:            
            if args.bandpass_filter is not None:
                tr_corr = obspy.read(corr)[0].filter('bandpass',freqmin=args.bandpass_filter[0],freqmax=args.bandpass_filter[1],corners=args.bandpass_filter[2],zerophase=True)
            else:
                tr_corr = obspy.read(corr)[0]
    
    
        ## NEED TO MAYBE IMPLEMENT NORMALISATION HERE
        # get data and envelope
        data = tr_corr.data
        data_env = envelope(data)

        dt = tr_corr.stats.delta
        npts = tr_corr.stats.npts

        # calculate time array
        time_shift = (dt*npts-1)/2
        time = np.linspace(-time_shift,time_shift,npts)

        
        
        for it,phase in enumerate(args.phase_list):

            if args.stationary_phases:
                phase_1 = phase[0]
                phase_2 = phase[1]

                # load grid from dictionary
                mfp_grid = np.asarray([MFP_phases_dict[f"{phase_1}-{phase_2}"][0],MFP_phases_dict[f"{phase_1}-{phase_2}"][1]])

            else:
                # load grid from dictionary
                mfp_grid = np.asarray([MFP_dict[f"{phase}"][0],MFP_dict[f"{phase}"][1]])


            # iterate over each grid point and calculate arrival time
            for k in range(np.size(mfp_grid[0])):
                
                if k%100 == 0 and rank == 0 and args.stationary_phases:
                    print(f"At {k} of {np.size(mfp_grid[0])} gridpoints for phase {it+1} of {int(np.size(args.phase_list)/2)} on rank {rank}".ljust(100,' '),end="\r", flush=True)
                    
                if k%100 == 0 and rank == 0 and not args.stationary_phases:
                    print(f"At {k} of {np.size(mfp_grid[0])} gridpoints for phase {it+1} of {int(np.size(args.phase_list))} on rank {rank}".ljust(100,' '),end="\r", flush=True)

                g_point = [mfp_grid[0][k],mfp_grid[1][k]]


                lat_1,lon_1 = stat_dict[stat_1][0],stat_dict[stat_1][1]
                lat_2,lon_2 = stat_dict[stat_2][0],stat_dict[stat_2][1]

                
                # calculate distances
                dist_1_m = gps2dist_azimuth(g_point[0],g_point[1],lat_1,lon_1)[0]
                dist_2_m = gps2dist_azimuth(g_point[0],g_point[1],lat_2,lon_2)[0]
                
                dist_1 = kilometers2degrees(dist_1_m/1000)
                dist_2 = kilometers2degrees(dist_2_m/1000)


                # calculate arrival times to each station for the two phases
                if args.stationary_phases:
                    arr_1 = model.get_travel_times(source_depth_in_km=0,distance_in_degree=dist_1,phase_list=[phase_1])
                else:
                    # calculate arrival times to each station for the two phases
                    arr_1 = model.get_travel_times(source_depth_in_km=0,distance_in_degree=dist_1,phase_list=[phase])

                    
                # if there is no arrival, skip this one
                if len(arr_1) == 0:
                    continue
                else:
                    # Use the first arrival
                    arr_1_val = arr_1[0].time

                    
                if args.stationary_phases:
                    arr_2 = model.get_travel_times(source_depth_in_km=0,distance_in_degree=dist_2,phase_list=[phase_2])
                else:
                    arr_2 = model.get_travel_times(source_depth_in_km=0,distance_in_degree=dist_2,phase_list=[phase])


                # if there is no arrival, skip this one
                if len(arr_2) == 0:
                    continue
                else:
                    # Use the first arrival
                    arr_2_val = arr_2[0].time


                # get the expected arrival in the correlation
                # Check here if it's the right way around
                arr_diff = arr_2_val-arr_1_val

                # If the arrival time is outside the correlation, skip
                if np.abs(arr_diff) > np.max(np.abs(time)):
                    continue

                # get the index in the correlation
                corr_idx = np.argmin(np.abs(arr_diff-time),axis=0)
                
                # geometric spreading for surface waves
                            
                if args.geo_spreading and phase.endswith('kmps'):
                    
                    g_speed = float(phase.split('kmps')[0])*1000
                    # geometric spreading term
                    geo_dist_var = np.abs(dist_2_m + dist_1_m)/(2*1000) # average distance same units as speed
                    
                    if args.bandpass_filter is not None:
                        omega = (args.bandpass_filter[0]+args.bandpass_filter[1])/2
                    else:
                        if k == 0 and rank == 0:
                            print('No bandpass filter. Using 0.1 Hz as frequency for geometric spreading.')
                        omega = 0.1
                        
                    A = np.sqrt((2*g_speed)/(np.pi*omega*geo_dist_var))
                
                # get values and add to distribution 
                for meth in args.method:
                    # basic is dictionary index 2
                    # envelope is dictionary index 3
                    if args.stationary_phases:
                        if meth == "basic":
                            MFP_phases_dict[f"{phase_1}-{phase_2}"][2][k] += data[corr_idx]
                        elif meth == "envelope":
                            MFP_phases_dict[f"{phase_1}-{phase_2}"][3][k] += data_env[corr_idx]
                        else:
                            print(f"{meth} not implemented.")
                            
                    else:
                        if args.geo_spreading and phase.endswith('kmps'):
                            if meth == "basic":
                                MFP_dict[f"{phase}"][2][k] += data[corr_idx] * A
                            elif meth == "envelope":
                                MFP_dict[f"{phase}"][3][k] += data_env[corr_idx] * A
                            else:
                                print(f"{meth} not implemented.")
                                
                        else:
                            if meth == "basic":
                                MFP_dict[f"{phase}"][2][k] += data[corr_idx] 
                            elif meth == "envelope":
                                MFP_dict[f"{phase}"][3][k] += data_env[corr_idx] 
                            else:
                                print(f"{meth} not implemented.")
                        
            
            
                        
    # MPI NEED TO GATHER AND ADD
    comm.Barrier()
    
    if rank == 0:
        print("Matched Field Processing done.".ljust(100,' '))
    
    if args.stationary_phases:
        MFP_phases_dict_all = comm.gather(MFP_phases_dict,root = 0)
    else:
        MFP_dict_all = comm.gather(MFP_dict,root = 0)

    # expand the dictionary
    if rank == 0:
        if args.stationary_phases:
            MFP_phases_dict_exp = dict()

            for phases in args.phase_list:
                MFP_phases_dict_exp[f'{phases[0]}-{phases[1]}'] = np.asarray([sourcegrid[0],sourcegrid[1],np.zeros(np.shape(sourcegrid[0])),np.zeros(np.shape(sourcegrid[0]))])

            for subdict in MFP_phases_dict_all:
                for phases in subdict:
                    # add the MFP maps from the different ranks
                    MFP_phases_dict_exp[phases][2] += subdict[phases][2]
                    MFP_phases_dict_exp[phases][3] += subdict[phases][3]
        else:
            MFP_dict_exp = dict()

            for phase in args.phases:
                MFP_dict_exp[f'{phase}'] = np.asarray([sourcegrid[0],sourcegrid[1],np.zeros(np.shape(sourcegrid[0])),np.zeros(np.shape(sourcegrid[0]))])

            for subdict in MFP_dict_all:
                for phase in subdict:
                    # add the MFP maps from the different ranks
                    MFP_dict_exp[phase][2] += subdict[phase][2]
                    MFP_dict_exp[phase][3] += subdict[phase][3]
    else:
        if args.stationary_phases:
            MFP_phases_dict_exp = dict()
        else:
            MFP_dict_exp = dict()
    
    comm.Barrier()
    
    if args.stationary_phases:
        MFP_phases_dict_exp = comm.bcast(MFP_phases_dict_exp,root=0)
        MFP_final = MFP_phases_dict_exp
        
    else:
        MFP_dict_exp = comm.bcast(MFP_dict_exp,root=0)
        MFP_final = MFP_dict_exp

    
    return MFP_final