import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tri
from matplotlib import colors
import cmasher as cmr
from pandas import read_csv

cmap = cmr.cosmic    # CMasher
cmap = plt.get_cmap('cmr.cosmic')   # MPL



def plot_grid(file_path,data_incl=False,output_file=None,triangulate=False,cbar=False,only_ocean=False,title=None,stationlist_path=None):
    """
    Script to plot a grid with [[lat],[lon]]
    If data is included, should be [[lat],[lon],[data]]
    """
    
    grid = np.load(file_path)
    
    if data_incl:
        data = grid[2]
    # if no data is included, shouldn't triangulate
    else:
        triangulate = False
    
    plt.figure(figsize=(50,20))
    ax  = plt.axes(projection=ccrs.Robinson())
    
    if only_ocean:
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor=cfeature.COLORS['land']),zorder=2)
    else:
        ax.coastlines(color='k')
        
    if triangulate:
        grid_tri = tri.Triangulation(grid[0],grid[1])
        plt.tripcolor(grid_tri,data,cmap=cmap,linewidth=0.0,edgecolor='none',zorder=1,transform=ccrs.PlateCarree())

    else:
        # plot either data or grid
        if data_incl:
            plt.scatter(grid[0],grid[1],c=data,s=20,marker='o',cmap=cmap,zorder=1,transform=ccrs.PlateCarree())
        else:
            plt.scatter(grid[0],grid[1],c='k',s=20,marker='o',cmap=cmap,zorder=1,transform=ccrs.PlateCarree())
        
        
    if stationlist_path is not None:
        stationlist = read_csv(stationlist_path)
        lat = stationlist['lat']
        lon = stationlist['lon']
        plt.scatter(lon,lat,c='lawngreen',s=100,marker='^',edgecolor='k',linewidth=2,transform=ccrs.PlateCarree(),zorder=3)
        
    # colorbar
    if cbar:
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.tick_params(labelsize=25)
    
    # title
    if title is not None:
        plt.title(title,fontsize=30,pad=25)
    
    # save file? Only plot it if output_file is None
    if output_file is not None:
        plt.savefig(output_file,bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    return 
    
