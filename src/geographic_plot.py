import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pylab import meshgrid
import os 

import iris
import iris.plot as iplt
import iris.quickplot as qplt
from iris.time import PartialDateTime

from temperature_analysis import region_based_cube

def select_time(month_start: int, month_end: int, year_start: int, year_end: int, cube):
    """
    Query data in the given time range.
    Args: 
        month_start, month_end, year_start, year_end: input time range
        cube: input data cube
    Return: the cube after filtering
    """
    month_start = PartialDateTime(month = month_start)
    month_end = PartialDateTime(month = month_end)
    year_start = PartialDateTime(year = year_start)
    year_end = PartialDateTime(year = year_end)
    part_temp = cube.extract(iris.Constraint(time=lambda x : month_start<=x<=month_end) & \
                            iris.Constraint(time=lambda x : year_start<=x<=year_end))
    return part_temp

def process_cube(cube):
    """
        Do several steps of data processing(filtering time, filtering region, getting average).
        Provide data for plotting.
        Args:
            cube: the input data
        Returns: 
            Cube after processing
    """
    # filter by time
    cube = select_time(5, 7, 1992, 2015, cube)

    # filter by region, get the map of US
    # coordinates of the US
    coords = [24, 72, -170, -65] 
    cube = region_based_cube(cube, coords)

    # calculate average temperature
    mean_temp = cube.collapsed(['time'],iris.analysis.MEAN)

    return mean_temp

def plot_temperature_us(cube):
    """
    Plotting a geographic map from the given cube
    Args:
        cube: an iris cube object, representing data
    Returns:
        None. Just plotting.
    """
    # setting font family and font size
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 15
        }
    mpl.rc('font', **font)

    # get latitude and longitude points
    lat = cube.coord('latitude').points
    lon = cube.coord('longitude').points

    # draw background of the map
    f = plt.figure(figsize=[15,8])
    m = Basemap(projection='mill',\
    llcrnrlon=lon.min(),urcrnrlon=lon.max(), \
    llcrnrlat=lat.min(),urcrnrlat=lat.max(),)
    m.drawcoastlines(linewidth=0.25)
    m.drawcountries(linewidth=0.85,color='black')
    m.drawstates(linewidth=0.15)
    m.drawmapboundary()
    m.drawparallels(np.arange(0.,90.,20),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(-160,-70,20),labels=[0,0,0,1],fontsize=10)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='lightcyan',lakes=True,zorder=1)

    Lon,Lat = meshgrid(lon,lat)
    x, y = m(Lon,Lat)
    cs = m.pcolormesh(x,y,cube.data,shading='flat',cmap='viridis')

    # adjust the height of the label to be the same as the map
    cbar= plt.colorbar( fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (K)", labelpad=+1)
    # plt.savefig('./us_temp.png', dpi=400)
    plt.show()

    
if __name__ == "__main__":
    """
    For plotting rainfall/vegetation data, almost everything remains the same, except
    some tiny differences. For further reference, just check my ipython notebook called "geographic_map.ipynb"
    """

    # load data from directory
    path = '../../air.mon.mean.nc'
    cube = iris.load_cube(path)
    cube = process_cube(cube)
    plot_temperature_us(cube)



