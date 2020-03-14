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

def region_based_cube(cube,coords):
    '''
    Inputs:
        cube with latitude and longitude as coords
        coords: the region we want to extract from input cube
    Outputs: 
        A smaller cube within coords
    '''
    lat_min,lat_max,long_min,long_max = coords
    lat_cons = iris.Constraint(latitude = lambda x : lat_min < x < lat_max)
    if (long_min<0):
        long_min = long_min+360
    if(long_max <0) :
        long_max = long_max+360
    long_cons = iris.Constraint(longitude = lambda x : long_min < x < long_max)
    new_cube = cube.extract(lat_cons & long_cons)
    return new_cube

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

def process_cube_us(path):
    """
        Do several steps of data processing(filtering time, filtering region, getting average).
        Prepare data for plotting.
        Args:
            path: the input dataset path
        Returns: 
            A cube just focusing on the average data in US in the part several years.
    """
    # load dataset
    cube = iris.load_cube(path)

    # filter by time
    cube = select_time(5, 7, 1992, 2015, cube)

    # filter by region, get the map of US
    # coordinates of the US
    coords = [24, 72, -170, -65] 
    cube = region_based_cube(cube, coords)

    # calculate average temperature
    mean_temp = cube.collapsed(['time'],iris.analysis.MEAN)

    return mean_temp

def plot_geometric_map(cube, label_name, color):
    """
    Plotting a geographic map from the given cube
    Args:
        cube: an iris cube object, representing data
        label_name: name of the plot
        color: name of the color map
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
    cs = m.pcolormesh(x,y,cube.data,shading='flat',cmap=color)

    # adjust the height of the label to be the same as the map
    cbar= plt.colorbar( fraction=0.046, pad=0.04)
    cbar.set_label(label_name, labelpad=+1)
    # plt.savefig('./us_temp.png', dpi=400)
    plt.show()

    
if __name__ == "__main__":
    """
    For plotting rainfall/vegetation data, almost everything remains the same, except
    some tiny differences. For further reference, just check my ipython notebook called "geographic_map.ipynb"
    """

    cube = process_cube_us('../../air.mon.mean.nc')
    plot_geometric_map(cube,"Temperature (K)",'viridis')

    cube = process_cube_us('../../precip.mon.mean.nc')
    plot_geometric_map(cube,"Rainfall (mm/day)", plt.cm.YlGnBu)



