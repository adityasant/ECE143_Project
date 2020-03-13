import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15
       }

mpl.rc('font', **font)
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap


def extract_wildfire_data(data_path):
    '''
    Function to extract wildfire data from the dataset
    The fires extracted are exclusively for class G (>5000 hectares)
    Fire fields: size, latitude, longitude

    :param: String for the data path
    :return: dataframe with the wildfire information
    '''

    # Read the data
    conn = sqlite3.connect(data_path)
    fire_data = \
        pd.read_sql_query("SELECT fire_size, fire_size_class, fire_year , \
            latitude, longitude FROM fires;", conn)
    fire_data_grps = fire_data.groupby('FIRE_SIZE_CLASS')

    return fire_data_grps.get_group('G')


def plot_size_hist(wildfire_data):
    '''
    Plot the histogram for the fire data
    The y-axis is plotted in log scale

    :params: Wildfire data frame
    :return: fig object
    '''

    fig,ax = plt.subplots(figsize=(10, 5))
    ax.hist(wildfire_data['FIRE_SIZE'],bins=20,edgecolor='white', linewidth=1);
    ax.set_yscale('log')
    ax.set_xlabel('Area in hectares',fontsize=15);
    ax.set_ylabel('Num of Fires');
    ax.set_title('Distribution of Wildfires from 1992 - 2015');
    plt.show()

    return fig

def plot_increasing_trend(wildfire_data):
    '''
    Plot the increasing trend of the fires

    :param: wildfire dataframe
    :return: fig object
    '''

    n, bins, patches = plt.hist(wildfire_data['FIRE_YEAR'],bins=24,edgecolor='white', linewidth=1)

    np.reshape(bins[1:],[-1,3])
    np.reshape(n,[-1,3])

    fig,ax = plt.subplots(figsize=(15, 7))
    ax.plot(np.reshape(n,[-1,3]).sum(1),'D-b',markersize=12)
    ax.set_xticklabels(['1','1992-1995','1995-1998','1998-2001','2001-2004','2004-2007','2007-2010',\
                        '2010-2013','2013-2016']);
    ax.set_ylabel('Number of large fires (>5000 hectares)',fontsize=15);
    ax.set_xlabel('Year range');
    ax.grid()
    plt.show()

    return fig



def plot_fire_map(wildfire_data):
    '''
    Plot to show the geographic distribution of the fire data
    Takes the input as the lower and upper limit of the fire size

    :param: wildfire dataframe, basemap object, int, int
    :return: fig object
    '''
    plt.subplots(figsize=[15,8])
    map = Basemap(projection='cyl',llcrnrlat=24,urcrnrlat=72,\
            llcrnrlon=-170,urcrnrlon=-65,)
    map.drawlsmask(ocean_color='lightcyan',lakes=True,zorder=1)
    map.fillcontinents(color='darkgrey',zorder=2,lake_color='lightcyan')
    map.drawcoastlines(linewidth=0.75,zorder=3)
    map.drawcountries(linewidth=0.85,color='black',zorder=3)
    map.drawstates(linewidth=0.15,zorder=3)
    #map.drawmapboundary()
    map.drawparallels(np.arange(0.,90.,20),labels=[1,0,0,0],fontsize=10,zorder=3)
    map.drawmeridians(np.arange(-160,-90,20),labels=[0,0,0,1],fontsize=10,zorder=3)

    ## Large fires
    lim_low = 65000
    lim_high = 3000000
    w1 = wildfire_data.loc\
        [(wildfire_data['FIRE_SIZE']<lim_high) & (wildfire_data['FIRE_SIZE']>=lim_low)]
    long_temp = w1['LONGITUDE']
    lat_temp = w1['LATITUDE']

    scat = map.scatter(long_temp, lat_temp,s=10);
    scat.set_color('r')
    scat.set_alpha(0.65)
    scat.set_zorder(12)

    # Medium fires
    lim_low = 30000
    lim_high = 65000
    w1 = wildfire_data.loc\
        [(wildfire_data['FIRE_SIZE']<lim_high) & (wildfire_data['FIRE_SIZE']>=lim_low)]
    long_temp = w1['LONGITUDE']
    lat_temp = w1['LATITUDE']

    scat = map.scatter(long_temp, lat_temp,s=10);
    scat.set_color('orange')
    scat.set_alpha(0.95)
    scat.set_zorder(11)

    ## Small fires
    lim_low = 10000
    lim_high = 30000
    w1 = wildfire_data.loc\
        [(wildfire_data['FIRE_SIZE']<lim_high) & (wildfire_data['FIRE_SIZE']>=lim_low)]
    long_temp = w1['LONGITUDE']
    lat_temp = w1['LATITUDE']

    scat = map.scatter(long_temp, lat_temp,s=10);
    scat.set_color('gold')
    scat.set_alpha(1)
    scat.set_zorder(10)

    ## Small fires
    lim_low = 5000
    lim_high = 10000
    w1 = wildfire_data.loc\
        [(wildfire_data['FIRE_SIZE']<lim_high) & (wildfire_data['FIRE_SIZE']>=lim_low)]
    long_temp = w1['LONGITUDE']
    lat_temp = w1['LATITUDE']

    scat = map.scatter(long_temp, lat_temp,s=10);
    scat.set_color('yellow')
    scat.set_alpha(0.75)
    scat.set_zorder(9)

    
    plt.show()

    return scat






if __name__ == "__main__":
    '''
    Code to plot wildfire basemap

    For a more detailed code, refer to wildfire_datacleaning.ipynb in Local Notebooks
    '''
    
    # Extract data
    fire_data = extract_wildfire_data('../Datasets/FPA_FOD_20170508.sqlite')
    
    # Plot histogram
    fd_hist = plot_size_hist(fire_data)

    # Plot increasing trend
    fd_inc = plot_increasing_trend(fire_data)
    
    # Plot geographical map
    fd_map = plot_fire_map(fire_data)


