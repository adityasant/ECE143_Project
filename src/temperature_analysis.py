import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import datetime
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pylab import *
from iris.pandas import as_cube, as_series, as_data_frame
import iris   
import iris.plot as iplt
import iris.quickplot as qplt
import iris.coord_categorisation as cat
from iris.analysis import Aggregator
import matplotlib as mpl
from iris.time import PartialDateTime

def create_wildfire_df(PATH):
    '''
    PATH: Linux path to sqlite for wildfire database
    Output:
        dataframe: of wildfire with the following columns:
        LATITUDE
        LONGITUDE
        STATE
        date
        fire_size
        fire_year
        fire_month
        fire_total (This is total fire in a particular latitude, longitude, fire_year and fire_month)
    '''
    assert(os.path.exists(PATH)),"Incorrect file path"
    conn = sqlite3.connect(PATH)
    data_dates = pd.read_sql_query("SELECT fire_year, discovery_date FROM fires;", conn)
    data_size = pd.read_sql_query("SELECT fire_size, fire_size_class FROM fires;", conn)
    data_location = pd.read_sql_query("SELECT latitude, longitude, state FROM fires;", conn)
    date = pd.read_sql_query("select datetime(DISCOVERY_DATE) as DISCOVERY_DATE from fires;", conn)
    data_dates_arr = date['DISCOVERY_DATE']
    fire_year_arr = data_dates['FIRE_YEAR']
    fire_size_arr = data_size['FIRE_SIZE']  
    #Create a dataframe df 
    df = data_location
    df['date'] = data_dates_arr
    df['fire_size'] = fire_size_arr
    df['fire_year'] = fire_year_arr
    df['fire_month'] = pd.DatetimeIndex(df['date']).month
    #Converted Lat and Long to int to remove the exact precision
    df['LATITUDE'] = df['LATITUDE'].round(2)
    df['LONGITUDE'] = df['LONGITUDE'].round(2)
##Compute total fire based on Latitude, Longitude, fire_year and fire_month
    df['fire_total'] = df.groupby(['LATITUDE','LONGITUDE', 'fire_year','fire_month'])['fire_size'].transform(sum)
    
    return df


    
def large_fire_coord(state, fire_size, df):
    '''
    Function: Extact locations for very large wildfires
    Input: 
        state: The state we want to focus on.
        df: Input dataframe
        fire_size: fire size above which is considered large
    Output:
        lat_min,lat_max,long_min,long_max: the coordinates with fires above fire_size in state 'state'
    '''
    #print(state)
    df = df.query("fire_total >= fire_size and STATE == @state")
    lat_max = df['LATITUDE'].max()
    lat_min = df['LATITUDE'].min()
    long_max = df['LONGITUDE'].max()
    long_min = df['LONGITUDE'].min() 
    return lat_min,lat_max,long_min,long_max
    
    
def process_df(coords, fire_size, df,year):
    '''
    Input: 
        coords: A tuple of lat_min,lat_max,long_min,long_max in which we need to find wild fires
        Find sum of fire occuring within the location, for a given year and month
        output: (fire_sz,latitudes,longitudes), with area burnt in a particular location, fire size larger than  a large fire, and particular fire year
    '''
    lat_min,lat_max,long_min,long_max = coords
    print("Inital", len(df))
    #print(lat_min,lat_max,long_min,long_max)
    df_new = df.query("LATITUDE >= @lat_min and LATITUDE <= @lat_max and LONGITUDE >= @long_min and LONGITUDE <= @long_max and fire_total>=@fire_size and fire_year == @year")
    print("Particular Year & location", len(df_new))
    df_new['fire_total_month'] = df_new.groupby(['LATITUDE','LONGITUDE', 'fire_year'])['fire_total'].transform(sum)    
    df_new = df_new.sort_values('fire_total', ascending=False).drop_duplicates(['fire_total_month'])
    print("Sum over months in year", len(df_new))
    fire_sz= np.array(df_new['fire_total_month'].to_list())
    lat_fire = np.array(df_new['LATITUDE'].tolist())
    lon_fire = np.array(df_new['LONGITUDE'].tolist())+360
    return fire_sz,lat_fire,lon_fire
    
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
    
def time_based_cube(cube,year):
    time_cons = iris.Constraint(year = year)
    new_cube = cube.extract(time_cons)
    return new_cube
    
##Adding auxilary axis:
def get_decade(coord, value):
    date = coord.units.num2date(value)
    return date.year - date.year % 10
def get_year(coord, value):
    date = coord.units.num2date(value)
    return date.year

def get_month(coord, value):
    date = coord.units.num2date(value)
    return date.month
    
##Get the cube for a particular year. It takes maximum temperature seen in that year
def get_cube_data(cube,year):
    cube_time = time_based_cube(cube,year)
    cube_mean = cube_time.collapsed(['month'], iris.analysis.MAX)
    lat = cube_mean.coord('latitude').points
    lon = cube_mean.coord('longitude').points
    return cube_mean.data,lat,lon
    
##Time Analysis

def process_df_local(coords, df):
    '''
    Input: 
        coords: A tuple of lat_min,lat_max,long_min,long_max in which we need to find wild fires
        Find sum of fire occuring within the location, for a given year and month
    
    '''
    lat_min,lat_max,long_min,long_max = coords
    #print(lat_min,lat_max,long_min,long_max)
    df_new = df.query("LATITUDE >= @lat_min and LATITUDE <= @lat_max and LONGITUDE >= @long_min and LONGITUDE <= @long_max")
    #print(df_new)
    df_new['fire_total_area'] = df_new.groupby(['fire_year','fire_month'])['fire_total'].transform(sum)
    df_new = df_new.sort_values('fire_total_area', ascending=False).drop_duplicates(['fire_year','fire_month'])
    df_final = df_new[['fire_year','fire_month','fire_total_area']]
    return df_final
    
def create_plt(month,result):
    '''
    Inputs: Month for which plot is needed
    result: A dataframe containing temperature and fire_total_area
    '''
    temp = result[result['month'] == month]['Temperature'].tolist()
    area = result[result['month'] == month]['fire_total_area'].tolist()
    m_temp = [mean(temp)]*len(temp)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_ylabel('Area')
    ax1.set_xlabel('Years')
    ax1.set_ylim([0,5000000])
    ax1.plot(area,color='r',label='Area')
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temp")
    #ax2.set_ylim(270,300)
    ax2.plot(temp,label="Temp")
    ax2.plot(m_temp,label="Mean")
    fig.tight_layout()
    plt.legend()
    plt.show()
#plt.close

def cube_to_df(cube_max):
    df = as_data_frame(cube_max, copy=True)
    df = df.reset_index()
    df['month'] = pd.DatetimeIndex(df['index']).month
    df['year'] = pd.DatetimeIndex(df['index']).year
    df.drop('index',axis=1,inplace=True)
    df = df.rename(columns={0:"Temperature"})
    df = df.query("year >= 1992 and year <=2015")

    return df
   
def create_map(year,coords,cube,fire_df):
    '''
    Creates a map of Alaska. With temperature gradients. And wildfire scale
    Input: year for which analysis needs to be done
    Output: A Map
    '''
    ##Plot showing temperature with wildfire size for 2004. 
    ##Size of circles denote and color denote area burnt due to wildfire.
    #%matplotlib notebook
    
    plt.rcParams['font.weight'] = 'black'
    cmap = mpl.cm.Reds(np.linspace(0,1,20))
    cmap = mpl.colors.ListedColormap(cmap[15:,:-1])
    #def create_map(year):
    year = 2004
    plt.figure(figsize=[15,8])
    #fig, ax = plt.subplots(figsize=(10, 4))
    #fig, ax = plt.subplots()
    m=Basemap(projection='mill',lat_ts=10, \
      llcrnrlon=coords[2]+360,urcrnrlon=coords[3]+360, \
      llcrnrlat=coords[0],urcrnrlat=coords[1], \
      resolution='c')
    m.drawcoastlines(linewidth=0.85)
    m.drawcountries(linewidth=0.85,color='red')
    data,lat,lon=get_cube_data(cube,year)
    Lon,Lat = meshgrid(lon,lat)
    x, y = m(Lon,Lat)
    cs = m.pcolormesh(x,y,data,shading='flat',cmap='summer',vmin=275,vmax=292)
    #print("CS..",type(cs))
    cbar= plt.colorbar()
    cbar.set_label("Temperature (K)", labelpad=+2,fontdict={'fontsize': 12, 'fontweight': 'black'})
    fire_sz,lat_fire,lon_fire = process_df(coords,100,fire_df,year)
    a,b = m(lon_fire,lat_fire)
    scat = m.scatter(lon_fire, lat_fire, s=fire_sz/1500, latlon=True,
             c=fire_sz,cmap=cmap,
              alpha=1.2, vmin=90000,vmax=550000)

    plt.show()

def plot_rainfall(st_month,end_month,year,rain_cube,fire_df,coords):
    '''
    Inputs: st_month: Intial rain month
            end_month: Final rain month
            rain_cube: A cube with rain data
    '''
    month_start = PartialDateTime(month = st_month)
    month_end = PartialDateTime(month = end_month)
    #year_start = PartialDateTime(year = 1992)
    year_end = PartialDateTime(year = year)
    part_rain = rain_cube.extract(iris.Constraint(time=lambda x : month_start<=x<=month_end) & \
                             iris.Constraint(time=year_end))
    #print(part_rain)
    mean_rain = part_rain.collapsed(['time'],iris.analysis.MEAN)
    us_rain = region_based_cube(mean_rain, coords)
    lat = us_rain.coord('latitude').points
    lon = us_rain.coord('longitude').points
    cmap = mpl.cm.Reds(np.linspace(0,1,20))
    cmap = mpl.colors.ListedColormap(cmap[15:,:-1])
    plt.figure(figsize=[15,8])
    m=Basemap(projection='mill',lat_ts=10, \
      llcrnrlon=lon.min(),urcrnrlon=lon.max(), \
      llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
      resolution='c')
    # m=Basemap(projection='mill',llcrnrlat=25,urcrnrlat=50,\
    #             llcrnrlon=-125,urcrnrlon=-80,)
    m.drawcoastlines(linewidth=0.75)
    m.drawcountries(linewidth=0.85,color='red')
    m.drawstates(linewidth=0.15)
    m.drawmapboundary()
    m.drawparallels(np.arange(0.,90.,20),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(-160,-70,20),labels=[0,0,0,1],fontsize=10)
    # m.fillcontinents(color='lightgrey', zorder=1,lake_color='aqua')
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='white',lakes=True,zorder=1)

    Lon,Lat = meshgrid(lon,lat)
    x, y = m(Lon,Lat)
    cs = m.pcolormesh(x,y,us_rain.data,shading='flat',cmap=plt.cm.YlGnBu,vmin=0,vmax=3.5)
    cbar= plt.colorbar()
    #cbar.set_label("Rainfall (mm/day)", labelpad=+1)
    cbar.set_label("RAINFALL (mm/day)", labelpad=+2,fontdict={'fontsize': 12, 'fontweight': 'black'})
    fire_sz,lat_fire,lon_fire = process_df(coords,100,fire_df,year)
    a,b = m(lon_fire,lat_fire)
    scat = m.scatter(lon_fire, lat_fire, s=fire_sz/1500, latlon=True,
             c=fire_sz,cmap=cmap,
              alpha=1.2, vmin=90000,vmax=550000)
    plt.show()    


if __name__ == "__main__":
    '''
    Code to plot Alaska case study
    For a more detailed code, refer to main_notebook.ipynb
    '''
    
    # create data
    PATH = '../Datasets/FPA_FOD_20170508.sqlite'
    fire_df = create_wildfire_df(PATH)
    
    coords = large_fire_coord('AK',10000,fire_df)
    
    PATH_temp = "../Datasets/air.mon.mean.nc"
    cube_temp = iris.load_cube(PATH_temp)
    cat.add_categorised_coord(cube_temp, 'year', 'time', get_year)
    cat.add_categorised_coord(cube_temp, 'month', 'time', get_month)
    cube_local = region_based_cube(cube_temp,coords)
    
    # geometric temperature
    create_map(2004,coords,cube_local,fire_df)
    
    # trend analysis
    fire_time_based = process_df_local(coords,fire_df)
    fire_time_based = fire_time_based.rename(columns={"fire_year": "year", "fire_month": "month"})
    PATH_temp = "../Datasets/air.mon.mean.nc"
    cube_temp = iris.load_cube(PATH_temp)
    cube_local = region_based_cube(cube_temp,coords)
    
    cube_max = cube_local.collapsed(['latitude','longitude'], iris.analysis.MAX)
    df = cube_to_df(cube_max)
    result = pd.merge(df, fire_time_based, on=['year', 'month'],how='left')
    result.fillna(0,inplace=True)
    create_plt(6,result)
    
    # geometric rainfall
    PATH_rain = "../Datasets/precip.mon.mean.nc"
    rain_cube = iris.load_cube(PATH_rain)
    plot_rainfall(5,7,2004,rain_cube,fire_df,coords)
    
    

