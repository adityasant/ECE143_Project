import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import os 
import iris
import iris.plot as iplt
import iris.quickplot as qplt
from iris.time import PartialDateTime

import matplotlib as mpl

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}
mpl.rc('font', **font)



def extract_fire_gas(fire_data, gas_data):
    '''
    :param fire_data: fire data path
    :param gas_data: gas data path
    :return: data frame of fire + gas
    '''
    conn = sqlite3.connect(fire_data)
    data_dates = pd.read_sql_query("SELECT fire_year, discovery_date FROM fires;", conn)
    data_size = pd.read_sql_query("SELECT fire_size, fire_size_class FROM fires;", conn)
    data_dates['FIRE_SIZE'] = data_size['FIRE_SIZE']
    
    emis = pd.read_csv(gas_data, index_col='year', thousands=",")
    df = emis.loc[emis['country_or_area']=='United States of America'].pivot_table(index='year',
                                                                             columns='category',
                                                                             values='value')
    df.rename(columns={'carbon_dioxide_co2_emissions_without_land_use_land_use_change_and_forestry_lulucf_in_kilotonne_co2_equivalent':'CO2',
                   'greenhouse_gas_ghgs_emissions_including_indirect_co2_without_lulucf_in_kilotonne_co2_equivalent':'GreenHouse Gas1',
                   'greenhouse_gas_ghgs_emissions_without_land_use_land_use_change_and_forestry_lulucf_in_kilotonne_co2_equivalent':'GreenHouse Gas2',
                   'hydrofluorocarbons_hfcs_emissions_in_kilotonne_co2_equivalent':'HFC',
                   'methane_ch4_emissions_without_land_use_land_use_change_and_forestry_lulucf_in_kilotonne_co2_equivalent':'CH4',
                   'nitrogen_trifluoride_nf3_emissions_in_kilotonne_co2_equivalent':'NF3',
                   'nitrous_oxide_n2o_emissions_without_land_use_land_use_change_and_forestry_lulucf_in_kilotonne_co2_equivalent':'N2O',
                   'perfluorocarbons_pfcs_emissions_in_kilotonne_co2_equivalent':'PFC',
                   'sulphur_hexafluoride_sf6_emissions_in_kilotonne_co2_equivalent':'SF6'},
              inplace=True)
    df.drop(columns=['GreenHouse Gas1','GreenHouse Gas2'],inplace=True)
    df.drop(columns=['unspecified_mix_of_hydrofluorocarbons_hfcs_and_perfluorocarbons_pfcs_emissions_in_kilotonne_co2_equivalent'],
            inplace=True)
    
    fire = data_dates.groupby(by='FIRE_YEAR',as_index=True).sum()
    df_show = pd.merge(left=df,right=fire[['FIRE_SIZE']],left_index=True,right_index=True,how='inner')
    df_show.rename(columns={'FIRE_SIZE':'Fire Size'},inplace=True)
    df_show = df_show[['Fire Size', 'CO2', 'HFC', 'CH4', 'NF3', 'N2O', 'PFC', 'SF6']]
    
    return df_show


def fire_gas_heatmap(df_show):
    '''
    :param df_show: data frame gained by extract_fire_gas_data()
    :return: no
    '''
    df_show.rename(columns={'FIRE_SIZE':'Fire Size'},inplace=True)
    df_show = df_show[['Fire Size', 'CO2', 'HFC', 'CH4', 'NF3', 'N2O', 'PFC', 'SF6']]
    corr = df_show.corr()
    
    
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(13, 10))
        ax = sns.heatmap(corr,
                         annot=True,
                         #mask=mask,
                         vmin=-2, vmax=2,
                         square=True,
                         cmap=sns.color_palette("RdBu_r",100) )
        ax.set_title('GreenHouse Gas and Wild Fire\n Correlation in Heat-Map')
        plt.xticks(rotation=45)
        plt.yticks(rotation=-45)

def extract_fire_sample(fire_data, n, state):
    '''
    :param dire_data: fire data path
    :param n: number of samples
    :param state: string 'ALL' or 'AK', represent all US or only AK, (when using 'AK', n does not matter because we will take all samples)
    :return: sample data frame containing it
    '''
    conn = sqlite3.connect(fire_data)
    data_size = pd.read_sql_query("SELECT fire_size, fire_size_class FROM fires;", conn)
    data_location = pd.read_sql_query("SELECT latitude, longitude, state FROM fires;", conn)
    date = pd.read_sql_query("select datetime(DISCOVERY_DATE) as DISCOVERY_DATE from fires;", conn)
    
    data_dates_arr = date['DISCOVERY_DATE']
    size_class = []
    for c in data_size['FIRE_SIZE_CLASS']:
        d = {"A":1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
        size_class.append(d[c])
    df = data_location
    df['date'] = data_dates_arr
    df['fire_class'] = size_class
    
    if state == 'ALL':
        sample_df = pd.DataFrame()
        num_samples = n
        for i in range(1, 8):
            s = df[df['fire_class'] == i].sample(n = num_samples)
            sample_df = pd.concat([sample_df, s])
    elif state == 'AK':
        sample_df = df[df['STATE']=='AK']
    
    return sample_df


def query_wind_pres(y, m, lat, long, cube_w, cube_p):
    """
    Args:
        year: integer indicate year, support 1979 to 2019
        month: integer indicate month, 1 to 12
        latitude: float, -90 to 90, resolution 0.25
        longitude: float, 0 to 360, 5 means East5, 185 means West175, resolution 0.25
        cube: precipation data
    Return:
        the average precipation in that given month (mm/day)
        its type is float  
    """
    if long < 0:
        long += 360
    time_cons = iris.Constraint(time = PartialDateTime(year = y, month = m))
    lat_cons = iris.Constraint(latitude = lambda x : lat-0.125 <= x <lat+0.125)
    long_cons = iris.Constraint(longitude = lambda x : long-0.125 <= x <long+0.125)

    wind = cube_w.extract(lat_cons & long_cons & time_cons)
    pres = cube_p.extract(lat_cons & long_cons & time_cons)
    
    return (float(wind.data), float(pres.data))


def extract_wind_air(sample_df, wind_air_data):
    '''
    :param sample_df: data frame to change
    :param wind_air_data: wind and air data path
    :return: sample data frame with 2 more columns
    '''
    raw = iris.load_raw(wind_air_data)
    wind_series = []
    pres_series = []
    i=0
    for index, row in sample_df.iterrows():
        
        if np.mod(i, len(sample_df)//5)==0:
            print('Extracting wind/air on: %d/%d'%(i,len(sample_df)))
        i += 1
        
        date_list = row['date'].split('-')
        year = int(date_list[0])
        month = int(date_list[1])
        lat = row['LATITUDE']
        long = row['LONGITUDE']
        data = query_wind_pres(year, month, lat, long, raw[0],raw[1])
        wind_series.append(data[0])
        pres_series.append(data[1])
    
    sample_df['wind'] = wind_series
    sample_df['air_pressure'] = pres_series
    
    return sample_df

def query_rain(y, m, lat, long, cube):
    """
    Args:
        year: integer indicate year, support 1979 to 2019
        month: integer indicate month, 1 to 12
        latitude: float, -90 to 90
        longitude: float, 0 to 360, 5 means East5, 185 means West175
        cube: precipation data
    Return:
        the average precipation in that given month (mm/day)
        its type is float  
    """
    if long < 0:
        long += 360
    time_cons = iris.Constraint(time = PartialDateTime(year = y, month = m))
    lat_cons = iris.Constraint(latitude = lambda x : lat-1.25 <= x <lat+1.25)
    long_cons = iris.Constraint(longitude = lambda x : long-1.25 <= x <long+1.25)
    rain = cube.extract(lat_cons & long_cons & time_cons)
    return float(rain.data)


def extract_rain(sample_df, rain_data):
    '''
    :param sample_df: data frame to change
    :param rain_data: rain data path
    :return: sample data frame with 1 more columns
    '''
    cube = iris.load_cube(rain_data)
    rain_series = []
    i=0
    for index, row in sample_df.iterrows():
        
        if np.mod(i, len(sample_df)//5)==0:
            print('Extracting rain on: %d/%d'%(i,len(sample_df)))
        i += 1
        
        date_list = row['date'].split('-')
        year = int(date_list[0])
        month = int(date_list[1])
        lat = row['LATITUDE']
        long = row['LONGITUDE']
        data = query_rain(year, month, lat, long, cube)
        rain_series.append(data)
    
    sample_df['rain'] = rain_series
    
    return sample_df

def extract_temp(sample_df, temp_data):
    '''
    :param sample_df: data frame to change
    :param temp_data: temp data path
    :return: sample data frame with 1 more columns
    '''
    cube = iris.load_cube(temp_data)
    temp_series = []
    i=0
    for index, row in sample_df.iterrows():
        
        if np.mod(i, len(sample_df)//5)==0:
            print('Extracting temperature on: %d/%d'%(i,len(sample_df)))
        i += 1
        
        date_list = row['date'].split('-')
        year = int(date_list[0])
        month = int(date_list[1])
        lat = row['LATITUDE']
        long = row['LONGITUDE']
        data = query_rain(year, month, lat, long, cube)
        temp_series.append(data)
    
    sample_df['temperature'] = temp_series
    
    return sample_df


def change_col_name(sample_df, fire_data):
    '''
    :param sample_df: data frame to change
    :param fire_data: fire data path
    :return: changed columns name of sample_df
    '''
    conn = sqlite3.connect(fire_data)
    data_size = pd.read_sql_query("SELECT fire_size, fire_size_class FROM fires;", conn)
    df2 = data_size.drop(columns=['FIRE_SIZE_CLASS'])
    sample_df_show = pd.merge(left=sample_df, right=df2, left_index=True, right_index=True, how='inner')

    sample_df_show = sample_df_show[['LATITUDE','LONGITUDE','fire_class','FIRE_SIZE','wind','air_pressure','rain','temperature']]
    sample_df_show.rename(columns={'LATITUDE':'Latitude',
                                   'LONGITUDE':'Longitude',
                                  'fire_class':'Fire Class',
                                  'FIRE_SIZE':'Fire Size',
                                  'wind':'Wind',
                                  'air_pressure':'Air Pressure',
                                  'rain':'Rain',
                                  'temperature':'Temperature'},
                                  inplace=True)
    return sample_df_show


def scatter_corr(sample, col1, col2, threshold):
    df = sample.loc[sample['Fire Class']>=threshold[0]].loc[sample['Fire Class']<=threshold[1]]

    x = df[col1].values
    y = df[col2].values
    
    plt.style.use('seaborn-whitegrid')
    f, ax = plt.subplots(figsize=(7, 5))
    
    plt.scatter(x , y, s=1.5)
    plt.title('Correlation in Scatter')
    plt.xlabel(col1)
    plt.ylabel(col2)
    

def heatmap_corr(sample, threshold):
    df = sample.loc[sample['Fire Class']>=threshold[0]].loc[sample['Fire Class']<=threshold[1]]
    corr = df.corr()
    
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(11, 10))
        ax = sns.heatmap(corr,
                         annot=True,
                         #mask=mask,
                         vmin=-1, vmax=1,
                         square=True,
                         cmap=sns.color_palette("RdBu_r", 100) )
        ax.set_title('US Mainland\n Correlation in Heat-Map')
        plt.xticks(rotation=45)
    return corr

    
if __name__ == "__main__":
    '''
    Code to plot wildfire basemap
    For a more detailed code, refer to Final_correlation.ipynb in Local Notebooks
    '''
    
    # fire and gas
    fire_data = '../Datasets/FPA_FOD_20170508.sqlite'
    gas_data = '../Datasets/greenhouse_gas_inventory_data_data.csv'
    
    df = extract_fire_gas(fire_data, gas_data)
    fire_gas_heatmap(df)
    
    # fire and weather
    # use state='AK' for only Alaska, then you can choose whatever n because we are choosing all AK samples
    #df = correlation.extract_fire_sample(fire_data, n=1, state='AK')
    # use state='ALL' for all US, then you should choose n(like 500 in our PRE) as number of samples
    df2 = extract_fire_sample(fire_data, n=100, state='ALL')
    
    wind_air_data = '../Datasets/adaptor.mars.internal.nc'
    df2 = extract_wind_air(df2, wind_air_data)

    rain_data = '../Datasets/precip.mon.mean.nc'
    df2 = extract_rain(df2, rain_data)

    temp_data = '../Datasets/air.mon.mean.nc'
    df2 = extract_temp(df2, temp_data)
    
    df2 = change_col_name(df2, fire_data)
    # scatter correlation
    scatter_corr(df2, col1='Rain', col2='Fire Size', threshold=[2,6])
    
    # heatmap correlation
    heat = heatmap_corr(df2, threshold=[2,6])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
