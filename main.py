'''
Main script to run the entire visualization and correlation plots.

This has different parts to do the individual functionality

'''


'''
This part deals with extracting the wildfire data and plotting the histograms and geographical plots

extract_wildfire_data - Construct the wildfires dataframe
plot_size_hist - Show the distribution of fire sizes
plot_increasing_trend - Plot the variation of sizes of the fires over time
plot_fire_map - Geographically show the locations and information of the fires
'''

wildfire_path = './Datasets/FPA_FOD_20170508.sqlite'
from src.wildfire_plots import extract_wildfire_data, plot_size_hist, plot_increasing_trend, plot_fire_map


fire_data = extract_wildfire_data(wildfire_path)
fd_hist = plot_size_hist(fire_data)
fd_inc = plot_increasing_trend(fire_data)
fd_map = plot_fire_map(fire_data)


'''
This part contains the code to show the geographic map of the climate and vegetation features

process_cube_us - Do several steps of data processing(filtering time, filtering region, getting average) and prepare for plotting
plot_geometric_map - Plotting a geographic map from the given cube

'''

import src.geographic_plot as geo
import matplotlib.pyplot as plt

path = './Datasets/air.mon.mean.nc'

cube = geo.process_cube_us(path)
geo.plot_geometric_map(cube,"Temperature (K)",'viridis')
cube = geo.process_cube_us('./Datasets/precip.mon.mean.nc')
geo.plot_geometric_map(cube,"Rainfall (mm/day)", plt.cm.YlGnBu)



'''
This part is used to extract the data and plot the correlation

extract_fire_gas - data frame of fire + gas
fire_gas_heatmap - Correlation of data frame gained by extract fire and gas data
extract_fire_sample - sample data frame containing it
query_wind_pres - the average precipation in that given month (mm/day)
extract_wind_air - 
query_rain - the average precipation in that given month (mm/day)

'''
import src.correlation_plots as correlation
fire_data = './Datasets/FPA_FOD_20170508.sqlite'
gas_data = './Datasets/greenhouse_gas_inventory_data_data.csv'

df = correlation.extract_fire_gas(fire_data, gas_data)
correlation.fire_gas_heatmap(df)

fire_data = './Datasets/FPA_FOD_20170508.sqlite'
df = correlation.extract_fire_sample(fire_data, n=100, state='ALL')

wind_air_data = './Datasets/adaptor.mars.internal.nc'
df = correlation.extract_wind_air(df, wind_air_data)

rain_data = './Datasets/precip.mon.mean.nc'
df = correlation.extract_rain(df, rain_data)

temp_data = './Datasets/air.mon.mean.nc'
df = correlation.extract_temp(df, temp_data)

df = correlation.change_col_name(df, fire_data)

correlation.scatter_corr(df, col1='Rain', col2='Fire Size', threshold=[2,6])

heat = correlation.heatmap_corr(df, threshold=[2,6])




'''
This script plots the temperature and rainfall specifically for Alaska

create_wildfire_df - create wildfire dataframe
large_fire_coord - Extact locations for very large wildfires
process_df - Find sum of fire occuring within the location, for a given year and month
process_df_local - Find sum of fire occuring within the location, for a given year and month
create_plt - A dataframe containing temperature and fire_total_area
create_map - Creates a map of Alaska. With temperature gradients. And wildfire scale
'''


from temperature_analysis import *
from temperature_analysis import *#create fire_df which is a dataframe containing all wildfire information
PATH = './Datasets/FPA_FOD_20170508.sqlite'
fire_df = create_wildfire_df(PATH)

coords = large_fire_coord('AK',10000,fire_df)

##Create cube_local which is cube extracted for same coords as wildfire.
PATH_temp = os.path.join(os.getcwd(), "/Datasets/air.mon.mean.nc")
cube_temp = iris.load_cube(PATH_temp)
cat.add_categorised_coord(cube_temp, 'year', 'time', get_year)
cat.add_categorised_coord(cube_temp, 'month', 'time', get_month)
cube_local = region_based_cube(cube_temp,coords)

##Create a map for any year
create_map(2004,coords,cube_local,fire_df)

fire_time_based = process_df_local(coords,fire_df)
fire_time_based = fire_time_based.rename(columns={"fire_year": "year", "fire_month": "month"})


PATH_temp = os.path.join(os.getcwd(), "/Datasets/air.mon.mean.nc")
cube_temp = iris.load_cube(PATH_temp)
cube_local = region_based_cube(cube_temp,coords)

cube_max = cube_local.collapsed(['latitude','longitude'], iris.analysis.MAX)

df = cube_to_df(cube_max)
result = pd.merge(df, fire_time_based, on=['year', 'month'],how='left')
result.fillna(0,inplace=True)

create_plt(6,result)

from iris.time import PartialDateTime
PATH = os.path.join(os.getcwd(), "/Datasets/precip.mon.mean.nc")
rain_cube = iris.load_cube(PATH)

plot_rainfall(5,7,2004,rain_cube,fire_df,coords)