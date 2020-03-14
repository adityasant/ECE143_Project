# Study of Wildfires in USA: ECE143 Project
This repository contains the codes for the data extraction and analysis for the different wildfires in the USA. Our aim here is to visualize wildfires along with other climatic conditions. Additionally we provide numerical metrics in terms of the correlations for the different factors. We also provide a specific case study for the wildfires in Alaska. 

All the python scripts are placed in the src directory. The different notebooks for the experiments conducted can be found in "Local Notebooks".

## Required python dependencies
The following python 3.7 packages need to be installed on the machine prior to running the scripts. 
- numpy
- pandas
- matplotlib
- Basemap
- sqlite3
- seaborn
- iris

## Preprocessing stage: Data Extraction
In order to make the repository light, the datasets are not saved online. The different datasets are downloaded from the references provided and stored in the folder "Datasets", in the main repository. And do not change their filename.
1. Wildfire Dataset
    - The wildfire dataset can be extracted from the million fires dataset on Kaggel
    - This dataset contains information about 1.8 million fires in USA from 1992 to 2015
    - Dataset link: https://www.kaggle.com/rtatman/188-million-us-wildfires
    
2. Rainfall Dataset
    - This dataset contains monthly average precipitation from 1979/01 to 2020/02
    - Dataset link: https://www.esrl.noaa.gov/psd/data/gridded/data.cmap.html
    
3. Wind and Air Pressure Dataset
    - This dataset contains monthly averaged wind speed and air-pressure on single levels from 1979 to present
    - Dataset link: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview
    
4. Gas Emission Dataset
    - This dataset contains year-total gas emission of different greenhouse gas from 1990 to 2014
    - Dataset link: https://www.kaggle.com/unitednations/international-greenhouse-gas-emissions
    
5. Temperature Dataset
    - This dataset contains monthly average temperature from 1992 to 2015
    - Dataset link: https://www.esrl.noaa.gov/psd/repository/entry/show?entryid=synth%3Ae570c8f9-ec09-4e89-93b4-babd5651e7a9%3AL25jZXAucmVhbmFseXNpcy5kZXJpdmVkL3N1cmZhY2UvYWlyLm1vbi5tZWFuLm5j
    
6. Vegetation Dataset
    - This dataset contains NOAA Climate Data Record (CDR) of Normalized Difference Vegetation Index (NDVI) from 1981 to present
    - Dataset link: https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00813


## Running main notebook: Demo of plots
In order to visulaize the different plots, a demo notebook is prepared. 
- "main_notebook.ipynb" located in the main repo can be run directly using Jupyter notebooks
- However, edits cannot be made in terms of the desired year and fire size information

## Extracting and plotting wildfire information 
The script wildfire_plots.py contains the script to plot the histograms and the different geographic plots
```
- location: src/wildfire_plots.py
- run the command: python wildfires_plots.py
```
This script will plot the different wildfires of size >5000 hectares from the years 1992-2015. Due to the smaller number of fires, we do not include the functionality to change the dates or fire sizes. 


## Plotting climate features


## Showing climate correlations


## Specific case study: Alaska wildfires
