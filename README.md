# Study of Wildfires in USA: ECE143 Project
This repository contains the codes for the data extraction and analysis for the different wildfires in the USA. Our aim here is to visualize wildfires along with other climatic conditions. Additionally we provide numerical metrics in terms of the correlations for the different factors. We also provide a specific case study for the wildfires in Alaska. 

All the python scripts are placed in the src directory. The different notebooks for the experiments conducted can be found in "Local Notebooks and Codes".

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
    - This dataset contains
    - Dataset link: 
    
4. Gas Emission Dataset
    - This dataset contains year-total gas emission of different greenhouse gas from 1990 to 2014
    - Dataset link: https://www.kaggle.com/unitednations/international-greenhouse-gas-emissions
    
5. Temperature Dataset
    - This dataset contains
    - Dataset link: 
    
6. Plantation Dataset
    - This dataset contains
    - Dataset link: 


## Extracting and plotting wildfire information 
Once the different datasets have been dowloaded and put in the "Datasets" folder, the notebook main_notebook.ipynb in the repository can be run to provide the different results. 


## Plotting climate features


## Showing climate correlations


## Specific case study: Alaska wildfires
