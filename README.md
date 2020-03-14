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
- 

## Preprocessing stage: Data Extraction
In order to make the repository light, the datasets are not saved online. The different datasets are downloaded from the references provided and stored in the folder "Datasets", in the main repository. 
1. Wildfire Dataset
    - The wildfire dataset can be extracted from the million fires dataset on Kaggel
    - This dataset contains information about 1.8 million fires in USA from 1992 to 2015
    - Dataset link: https://www.kaggle.com/rtatman/188-million-us-wildfires


## Extracting and plotting wildfire information 
Once the different datasets have been dowloaded and put in the "Datasets" folder, the notebook main_notebook.ipynb in the repository can be run to provide the different results. 


## Plotting climate features


## Showing climate correlations


## Specific case study: Alaska wildfires
