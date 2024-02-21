import sys, os
import numpy as np
from pyhdf.SD import SD
import pandas as pd
from datetime import datetime, time, timedelta
import re
import time
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import cartopy.crs as ccrs
import cartopy.feature as cfeature
#functions used in dust_score analysis

def read_xlsx_tecq(file_name):
    """
    Function for reading xlsx TECQ file in the same file path as the jupyter notebook
    Dataframe elements are coerced to numeric, leaving NaN values present
    The 'Date' columns is set as the index, this helps to search rows by datetime later
    Row indices and column headers are datetime objects

    Parameters:
        files_name (str): name of file
    """
    tecq_df=pd.read_excel(os.path.join(sys.path[0],file_name))
    for col in tecq_df.columns[1:]:
        tecq_df[col]=pd.to_numeric(tecq_df[col], errors='coerce')
    tecq_df.set_index('Date', inplace=True)
    return tecq_df

def coordinates(lat,long):
    """
    Function takes separate latitude and longitude values from the arrays present in the AIRS hdf file and combines them into one lat, long point
    
    Parameters:
        lat,long (list): separate coordinate points
    """
    #create multidimensional array with same dimensions as lat and long arrays in hdf file
    coords=np.zeros((lat.shape[0],lat.shape[1],2),dtype='float')
    for i in range(lat.shape[0]):
        for j in range(lat.shape[1]):
            #create an array from lat long points before saving into coords array
            coords[(i,j)]=np.array([lat[i,j],long[i,j]])        
    return coords

def round_nearest_hour(datetime_obj):
    """
    Function takes a datetime object and rounds up or down to nearest hour

    Parameters:
        datetime_obj (datetime): datetime object
    """
    start_hour = datetime_obj.replace(minute=0, second=0, microsecond=0)
    half_hour = datetime_obj.replace(minute=30, second=0, microsecond=0)
    #rounding up or down 
    if datetime_obj >= half_hour:
        datetime_obj = start_hour + timedelta(hours=1)
    else:
        datetime_obj = start_hour
    return datetime_obj

def find_rangedatetime(global_attributes):
    #RANGEENDINGDATE  "2019-01-01"
    #RANGEENDINGTIME  "20:53:20.999999Z"
    time_pattern = r'VALUE="\d{2}:\d{2}:\d{2}\.\d{6}Z"'
    date_pattern = r'VALUE="\d{4}-\d{2}-\d{2}"'
    
    core_attributes=global_attributes.get('coremetadata')
    core_attributes_groups=core_attributes.split('\n')
    core_attributes_groups_whitespace = [element.strip() for element in core_attributes_groups]
    core_attributes_groups_strspace = [element.replace(' ', '') for element in core_attributes_groups_whitespace]

    matches_date = [string for string in core_attributes_groups_strspace if re.match(date_pattern, string)][0]
    matches_time = [string for string in core_attributes_groups_strspace if re.match(time_pattern, string)][0]
    print(matches_date, matches_time)
    # Find the PRODUCTIONDATETIME of the HDF file
    datetime_hdf_raw=f'{matches_date.split('=')[1].strip('"')}T{matches_time.split('=')[1].strip('"')}'
    return datetime_hdf_raw

#PRODUCTIONDATETIME IS NOT A GOOD MEASURE FOR WHEN DATA IS FIRST CAPTURED
def find_productiondatetime(global_attributes):
    # Regex for PRODUCTIONDATETIME YYYY-MM-DD pattern
    datetime_pattern = r'VALUE="\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"'  
    
    core_attributes=global_attributes.get('coremetadata')
    core_attributes_groups=core_attributes.split('\n')
    core_attributes_groups_whitespace = [element.strip() for element in core_attributes_groups]
    core_attributes_groups_strspace = [element.replace(' ', '') for element in core_attributes_groups_whitespace]

    # Search for strings that match the datetime pattern in the index
    # There should be only one match
    matches = [string for string in core_attributes_groups_strspace if re.match(datetime_pattern, string)][0]
    print(matches)
    # Find the PRODUCTIONDATETIME of the HDF file
    datetime_hdf_raw=matches.split('=')[1].strip('"')
            
            
    return datetime_hdf_raw

def match_coords(coords, array_to_match):
    matched_coords=np.zeros((coords.shape[0],coords.shape[1],coords.shape[2]),bool)
    for i in range(len(coords[:])):
        for j in range(len(coords[0,:])):
            matched_coords[i,j] = np.isclose(coords[i,j],array_to_match,rtol=0.05)
            if np.all(matched_coords[i,j] == [False, True]) or np.all(matched_coords[i,j] == [True, False]):
                matched_coords[i,j]=[False, False]
    return matched_coords
    
def create_mask(matched_coords):
    mask=np.zeros((matched_coords.shape[0],matched_coords.shape[1]),bool)
    for i in range(len(matched_coords[:])):
        for j in range(len(matched_coords[0,:])):
            if np.all(matched_coords[i,j] == [True, True]):
                mask[i,j]=[True]
    return mask