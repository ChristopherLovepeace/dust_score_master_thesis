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
import geopy
from geopy.distance import great_circle
from matplotlib.patches import Ellipse
def distance(point1, point2):
    """Calculate the distance between two geographical points in miles."""
    #distance_miles=geodesic(point1, point2).miles
    distance_miles=great_circle(point1, point2).miles
    return distance_miles

def points_inside_circle(center, radius, points):
    """Find points inside the given circle."""
    distances = np.array([distance(center, point) for point in points])
    inside_points = points[distances <= radius]

    return inside_points 
def points_between_circles(center, inner_radius, outer_radius, points):
    """Find points between two concentric circles."""
    inner_points = points_inside_circle(center, inner_radius, points)
    outer_points = points_inside_circle(center, outer_radius, points)
    between_points = np.array([point for point in outer_points if point not in inner_points])

    return between_points
    

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
    #print(matches_date, matches_time)
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
    #print(matches)
    # Find the PRODUCTIONDATETIME of the HDF file
    datetime_hdf_raw=matches.split('=')[1].strip('"')       
    return datetime_hdf_raw

def match_coords(coords, array_to_match,tolerance):
    matched_coords=np.zeros((coords.shape[0],coords.shape[1],coords.shape[2]),bool)
    for i in range(len(coords[:])):
        for j in range(len(coords[0,:])):
            matched_coords[i,j] = np.isclose(coords[i,j],array_to_match,rtol=tolerance)
            if np.all(matched_coords[i,j] == [False, True]) or np.all(matched_coords[i,j] == [True, False]):
                matched_coords[i,j]=[False, False]
    return matched_coords

def match_coords_circle(coords, circle_points):
    matched_coords=np.zeros((coords.shape[0],coords.shape[1]),bool)
    for i in range(len(coords[:])):
        for j in range(len(coords[0,:])):
            for point in circle_points:
                if point[0] == coords[i,j][0] and point[1] == coords[i,j][1]:
                    matched_coords[i,j] = [True]#np.any(coords[i,j],point)
                else:
                    matched_coords[i,j]=[False]
            #if np.all(matched_coords[i,j] == [False, True]) or np.all(matched_coords[i,j] == [True, False]):
            #    matched_coords[i,j]=[False, False]
    return matched_coords
    
def create_mask(matched_coords):
    mask=np.zeros((matched_coords.shape[0],matched_coords.shape[1]),bool)
    for i in range(len(matched_coords[:])):
        for j in range(len(matched_coords[0,:])):
            if np.all(matched_coords[i,j] == [True, True]):
                mask[i,j]=[True]
            else:
                mask[i,j]=[False]
    return mask

def draw_grid_plot(tecq_coords, coords, inside_points, file, datetime, dust_score, dist, plot_ellipse, wind_dir=None, center_windv=None, semi_major_axis=None, semi_minor_axis=None, between_points=None):
    start_time=time.time()
    # Create a map plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lats=coords[:,:,0]
    longs=coords[:,:,1]
    
    # Plot each coordinate on the map
    sc = ax.scatter(longs, lats, c=dust_score, cmap='viridis', norm=Normalize(vmin=dust_score.min(), vmax=dust_score.max()), s=5, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label('Dust_score')
    
    #print(inside_points[:,1], inside_points[:,0])
    ax.plot(inside_points[:,1], inside_points[:,0], 'w.', markersize=1, transform=ccrs.PlateCarree())
    if between_points!=None:
        ax.plot(between_points[:,1], between_points[:,0], 'r.', markersize=1, transform=ccrs.PlateCarree())
    '''
    lats_masked=coords_masked[:,0]
    longs_masked=coords_masked[:,1]
    
    #ax.plot(long, lat, 'go', markersize=1, transform=ccrs.PlateCarree())
    ax.plot(longs_masked, lats_masked, 'r.', markersize=1, transform=ccrs.PlateCarree())
    '''
    ax.plot(tecq_coords[1],tecq_coords[0], 'bx', markersize=3, transform=ccrs.PlateCarree())
    # Add gridlines and coastlines
    ax.gridlines()
    ax.coastlines()
    # Add country borders
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1, edgecolor='black')
    # Add state borders
    ax.add_feature(cfeature.STATES, linestyle='-', linewidth=0.5, edgecolor='black')
    # Add title and show the plot
    plt.title(f'{datetime}, Distance to next focus: {dist}miles')
    if plot_ellipse==True:
        draw_helper_lines(ax, wind_dir, tecq_coords, center_windv, semi_major_axis, semi_minor_axis)

    ax.set_extent([-115, -85, 15, 40], crs=ccrs.PlateCarree()) 
    plt.show()
    #plt.savefig(f'{os.path.join(file_path_plots,file)}.png')
    end_time=time.time()
    run_time=end_time-start_time
    print(f"draw_grid_plot execution time: {run_time} msec")


def draw_helper_lines(ax, wind_dir, center_cams, center_windv, semi_major_axis, semi_minor_axis):
    print("Hello")
    center_distance= distance(center_windv,center_cams)/2
    lat_deg_change=geopy.units.degrees(arcminutes=geopy.units.nautical(miles=center_distance*np.sin(np.radians(wind_dir))))
    long_deg_change=geopy.units.degrees(arcminutes=geopy.units.nautical(miles=center_distance*np.cos(np.radians(wind_dir))))
    center_ellipse=[center_cams[0]+lat_deg_change,center_cams[1]+long_deg_change]
    ellipse=Ellipse(xy=center_ellipse, width=semi_minor_axis, height=semi_major_axis, angle=wind_dir, edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse)
    # Set aspect of the plot to 'equal' to make sure the ellipse is not distorted
    ax.set_aspect('equal')
