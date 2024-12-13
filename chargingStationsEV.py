"""
Author: Ewan Jones
Date: 13/12/2024

This program is the foundation of a basic framework to create plots and run simple interactive
demonstrations pertaining to electric vehicle charging in the UK. It is intended to collect,
organise and processes data from the Open Charge Map for use in outreach and public engagement.

** README **

    To fetch the required EV charging data you will need to set an envrionment variable named 'OPENCHARGEMAP_API_KEY' to an
    API key assigned to your account on https://openchargemap.org
    
    To fetch the required geolocation data you will need to set an environment variable named 'GEOAPIFY_API_KEY' to an
    API key assigned to your account on https://www.geoapify.com
"""
from scipy.stats import binned_statistic
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import os

# Figure styling using seaborn
sns.set_style("darkgrid")
sns.set_context("talk")

# Get API keys from environment variables
API_KEY_EV  = os.getenv("OPENCHARGEMAP_API_KEY")
API_KEY_GEO = os.getenv("GEOAPIFY_API_KEY")
        
""" Class to represent all chargers within a given UK city """
class CityChargers:
    
    valid_attributes = {"UUID", "coordinates", "num_points", "date_added"}
    
    # Constructor
    def __init__(self, name, bbox, coord):
        """
        INPUTS
            name (string)         -> name of city
            bbox (tuple(float))   -> bounding box (lat1, lon1, lat2, lon2) of city
            coord (tuple(float))  -> coordinate (lat, lon) of city center
        """
        self.city_name   = name
        self.city_bbox   = bbox
        self.city_coord  = coord
        
        # Data for chargers must be added through the add_chargers method, this 
        #  initialises their storage
        for attr in CityChargers.valid_attributes:
            setattr(self, attr, None)
        
    # Method to add chargers to self
    def add_chargers(self, charger_data):
        """
        INPUTS
            charger_data (dict) -> dictionary containing lists of charger attributes,
                                    invalid attributes will be ignored
        """
            
        for attr in charger_data.keys():
            # Ignore invalid attributes
            if attr not in CityChargers.valid_attributes:
                continue
            
            if getattr(self, attr) is None:
                setattr(self, attr, np.array(charger_data[attr]))
                continue
            
            _data = np.concatenate(getattr(self, attr), np.array(charger_data[attr]))
            
            # Can assume that where data is missing corresponds to 1 charging point for minimum estimate
            if attr is "num_points":
                filt = _data is None
                _data[filt] = 1
            
            setattr(self, attr, _data)
     
# Convenience function to check response from API request
def check_api_response(response):
    if response.status_code != 200:
        print(f"Unsuccessful API call with status code {response.status_code}\nExiting...")
        exit(1)     
            
""" Get EV charger (poi) data with certain conditions """
def getdata_poi(conds={}):
    """
    INPUTS
        conds (dict) -> conditions a poi must meet.
                        eg. conds={"countryid":"GB} restricts pois to those in GB
    OUTPUTS
        Requested data in json format
    """
    # Get the url for the API call
    url = "https://api.openchargemap.io/v3/poi"
    if len(conds.keys()):
        url += "/?"
        for key, value in conds.items():
            if key == "boundingbox":
                value = f"{value[0]},{value[1]}"
            url += f"{key}={value}&"
        url = url[:-1]
    
    # Call to the API to retrieve data
    response = requests.get(url, params={"key":API_KEY_EV}, headers={"Accept":"application/json"})
    check_api_response(response)
    _data = response.json()
    
    # Unpack the json data into a dictionary containing subset of data
    data_allchargers = defaultdict(lambda: [])
    for charger in _data:
        data_allchargers["UUID"].append(charger["UUID"])
        data_allchargers["coordinates"].append( (charger["AddressInfo"]["Latitude"], charger["AddressInfo"]["Longitude"]) )
        data_allchargers["num_points"].append(charger["NumberOfPoints"])
        data_allchargers["date_added"].append(charger["DateCreated"]) # Date added to database not physically installed
    
    return data_allchargers

""" Convert UK city names to latitude-longitude coordinates and get their bounding boxes """
def geocode_ukcities(city_names):
    """
    INPUTS
        city_names (list(string)) -> names of cities within the UK
    OUTPUTS
        dictionary of (coord, bbox), keyed by city name
            coord = (latitude, longitude)
            bbox = (coord_upper_left, coord_lower_right)
    """
    url = f"https://api.geoapify.com/v1/geocode/search?filter=countrycode:uk&lang=en&limit=1&type=city&apiKey={API_KEY_GEO}"
    
    geocodes = {}
    for city in city_names:
        city_url = url + f"&text={city}"
        response = requests.get(city_url, headers={"Accept":"application/json"}, params={"key":API_KEY_GEO})
        check_api_response(response)
        
        data_properties = response.json()["features"][0]["properties"]
        data_bbox       = response.json()["features"][0]["bbox"]
        
        # Reformat bbox for use with opencharge API
        upper_left  = (data_bbox[3], data_bbox[0])
        bottom_right = (data_bbox[1], data_bbox[2])
        bbox = (upper_left, bottom_right)
        
        geocodes[city] = ((data_properties["lat"], data_properties["lon"]), bbox)
        
    return geocodes

# Haversine formula to find distance between two points on the Earth
def haversine_earth(lat1, lon1, lat2, lon2):
    radius = 6378 #km

    delta_lat = (lat2 - lat1) * np.pi / 180
    delta_lon = (lon2 - lon1) * np.pi / 180
    
    numerator = 1 - np.cos(delta_lat) + np.cos(lat1*np.pi/180) * np.cos(lat2*np.pi/180) * (1 - np.cos(delta_lon))
    return 2 * radius * np.arcsin(np.sqrt(numerator/2))

""" Function which creates and saves a static map of given region """
def save_mapimage(savepath, center_coord, zoom_level, marker_coords=None, map_style="klokantech-basic"):
    """
    INPUTS
        savepath (string)    -> filepath to save the created map
        center_coord (tuple) -> tuple of (latitude, longitude) coordinates for map center
        zoom_level (float)   -> zoom applied to map about center
        marker_coords (list) -> list of (latitude, longitude) coordinates for placing markers on map
        map_style (string)   -> design styling of map
    OUTPUTS
        png image of map as defined by input parameters
    """
    
    url = f"https://maps.geoapify.com/v1/staticmap?apiKey={API_KEY_GEO}"

    # Body of the request
    body = {"width":600, "height":400, "zoom":zoom_level, "format":"png", "style":map_style}
    body["center"] = {"lat": center_coord[0], "lon": center_coord[1]}
   
    # # Add markers
    if marker_coords is not None:
        body["markers"] = []
        for coord in marker_coords:
            body["markers"].append({"lat":coord[0], "lon":coord[1], "color":"orange", "size":"small",
                                    "icontype":"awesome", "icon":"charging-station", "shadow":"no",
                                    "type":"material"})

    # Call API to retrieve this map
    response = requests.post(url, headers={"Accept":"application/json"}, json=body)
    check_api_response(response)

    # Save image to save path
    with open(savepath.split(".")[0] + ".png", "wb") as fptr:
        fptr.write(response.content)
    
if __name__ == "__main__":

    """
    EXAMPLE 1
        Compare the rate at which EV chargers are installed in various UK cities.
    """
    # Get data for EV chargers in various cities
    cities       = ["edinburgh", "glasgow", "london", "bristol", "birmingham"]
    geocodes     = geocode_ukcities(cities)
    city_data = {}
    for city in cities:
        city_data[city] = CityChargers(city, geocodes[city][1], geocodes[city][0])
        city_data[city].add_chargers(getdata_poi(conds={"boundingbox":city_data[city].city_bbox, "maxresults":100_000}))
        
    # Convert dates added to POSIX timestamps which are easier to deal with for now
    tadded = defaultdict(lambda: [])
    for city in cities:
        for date in city_data[city].date_added:
            tadded[city].append(datetime.timestamp(datetime.strptime(date.split("T")[0], "%Y-%m-%d")))
            
    # Create bin edges for analysis
    min_tstamp = np.min([np.min(tadded[city]) for city in cities])
    max_tstamp = np.max([np.max(tadded[city]) for city in cities])
    
    # Align the min and max timestamps with the new year
    min_tstamp = datetime.timestamp(datetime(datetime.fromtimestamp(min_tstamp).year, 1, 1))
    max_tstamp = datetime.timestamp(datetime(datetime.fromtimestamp(max_tstamp).year+1, 1, 1))
    
    bin_width = timedelta(weeks=52).total_seconds() # I want bins that are 12 months wide
    bin_edges = np.array([min_tstamp + i*bin_width for i in range(int((max_tstamp - min_tstamp)/bin_width))])
    
    # Get bin centers in standard date format
    bin_centers = bin_edges[1:] - bin_width/2
    bin_centers = [datetime.fromtimestamp(i) for i in bin_centers]
    
    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(8,6))
    colours = {"edinburgh":"red", "glasgow":"blue", "london":"black", "bristol":"green", "birmingham":"orange"} # Line colours
    
    for city in cities:
        # Let's skip london for now
        if city.lower() == "london": continue
        
        # Count number of chargers added in each bin
        count, _, _ = binned_statistic(tadded[city], tadded[city], statistic="count", bins=bin_edges)
        ax.plot(bin_centers, count, color=colours[city], label=city.capitalize())
    
    ax.set_title("Number of EV charging stations added to the \nOpen Charge Map database in various UK cities \n(excluding London)", pad=20)
    ax.set_xlabel("Year", labelpad=15)
    ax.set_ylabel("#Stations added", labelpad=15)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig("stationDevelopment_exclLondon.png", bbox_inches="tight", dpi=800)
    
    # Now let's include London and see how it compares.
    # I would write a function for this plotting to avoid duplication but I don't want to overcomplicate for this single demonstrative use.
    fig, ax = plt.subplots(figsize=(8,6))
    for city in cities:
        # Count number of chargers added in each bin
        count, _, _ = binned_statistic(tadded[city], tadded[city], statistic="count", bins=bin_edges)
        ax.plot(bin_centers, count, color=colours[city], label=city.capitalize())
    
    ax.set_yscale("log")
    ax.set_title("Number of EV charging stations added to the \nOpen Charge Map database in various UK cities", pad=20)
    ax.set_xlabel("Year", labelpad=15)
    ax.set_ylabel("#Stations added", labelpad=15)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig("stationDevelopment.png", bbox_inches="tight", dpi=800)
    
    """
    EXAMPLE 2
        Find and save the location of 20 closest charging stations to city centers. The maps created
        are saved to disk and can be customised by changing the function parameters.
        This could easily be extended to work around arbitrary postcodes.
        
        I would clean the data thoroughly to avoid duplicates and deal with overlapping points if I made this for real,
        at the moment there may be slightly fewer markers than requested due to this.
    """
    zoom_level  = 13
    num_markers = 20
    for city in cities:
        charger_coords = city_data[city].coordinates
        city_center    = city_data[city].city_coord
        # Get charging stations closest to city center
        distance = []
        for coord in charger_coords:
            if None in coord: continue
            distance.append(haversine_earth(*city_center, *coord))
        distance = np.array(distance)
        sort_filt = np.argsort(distance)
        marker_coords = charger_coords[sort_filt]
        
        save_mapimage(f"central_chargers_{city}", city_center, zoom_level, marker_coords=marker_coords[:num_markers])