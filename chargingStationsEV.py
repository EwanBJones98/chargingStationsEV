"""
Author: Ewan Jones
Date Created: 13/12/2024
Date Last Modified: 16/12/2024
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
sns.set_style()
sns.set_context("talk")

# Get API keys from environment variables
API_KEY_EV  = os.getenv("OPENCHARGEMAP_API_KEY")
API_KEY_GEO = os.getenv("GEOAPIFY_API_KEY")
        
""" Class to represent all chargers within a given UK city """
class CityChargers:
    
    valid_attributes = {"UUID", "coordinates", "num_points", "date_added"}
    
    # Constructor
    def __init__(self, name, coord, bbox):
        """
        INPUTS
            name (string)         -> name of city
            coord (tuple(float))  -> coordinate (lat, lon) of city center
            bbox (tuple(float))   -> bounding box (lat1, lon1, lat2, lon2) of city
            
        """
        
        self.city_name   = name
        self.city_coord  = coord
        self.city_bbox   = bbox
        
        # Data for chargers must be added through the add_chargers method,
        #   this initialises their storage
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
            if attr not in CityChargers.valid_attributes: continue
            
            if getattr(self, attr) is None:
                setattr(self, attr, np.array(charger_data[attr]))
                continue
            
            _data = np.concatenate(getattr(self, attr), np.array(charger_data[attr]))
            
            # Can assume that where data is missing corresponds to 1 charging point for minimum estimate
            if attr == "num_points":
                filt = _data is None
                _data[filt] = 1
            
            setattr(self, attr, _data)
     
# Convenience function to check response from API request
def check_api_response(response):
    if response.status_code != 200:
        print(f"Unsuccessful API call with status code {response.status_code}\nExiting...")
        exit(1)     
            
""" Get data for EV charging stations with certain conditions """
def get_ev_station_data(conds={}):
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
def get_coordinates_from_city_name(city_names):
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
def haversine_distance_earth(lat1, lon1, lat2, lon2):
    radius = 6378 #km
    delta_lat = (lat2 - lat1) * np.pi / 180
    delta_lon = (lon2 - lon1) * np.pi / 180
    numerator = 1 - np.cos(delta_lat) + np.cos(lat1*np.pi/180) * np.cos(lat2*np.pi/180) * (1 - np.cos(delta_lon))
    return 2 * radius * np.arcsin(np.sqrt(numerator/2))

# Function to read in population data files
def read_population_data(city_name, dirpath=None):
    """ Returns dictionary of population keyed by year """
    # Use default data directory path if none supplied
    if dirpath is None:
        dirpath = os.path.join(os.path.dirname(__file__), "population_data")
    filepath = os.path.join(dirpath, f"{city_name.lower()}.csv")
    
    data = {}
    with open(filepath, "r") as fptr:
        for line in fptr.readlines():
            # Skip commented lines
            if line.startswith("#"): continue
            line_data = [int(i.strip()) for i in line.split(",")]
            data[line_data[0]]  = line_data[1]
    return data

""" Function which creates and saves a static map of given region """
def save_static_map_image(savepath, center_coord, zoom_level, marker_coords=None, map_style="klokantech-basic"):
    """
    INPUTS
        savepath (string)    -> filepath to save the created map
        center_coord (tuple) -> tuple of (latitude, longitude) coordinates for map center
        zoom_level (float)   -> zoom applied to map about center
        marker_coords (list) -> list of (latitude, longitude) coordinates for placing markers on map
        map_style (string)   -> design styling of map
    OUTPUTS
        .png image of map as defined by input parameters
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
        
""" Function which creates CityChargers objects from city names"""
def get_city_chargers(city_names):
    """
    INPUTS
        city_names (list(string)) -> names of cities
    OUTPUTS
        Dictionary keyed by city name, containing CityChargers objects
    """
    
    geocodes = get_coordinates_from_city_name(cities)
    
    city_data = {}
    for city in cities:
        city_data[city] = CityChargers(city, *geocodes[city])
        city_conds = {"boundingbox":city_data[city].city_bbox, "maxresults":100_000}
        city_data[city].add_chargers(get_ev_station_data(conds=city_conds))
        
    return city_data
    
if __name__ == "__main__":
    
    """
    EXAMPLE 1
        Compare how many EV chargers are installed per year in various UK cities from 2014-2024
    """
    # Get data for EV chargers in various cities
    cities       = ["edinburgh", "glasgow", "london", "bristol", "manchester"]
    city_data    = get_city_chargers(cities)
        
    # Convert dates added to POSIX timestamps which are easier to deal with for now
    tadded = defaultdict(lambda: [])
    for city in cities:
        for date in city_data[city].date_added:
            tadded[city].append(datetime.timestamp(datetime.strptime(date.split("T")[0], "%Y-%m-%d")))
            
    # Create time bins for analysis
    start_year, end_year = 2014, 2024
    bin_centers_tstamps = [datetime.timestamp(datetime(year, 1, 1)) for year in range(start_year, end_year+1)]
    bin_width = timedelta(weeks=52).total_seconds() # I want bins one year wide
    bin_edges = np.array([bin_centers_tstamps[0] + bin_width*(i - 1/2) for i in range(end_year-start_year+2)])
    
    bin_centers = [datetime.fromtimestamp(i) for i in bin_centers_tstamps] # Get bin centers in standard date format
    
    # Line colours for plots
    colours = {"edinburgh":"red", "glasgow":"blue", "london":"black", "bristol":"green", "manchester":"orange"}
    
    # Let's also look at the number of EV stations added per capita for a fairer comparison
    pop_data = {city: read_population_data(city) for city in cities} # Read in population data stored
    
    # Create figure
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(8,12))
    for city in cities:
        # Count number of chargers added in each time bin
        count, _, _ = binned_statistic(tadded[city], tadded[city], statistic="count", bins=bin_edges)
        # Divide chargers added by population for second axis
        count_per_capita = np.array([count[i] / pop_data[city][x.year] for i, x in enumerate(bin_centers)])
        
        axs[1].plot(bin_centers, count, color=colours[city], label=city.capitalize())
        axs[0].plot(bin_centers, 1000 * count_per_capita, color=colours[city])
    
    axs[1].set_xlabel("Year", labelpad=15)
    axs[1].set_ylabel("#Stations added", labelpad=15)
    axs[1].legend(fontsize=16)
    axs[0].set_title("Number of EV charging stations added to the \nOpen Charge Map database in various UK cities", pad=20)
    axs[0].set_ylabel("#Stations added per 1,000 citizens", labelpad=15)
    for ax in axs:
        ax.set_yscale("log")
        ax.grid(alpha=0.7)
        ax.tick_params(which="both", direction="inout", zorder=100)
        ax.set_xticks(bin_centers[::2], labels=[i.year for i in bin_centers[::2]])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.)
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
            distance.append(haversine_distance_earth(*city_center, *coord))
        distance = np.array(distance)
        sort_filt = np.argsort(distance)
        marker_coords = charger_coords[sort_filt]
        
        save_static_map_image(f"central_chargers_{city}", city_center, zoom_level, marker_coords=marker_coords[:num_markers])