# What is EV Charging Info?

EV Charging Info is a framework to aggregate data, create plots, and run simple
interactive demonstrations pertaining to electric vehicle (EV) charging. Currently
it uses data from the Open Charge Map (for the EV station data) in tandem with
geolocation data from Geoapify.

This code is currently targeted for use in outreach and public engagement,
however it has the potential to be extended for sophisticated data analysis
projects.

# The Example Output

The code within `main.py` currently runs two default examples. The first
generates a figure showing the number of EV chargers added to the open
charge map database over the last ten years. Statistically, this uses
a bin width of 12 months, with bin edges in June such that their
centres occur at each new year.

The second example generates static maps showing the locations of the
twenty closest EV charging stations to the city centres of the
locations used in the first example.

The results of these examples, as they are written in `main.py`, have
been precomputed and saved in the 'example_output' folder. This was
done with the intention to provide prospective users insight into
the code's applicability without the need to install it themselves: as
this requires signing up to APIs to retrieve necessary data.

# Installation

As this project is contained within a single Python file at the current stage
of development, the only installation procedures required are the installation
of necessary Python packages, and the creation of two environment variables
holding API keys.

## 1. Prerequisite Python Packages

Below is a list of Python packages required to run the code, alongside their
versions for which the code has been confirmed to run. These packages are
generally robust and so any recent version is expected to work without issue.
The recommended method of install for these is using conda, but all packages
should be available through pip also.

+ **python** (3.7.16)
+ numpy (1.21.0)
+ scipy (1.7.0)
+ matplotlib (3.4.1)
+ seaborn (0.12.2)
+ requests (2.28.1)

## 2. Environment Variables

You will need to set two environment variables, each to a valid API key
associated with the used service. Below is a list of the variables'
name and the website where you can sign up for the API and receive
your key.

+ OPENCHARGEMAP_API_KEY (https://openchargemap.org)
+ GEOAPIFY_API_KEY (https://www.geoapify.com)

# Acknowledgements

+ Service used for EV charging API: https://openchargemap.org
+ Service used for geolocation API: https://www.geoapify.com
+ Source of city population data: https://www.macrotrends.net