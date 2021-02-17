from __future__ import print_function
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Instructions ------------------- #
# 1. Create an empty csv file to write the data to. Once created keep this file closed when running the code.
# 2. Go to line 76 and change the file location to the file location you just created.
# 3. Chose a list of features from the list below and add their symbols to the variable 'feature_list'
# 4. Chose the interval, station id and state that you would like. Use this url to find station id's and state's code
#    https://wcc.sc.egov.usda.gov/nwcc/tabget. To find the state code. Click the state you want,
#    press 'Process Selection', then look at your new url. (e.g. For California it should have state=CA)
# 5. Then run the code.

# ------- Choosing variables ---------------#

# Feature list and Symbols #
# TAVG = air temperature average
# TMAX = air temperature maximum
# TMIN = air temperature minimum
# TOBS = air temperature observed
# BATT = battery
# PREC = precipitation accumulation
# PRCP = precipitation increment
# PRCPSA = precipitation increment - snow-adj
# SNWD = snow depth
# WTEQ = snow water equivalent
# WTEQX = snow water equivalent maximum
# SNDN = snow density
# SNRR = snow rain ratio
# WDIRV = wind direction average
# WSPDV = wind speed average
# WSPDX = wind speed maximum

feature_list = ['TAVG', 'PREC', 'SNDN']  # Get symbols from above
interval = 'daily'  # daily, monthly, semimonthly, hourly, annual_water_year, annual_calendar_year
time_period = 'start_of_period'
station_id = '1285'  # Use generator webpage to find station id's
state = 'AK'  # (e.g AK is Alaska, AZ is Arizona)
start = 'POR_BEGIN'  # Can be index back from 0 (e.g. start='-6' and end='0' for 7 days)
end = 'POR_END'

# --------- Creating the url ---------------- #

url_start = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport,metric/' \
            '{}/{}/{}:{}:SNTL%7Cid=%22%22%7Cname/{},{}/'.format(interval, time_period, station_id, state, start, end)
url_end = '?fitToScreen=false'
for i in range(len(feature_list)):
    if i == 0:
        url_start = url_start + feature_list[i] + '::value'
    else:
        url_start = url_start + ',' + feature_list[i] + '::value'

url = url_start + url_end
print(url)  # This is the url created

# ---------- Web scraping the url for the data ---------- #

req = requests.get(url)
src = req.text
soup = BeautifulSoup(src, 'html.parser')
doc = soup.get_text()

# ------------ Separating data from useless text ---------- #

separation = '#'
doc = doc.split(separation)
data = doc[-1]
data = data.split('\n')
for i in range(len(data)-1):
    data[i] = data[i] + ','

# -------------- Writing to the file ------------------------- #
print('Writing to file')
# Change this to the file location you chose
with open('D:/Users/Joshg/Documents/MDM3/Alaska/filename.csv', 'w') as f:
    for i in range(len(data)-1):
        i += 1
        f.write(data[i])
        f.write('\n')

print('Finished writing to file')
