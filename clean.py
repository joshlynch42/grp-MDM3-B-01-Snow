import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Instructions ------------------- #
# 1. Change the file location to read from (line 13)
# 2. Change the file location to write the cleaned data to (line 108)
# 3. Uncomment print statements in lines 19 and 20 to find start and end dates. Use these in line 21
# 4. Uncomment line 91 to plot clean vs unclean data
# ---------------------------------------------------- #

# load in data
S1 = pd.read_csv('C:/Users/ale/Documents/Code/MDM3_Snow/Alaska/Alexander_Lake_1267.csv')

# create a list of dates: replace start and end values with the range of the data at the station
min_date = S1['Date'].iloc[0]
max_date = S1['Date'].iloc[-1]
# print(min_date)
# print(max_date)
time_axis = pd.date_range(start=min_date, end=max_date)
time_axis = pd.Series(time_axis)
# remove date column since not needed
station_no_date = S1.drop(['Date'], axis=1)
Data = []
# setting up indexes to remove nans
for row, column in station_no_date.iteritems():
    column = pd.DataFrame(column)
    index = column.isnull()
    index = index.astype(int)
    index = index.to_numpy()
    index = index.tolist()
    column_index = column.index.values.tolist()
    index = [column_index.index(i[0]) for i in index]
    value = column.to_numpy()
    value = value.tolist()
    values = []
    for i in value:
        values += i

    inds = []
    counter = 0
    for i in index:
        if i == 1:
            inds.append(counter)
        counter += 1


    # remove nans
    def nanSmoother():
        temp = []
        prev_found_value = 'nan'
        inds_index = 0

        for i in inds:
            if i == len(values) - 1:
                temp.append(i)
                for stored_indexes in temp:
                    values[stored_indexes] = prev_found_value

            elif np.isnan(values[i + 1]):
                temp.append(i)

            else:

                temp.append(i)
                for stored_indexes in temp:
                    if prev_found_value == 'nan':
                        values[stored_indexes] = values[i + 1]
                    else:
                        values[stored_indexes] = np.mean([values[i + 1], prev_found_value])

                temp = []
                try:
                    prev_found_value = values[inds[inds_index + 1] - 1]
                except IndexError:
                    print('reached end of data')
            inds_index += 1


    nanSmoother()
    Data.append(values)

Data = np.transpose(Data)
final_df = pd.DataFrame(Data)
Moving_average = final_df.rolling(31).mean()
Moving_average['Date'] = time_axis  # Add date column to df
print(Moving_average)
# print in blue the uncleaned data, in red the cleaned data
# change the date values to the range you wanna view
Moving_average.plot(kind='line', x='Date', y=1, color='red')
final_df.plot(kind='line', use_index=True, y=1, color='blue')
plt.show()


# -------------- Writing to the file ------------------------- #
# Change this to the file location you chose
cols = Moving_average.columns.tolist()  # Get column headers
cols = cols[-1:] + cols[:-1]  # Change order of columns
Moving_average = Moving_average[cols]  # Apply new order of columns

dict = {}
for i in range(len(Moving_average.columns)):  # Make dict of old and new column headers
    dict[Moving_average.columns[i]] = S1.columns[i]

Moving_average = Moving_average.dropna()  # Remove null values
Moving_average = Moving_average.rename(dict, axis='columns')  # Change headers to the orginal names
# Write data frame to a new file
Moving_average.to_csv('C:/Users/ale/Documents/Code/MDM3_Snow/Alaska/Alexander_Lake_1267_clean.csv', index=False)
