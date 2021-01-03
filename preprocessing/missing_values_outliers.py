import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_json('data/train.json', convert_dates = ['created'])

#Dealing with missing values, outliers
def number_missing(variable, missing):
    print("Number of values missing in " + variable + ": ", end = '')
    if (missing == 'na'):
        print(data[variable].isnull().sum())
    else:
        print(len(data[data[variable] == missing]))

#Find out the number of missing values in each variable.
number_missing('bathrooms','na')
number_missing('bedrooms','na')
number_missing('building_id', '0')
number_missing('created','na')
print("Number of values missing in description: ", data['description'].str.isspace().sum())
number_missing('display_address','na')
print("Number of values missing in features: ", len(data[data['features'].str.len() == 0]))
number_missing('latitude','na')
number_missing('listing_id','')
number_missing('longitude','na')
number_missing('manager_id','na')
print("Number of values missing in photos: ", len(data[data['photos'].str.len() == 0]))
number_missing('price','na')
number_missing('street_address','na')
number_missing('interest_level','na')

#Find out the number of outliers in each variable. Plot visualisations to
#demonstrate them. Handle outliers.
#Adapted from https://stackoverflow.com/questions/39068214/how-to-count-outliers-for-all-columns-in-python
data_num = data[['bathrooms','bedrooms','latitude','longitude','price']]
Q1 = data_num.quantile(0.25)
Q3 = data_num.quantile(0.75)
IQR = Q3 - Q1
print(((data_num < (Q1 - 1.5 * IQR)) | (data_num > (Q3 + 1.5 * IQR))).sum())

plt.boxplot(data_num['bathrooms'])
plt.title('Bathrooms Boxplot')
plt.ylabel('# of Bathrooms')
plt.show()
plt.boxplot(data_num['bedrooms'])
plt.title('Bedroom Boxplot')
plt.ylabel('# of Bedrooms')
plt.show()
plt.boxplot(data_num['latitude'])
plt.title('Latitude Boxplot')
plt.ylabel('Latitude')
plt.show()
plt.boxplot(data_num['longitude'])
plt.title('Longitude Boxplot')
plt.ylabel('Longitude')
plt.show()
plt.boxplot(data_num['price'])
plt.title('Price Boxplot')
plt.ylabel('Price')
plt.show()

#Can we safely drop the missing values? If not, how will you deal with them?
