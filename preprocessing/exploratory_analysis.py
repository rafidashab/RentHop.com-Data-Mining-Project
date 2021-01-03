import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_json('data/train.json', convert_dates = ['created'])

###### Exploratory data analysis ######

# CAN ALSO TRY MIN-MAX SCALING FOR PRICE ONLY
def remove_outliers_iqr(feature):
	q1 = data[feature].quantile(0.25)
	q3 = data[feature].quantile(0.75)
	IQR = q3-q1
	outliers_pruned = data[(data[feature] < (q3 + 1.5*IQR)) & (data[feature] > (q1 - 1.5*IQR))][feature]
	return outliers_pruned

def plot_hist(feature, bins):
	outliers_pruned = remove_outliers_iqr(feature)

	plt.hist(outliers_pruned, bins = bins)
	plt.xlabel(feature + ' bins')
	plt.ylabel('Frequency')
	plt.title(feature + ' w/ equal width binning')
	plt.savefig('figures/' + feature + '_equal_bin_hist.png')
	plt.clf()	

# Price boxplot
plt.boxplot(data['price'])
plt.title('Price Boxplot')
plt.ylabel('Price (USD)')
plt.savefig('figures/price_boxplot.png')
plt.clf()

# Equal-width binning histogram for price
plot_hist('price', 100)

# Equal-width binning for latitude and longitude
plot_hist('longitude', 100)
plot_hist('latitude', 100)


# 2-D histogram for longitude and latitude
lat_lon = data[['latitude', 'longitude']]

# THIS IS A CRUDE METHOD OF REMOVING OUTLIERS FROM 2D SPATIAL DATA
q1_lat = lat_lon['latitude'].quantile(0.25)
q3_lat = lat_lon['latitude'].quantile(0.75)
IQR_lat = q3_lat-q1_lat
lat_lon = lat_lon[(lat_lon['latitude'] < (q3_lat + 1.5*IQR_lat)) & (lat_lon['latitude'] > (q1_lat - 1.5*IQR_lat))]

q1_lon = lat_lon['longitude'].quantile(0.25)
q3_lon = lat_lon['longitude'].quantile(0.75)
IQR_lon = q3_lon-q1_lon
lat_lon = lat_lon[(lat_lon['longitude'] < (q3_lon + 1.5*IQR_lon)) & (lat_lon['longitude'] > (q1_lon - 1.5*IQR_lon))]

fig, ax = plt.subplots(tight_layout = True)
hist = ax.hist2d(lat_lon['longitude'], lat_lon['latitude'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('2D histogram for longitude and latitude')
plt.savefig('figures/lon_lat_2d_hist.png')
plt.clf()

#plot posting per hour historgram

data['hour'] = data['created'].dt.hour
plt.hist(data['hour'], bins=23)
plt.title('Number of postings per hour of day')
plt.xlabel('Hour (24 mod based)')
plt.ylabel('Number of postings')
plt.savefig('figures/hourly_postings.png')
plt.clf()


#Top 5 hours

posting_counts = data['hour'].value_counts().head(5)
print("Top 5 Postings per hours in sorted order:")
print(posting_counts)

# Proportion of target variable values

# interest_level histogram
data['interest_level'].value_counts().plot(kind = 'bar')
plt.xlabel('Interest level')
plt.ylabel('Postings')
plt.savefig('figures/interest_level_hist.png')
plt.clf()

# Pie chart
data['interest_level'].value_counts().plot(kind = 'pie')
plt.savefig('figures/interest_level_pie.png')
plt.clf()