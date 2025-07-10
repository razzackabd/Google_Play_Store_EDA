
# Google Play Store EDA Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Google_data.csv")

# Clean 'Installs' column
if data['Installs'].dtype == object:
    data['Installs'] = data['Installs'].str.replace('[+,]', '', regex=True)
    data = data[data['Installs'].str.isnumeric()]
    data['Installs'] = data['Installs'].astype(int)

# Clean 'Price' column
if data['Price'].dtype == object:
    data['Price'] = data['Price'].str.replace('$', '', regex=True)
    data = data[data['Price'].str.replace('.', '', regex=False).str.isnumeric()]
    data['Price'] = data['Price'].astype(float)

# Convert 'Reviews' column
if data['Reviews'].dtype == object:
    data['Reviews'] = data['Reviews'].astype(int)

# Drop null values in 'Rating'
data = data[data['Rating'].notnull()]

# Distribution plot of Ratings
plt.figure(figsize=(10,6))
sns.histplot(data['Rating'].dropna(), kde=True, bins=20, color='skyblue')
plt.title("Distribution of App Ratings")
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Ratings (Before log transform)
plt.figure(figsize=(10,6))
sns.boxplot(x='Rating', data=data, color='lightgreen')
plt.title("Boxplot of App Ratings")
plt.xlabel('Rating')
plt.show()

# Log Transform Ratings
data['Rating'] = np.log1p(data['Rating'])

# Boxplot of Ratings (After log transform)
plt.figure(figsize=(10,6))
sns.boxplot(x='Rating', data=data, color='lightgreen')
plt.title("Boxplot of Log Transformed Ratings")
plt.xlabel('Rating')
plt.show()

# Outlier Detection using IQR
Q1 = data['Rating'].quantile(0.25)
Q3 = data['Rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q1 + 1.5 * IQR

# Filter data without outliers
filterdata = data[(data['Rating'] >= lower_bound) & (data['Rating'] <= upper_bound)]

# Boxplot after removing outliers
plt.figure(figsize=(10,6))
sns.boxplot(x='Rating', data=filterdata, color='lightgreen')
plt.title("Boxplot after Removing Outliers")
plt.xlabel('Rating')
plt.show()
