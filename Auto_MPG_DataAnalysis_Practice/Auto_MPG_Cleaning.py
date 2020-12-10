#Auto_MPG_data

#Using car data again (different set), but this time I want to try and predict MPG
#based off of attributs such as weight, cylinders, horsepower

#Here, I clean the data
#If the horsepower if missing, I replace it with the average for the 
#bin of displacement of that row - Low, Medium, or High

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot

#Import my data, csv with headers, replace all ? marks with NaN
auto_df = pd.read_csv("auto-mpg.csv")
auto_df.replace("?", np.nan, inplace = True)

#Make a df which shows all missing values as bool True
missing_data = auto_df.isnull()

#Shows which columns are missing data
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 

#Shows the types of column    
print(auto_df.dtypes)
    
#Horsepower is an object, it should be a float
auto_df[["Horsepower"]] = auto_df[["Horsepower"]].astype("float")

#In order to fill in the missing horsepower values, I think it should
#depend on displacment. I check that by seeing the correlation
print(auto_df[["Horsepower", "Displacement"]].corr())

#Makes my bins
bins = np.linspace(min(auto_df["Displacement"]), max(auto_df["Displacement"]), 4)
group_names = ['Low', 'Medium', 'High']
auto_df['Displacement-binned'] = pd.cut(auto_df['Displacement'], bins, labels=group_names, include_lowest=True )

#I find the means for each group
#I then split the df into slices for each group and replace 
#horsepower with the mean for that group. Then combine.
grouped = auto_df.groupby(['Displacement-binned']).mean()
means=[]
x = grouped.values.tolist()
for i in range(0,3):
    means.append(x[i][3])
    temp = auto_df.loc[(auto_df['Displacement-binned']==group_names[i])]
    temp['Horsepower'].replace(np.nan,means[i] , inplace = True)
    auto_df.update(temp)

#This changed my types - I want Cylinders, Model Year, and Origin to be int
auto_df[["Cylinders", "Model Year","Origin"]] = auto_df[["Cylinders", "Model Year","Origin"]].astype("int")

auto_df.to_csv('clean_auto_df.csv')

