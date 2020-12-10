#Auto_MPG_data

#Using car data again (different set), but this time I want to try and predict MPG
#based off of attributs such as weight, cylinders, horsepower

#I'm looking for correlation here, and deciding which variables are important

#Cylinders looks important, maybe as a polynomial type
#The data for 3 and 5 is really small though, which makes it hard

#Origins and Model year appear to show some small importance

#Acceleration shows small importance, linear type
#Not strong correlation, but statistically significant

#Horsepower and displacment both appear to show a polynomial type fit
#However, horsepower and displacement are directly correlated
#May be best to only use one of these

#Weight looks very important, maybe with a polynomial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("clean_auto_df.csv")
#I got basically an extra indexing column here from when I saved it, so I remove
df.drop("Unnamed: 0",axis=1, inplace=True)


#Correlation for all variables, just to check
corr = df.corr()

#How about acceleration?
sns.regplot(x="Acceleration", y="MPG", data=df)
plt.ylim(0,)
plt.figure()

pearson_coef, p_value = stats.pearsonr(df['Acceleration'], df['MPG'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


sns.boxplot(x="Cylinders", y="MPG", data=df)
plt.figure()
sns.boxplot(x="Origin", y="MPG", data=df)
plt.figure()
sns.boxplot(x="Model Year", y="MPG", data=df)
plt.figure()

cyls = df['Cylinders'].value_counts()

sns.regplot(x="Displacement", y="MPG", data=df)
plt.ylim(0,)
plt.figure()

sns.regplot(x="Horsepower", y="MPG", data=df)
plt.ylim(0,)
plt.figure()

sns.regplot(x="Displacement", y="Horsepower", data=df)
plt.ylim(0,)
plt.figure()

sns.regplot(x="Weight", y="MPG", data=df)
plt.ylim(0,)
plt.figure()
