#Auto_MPG_data

#Using car data again (different set), but this time I want to try and predict MPG
#based off of attributs such as weight, cylinders, horsepower

#Try and develop models to fit the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Import my data
df = pd.read_csv("clean_auto_df.csv")
#I got basically an extra indexing column here from when I saved it, so I remove
df.drop("Unnamed: 0",axis=1, inplace=True)

#Create the linear regression
lm = LinearRegression()
#These are the variables we will compare
X = df[['Weight']]
Y = df['MPG']
#And now perform the regression
lm.fit(X,Y)
#This is the prediction based on the regression
Yhat=lm.predict(X)
#print the intercept and slope
print(lm.intercept_)
print(lm.coef_)

#Now, multiple linear regression
Z = df[['Horsepower', 'Weight', 'Cylinders','Model Year','Displacement']]
lm.fit(Z, df['MPG'])

#Data visualization
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="Weight", y="MPG", data=df)
plt.ylim(0,)
plt.figure()

plt.figure(figsize=(width, height))
sns.regplot(x="Horsepower", y="MPG", data=df)
plt.ylim(0,)
plt.figure()

#Residual plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['Weight'], df['MPG'])
plt.show()
plt.figure()

#Multiple linear regression, distribution plot
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['MPG'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for MPG')
plt.xlabel('MPG')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()
plt.figure()

#Function for plotting
def PlotPolly(model, independent_variable, dependent_variabble, Name,x_new,y_new):
    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('MPG')

    plt.show()
    plt.close()
    
x = df['Weight']
y = df['MPG']

# Here we use a polynomial
f = np.polyfit(x, y, 2)
p = np.poly1d(f)
x_new = np.linspace(1000, 5500, 100)
y_new = p(x_new)
PlotPolly(p, x, y, 'Weight',x_new,y_new)

plt.figure()

x = df['Cylinders']
y = df['MPG']

# Here we use a polynomial of the 2nd order 
f = np.polyfit(x, y, 2)
p = np.poly1d(f)
x_new = np.linspace(0, 10, 100)
y_new = p(x_new)
PlotPolly(p, x, y, 'Cylinders',x_new,y_new)

plt.figure()

pr=PolynomialFeatures(degree=2)
Z_pr=pr.fit_transform(Z,Y)


#Pipeline
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)

#finding R^2
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

#Calculate MSE
Yhat=lm.predict(X)

print('The output of the first four predicted value is: ', Yhat[0:4])
mse = mean_squared_error(df['MPG'], Yhat)
print('The mean square error of MPG and predicted value is: ', mse)

lm.fit(Z, df['MPG'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['MPG']))
Y_predict_multifit = lm.predict(Z)
print('The mean square error of MPG and predicted value using multifit is: ', \
      mean_squared_error(df['MPG'], Y_predict_multifit))
