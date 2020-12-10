#Auto_MPG_data

#Using car data again (different set), but this time I want to try and predict MPG
#based off of attributs such as weight, cylinders, horsepower

#Let's see how these models work - Here I standardize the 'x' data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Import my data
df = pd.read_csv("clean_auto_df.csv")
#I got basically an extra indexing column here from when I saved it, so I remove
df.drop("Unnamed: 0",axis=1, inplace=True)
#Numeric data only
df=df._get_numeric_data()

#Plotting functions
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('MPG')
    plt.ylabel('Proportion of Cars')
    plt.xlim([0,60])

    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([0, 50])
    plt.ylabel('MPG')
    plt.legend()
    
#Make new dfs, with and without price  
y_data = df['MPG']
x_data=df.drop('MPG',axis=1)

x_data=(x_data-x_data.mean())/x_data.std()

#Split the data into training and testing data
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

#Linear regression
from sklearn.linear_model import LinearRegression
lre=LinearRegression()
print(lre.fit(x_train[['Weight']], y_train))
print(lre.score(x_test[['Weight']], y_test))
print(lre.score(x_train[['Weight']], y_train))

#Cross validation
from sklearn.model_selection import cross_val_score
#4 Folds, or iterations of cross validation
Rcross = cross_val_score(lre, x_data[['Weight']], y_data, cv=5)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre,x_data[['Weight']], y_data,cv=5)


#Multiple linear regression
z = ['Weight','Horsepower','Cylinders','Displacement','Origin']
lr = LinearRegression()
lr.fit(x_train[z], y_train)

#Prediction using training data
yhat_train = lr.predict(x_train[z])

#Prediction using test data
yhat_test = lr.predict(x_test[z])

import seaborn as sns
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
plt.figure()

Title='Distribution  Plot of  Predicted Value Using Train Data vs Data Distribution of Train Data'
DistributionPlot(y_train,yhat_train,"Actual Values (Train)","Predicted Values (Train)",Title)
plt.figure()

from sklearn.preprocessing import PolynomialFeatures
#40% for testing, rest for training
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.40, random_state=1)


#2 degree polynomial transformation of weight
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['Weight']])
x_test_pr = pr.fit_transform(x_test[['Weight']])

poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)

PollyPlot(x_train[['Weight']], x_test[['Weight']], y_train, y_test, poly,pr)
poly.score(x_train_pr, y_train)
poly.score(x_test_pr, y_test)
#Negative R^2 is a sign of overfitting

#Tries different order polynomials
Rsqu_test = []

order = [1, 2, 3, 4,5,6,7,8,9,10]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['Weight']])
    
    x_test_pr = pr.fit_transform(x_test[['Weight']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))
 
plt.figure()
plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')

pr1=PolynomialFeatures(degree=2)
x_train_pr1=pr1.fit_transform(x_train[z])
x_test_pr1=pr1.fit_transform(x_test[z])
poly1=LinearRegression().fit(x_train_pr1,y_train)
yhat_train1=poly1.predict(x_train_pr1)
yhat_test1=poly1.predict(x_test_pr1)
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)
itle='Distribution  Plot of  Predicted Value Using Train Data vs Data Distribution of Train Data'
DistributionPlot(y_train, yhat_train1, "Actual Values (Train)", "Predicted Values (Train)", Title)


