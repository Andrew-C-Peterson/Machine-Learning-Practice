#Auto_MPG_data

#Using car data again (different set), but this time I want to try and predict MPG
#based off of attributs such as weight, cylinders, horsepower

#Let's see how these models work

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
z = ['Weight','Horsepower','Cylinders', 'Displacement','Model Year']
lr = LinearRegression()
lr.fit(x_train[z], y_train)

#Prediction using training data
yhat_train = lr.predict(x_train[z])

#Prediction using test data
yhat_test = lr.predict(x_test[z])

import seaborn as sns
Title='Test data for multivariable linear'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
plt.figure()

Title='Train data for multivariable linear'
DistributionPlot(y_train,yhat_train,"Actual Values (Train)","Predicted Values (Train)",Title)
plt.figure()

from sklearn.preprocessing import PolynomialFeatures
#40% for testing, rest for training
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=1)


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
Title='Test Data for Polynomial'
DistributionPlot(y_test,yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)
Title='Train Data for Polynomial'
DistributionPlot(y_train, yhat_train1, "Actual Values (Train)", "Predicted Values (Train)", Title)



Z = x_train[z]
#Pipeline is a better way to do this
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree = 2)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y_train)
ypipe=pipe.predict(Z)
ypipe_test=pipe.predict(x_test[z])
Title='Test Data for Pipeline'
DistributionPlot(y_test, ypipe_test, "Actual Values (Train)", "Predicted Values (Train)", Title)
Title='Train Data for Pipeline'
DistributionPlot(y_train, ypipe, "Actual Values (Train)", "Predicted Values (Train)", Title)

#Ridge Regression
from sklearn.linear_model import Ridge
RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train_pr1, y_train)
yhat = RigeModel.predict(x_test_pr1)

Rsqu_test = []
Rsqu_train = []
dummy1 = []
ALFA = [.1,1,10,100,1000]
for alfa in ALFA:
    RigeModel = Ridge(alpha=alfa) 
    RigeModel.fit(x_train_pr1, y_train)
    Rsqu_test.append(RigeModel.score(x_test_pr1, y_test))
    Rsqu_train.append(RigeModel.score(x_train_pr1, y_train))

width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALFA,Rsqu_test, label='validation data  ')
plt.plot(ALFA,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()

#Grid search
from sklearn.model_selection import GridSearchCV
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000]}]
RR=Ridge()
Grid1 = GridSearchCV(RR, parameters1,cv=5)
Grid1.fit(x_data[z], y_data)
BestRR=Grid1.best_estimator_
BestRR.score(x_test[z], y_test)
parameters2= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000],'normalize':[True,False]} ]
Grid2 = GridSearchCV(Ridge(), parameters2,cv=5)
Grid2.fit(x_data[z],y_data)
print("")
print(Grid2.best_estimator_)

RigeModel = Ridge(alpha=100) 
RigeModel.fit(x_train_pr1, y_train)
yhat_ridgetrain = RigeModel.predict(x_train_pr1)
yhat_ridgetest = RigeModel.predict(x_test_pr1)

Title='Test Data for Pipeline'
DistributionPlot(y_test, yhat_ridgetest, "Actual Values (Train)", "Predicted Values (Train)", Title)
Title='Train Data for Ridge'
DistributionPlot(y_train, yhat_ridgetrain, "Actual Values (Train)", "Predicted Values (Train)", Title)



print('Train data score is: ', poly1.score(x_train_pr1, y_train))
print('Test data score is: ', poly1.score(x_test_pr1, y_test))
