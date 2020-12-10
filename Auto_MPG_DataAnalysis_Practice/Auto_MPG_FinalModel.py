#Auto_MPG

#I like the polynomial regression of order 2, with the 5 features selected
#Here I'm just looking at how well it works
#Might try cross-validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

#Random int for random selection of train_test data
number = int(random.random()*10)

#Import my data
df = pd.read_csv("clean_auto_df.csv")
#I got basically an extra indexing column here from when I saved it, so I remove
df.drop("Unnamed: 0",axis=1, inplace=True)
#Numeric data only
df=df._get_numeric_data()
#The 5 features I think are important
z = ['Weight','Horsepower','Cylinders','Displacement','Model Year']

#Plotting function
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
    
    
#Make new dfs, with and without price  
y_data = df['MPG']
x_data=df.drop('MPG',axis=1)

#Split the data into training and testing data. I chose 25% for test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=number)

#MAke my polynomial order 2 
#Transform x train and test data, do linear fit on train data
#Find predicted values for test and train data
#Plot them
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[z])
x_test_pr=pr.fit_transform(x_test[z])
poly=LinearRegression().fit(x_train_pr,y_train)
yhat_train=poly.predict(x_train_pr)
yhat_test=poly.predict(x_test_pr)
Title='Train Data for Polynomial'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
Title='Test Data for Polynomial'
DistributionPlot(y_test,yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)

print('Train data score is: ', poly.score(x_train_pr, y_train))
print('Test data score is: ', poly.score(x_test_pr, y_test))

plt.scatter(y_train,yhat_train,label='Train Data')
plt.scatter(y_test,yhat_test,label='Test Data')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted MPG values")
plt.legend()
x=[]
for i in range(0,46):
    x.append(i)
plt.plot(x)
    
error = (((y_train - yhat_train)/y_train)**2)**(1/2)
error = error.tolist()

count=0
for i in range(0, len(error)):
    if error[i]<=0.10:
        count = count+1
error_per = 100*count/len(error)
print('This proportion of train data is within 10%: ', error_per)

error = (((y_test - yhat_test)/y_test)**2)**(1/2)
error = error.tolist()

count=0
for i in range(0, len(error)):
    if error[i]<=0.10:
        count = count+1
error_per = 100*count/len(error)
print('This proportion of test data is within 10%: ', error_per)
    
# prepare cross validation
k_fold = KFold(10,True,number)

y_data = y_data.to_frame()
y_predict = y_data.copy()
for k, (train, test) in enumerate(k_fold.split(df)):
    train = train.tolist()
    test=test.tolist()
    x_train_cv = x_data.iloc[train,:]
    x_test_cv=x_data.iloc[test,:]
    y_test_cv=y_data.iloc[test,0]
    y_train_cv=y_data.iloc[train,0]
    
    pr1=PolynomialFeatures(degree=2)
    x_train_cv1=pr1.fit_transform(x_train_cv[z])
    x_test_cv1=pr1.fit_transform(x_test_cv[z])
    poly1=LinearRegression().fit(x_train_cv1,y_train_cv)
    yhat_train_cv=poly1.predict(x_train_cv1)
    y_predict=poly1.predict(x_test_cv1)
    yhat_test_cv=poly1.predict(x_test_cv1)
    

Title='Test Data for Cross-Val Poly'
DistributionPlot(y_data,y_predict, "Actual Values (Test)", "Predicted Values (Test)", Title)