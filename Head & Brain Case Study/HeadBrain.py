import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#####################################################################################################################
## Function name :  HeadBrainPredictor    
## Input :          HeadBrain.csv
## Output :         Accuracy using Linear Regression
## Description :    Accuracy on Training and Testing Data Set using User Defined Linear Regression
## Author :         Girish Pradeep Patil
## Date :           10/03/2022
####################################################################################################################

def HeadBrainPredictor():
    data=pd.read_csv('HeadBrain1.csv')

    print("Size of data is:",data.shape)

    X=data['Head Size(cm^3)'].values
    Y=data['Brain Weight(grams)'].values

    X=X.reshape((-1,1))
    n=len(X)

    reg = LinearRegression()
    reg = reg.fit(X,Y)

    pred=reg.predict(X)

    r2 = reg.score(X,Y)
    print("Accuracy using Linear Regression is : ",r2)

########################################################
## Starter function for HeadBrainPredictor function
######################################################## 
def main():
    print('-----Girish Patil-------')
    print("Linear Regression on Head and Brain Data set")

    HeadBrainPredictor()

if __name__=="__main__":
    main()
