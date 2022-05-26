import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

########################################################################################################
## Function name :  HeadBrainPredictor    
## Input :          HeadBrain.csv
## Output :         Accuracy using User Defined Linear Regression
## Description :    Accuracy on Training and Testing Data Set using User Defined Linear Regression
## Author :         Girish Pradeep Patil
## Date :           10/03/2022
#######################################################################################################

def HeadBrainPredictor():
    data=pd.read_csv('HeadBrain1.csv')

    print("Size of data is:",data.shape)

    X=data['Head Size(cm^3)'].values
    Y=data['Brain Weight(grams)'].values

    mean_x=np.mean(X)
    mean_y=np.mean(Y)

    n=len(X)

    numerator=0
    denominator=0

    # Equation of line is   y = mx + c

    for i in range(n):
        numerator += (X[i]-mean_x)*(Y[i]-mean_y)
        denominator += (X[i]-mean_x)**2

    m = numerator/denominator

    c = mean_y - (m*mean_x)

    print("Slope of Regression line is: ",m)
    print("Y intercept of Regression line is: ",c)
        
    max_x = np.max(X) + 100
    min_y = np.min(Y) - 100

    # Display plotting of above points
    x = np.linspace(min_y,max_x,n)
    y = c +m*x

    plt.plot(x,y, color='#58b970', label='Regression Line')
    plt.plot(x,y, color='#ef5423', label='Scatter Plot')

    plt.xlabel('Head size in cm^3')
    plt.ylabel('Brain weight in gram')
    plt.legend()
    plt.savefig("LinearRegression.jpg")
    plt.show() 

########################################################
## Starter function for HeadBrainPredictor function
######################################################## 
def main():
    print('-----Girish Patil-------')
    print("User Defined Linear Regression on Head and Brain Data set")

    HeadBrainPredictor()

if __name__=="__main__":
    main()