import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter

########################################################################################################
##
## Function name :  DiabetesPredictor    
## Input :          diabetes.csv
## Output :         Accuracy using LogisticRegression
## Description :    Accuracy on Training and Testing Data Set by changing value of C
## Author :         Girish Pradeep Patil
## Date :           07/03/2022
##
########################################################################################################

def DiabetesPredictor():
    simplefilter(action='ignore',category=FutureWarning)

    diabetes=pd.read_csv('diabetes.csv')

    print("Columns of Dataset are:")
    print(diabetes.columns)

    print("First 5 records of dataset are:")
    print(diabetes.head())

    print("Dimension of diabetes data: {}".format(diabetes.shape))

    x_train,x_test,y_train,y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'],stratify= diabetes['Outcome'],random_state=66)

    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)

    print("Training set Accuracy using Logistic Regression is : {:.3f}".format(logreg.score(x_train,y_train)))
    print("Testing set Accuracy using Logistic Regression is : {:.3f}".format(logreg.score(x_test,y_test)))

    logreg01 = LogisticRegression(C=0.05)
    logreg01.fit(x_train,y_train)

    print("Training set Accuracy using Logistic Regression is : {:.3f}".format(logreg01.score(x_train,y_train)))
    print("Testing set Accuracy using Logistic Regression is : {:.3f}".format(logreg01.score(x_test,y_test)))
    

########################################################
## Starter function for DiabetesPredictor function
########################################################
def main():

    print("-----Girish Patil-----")
    print("------Diabetes Predictor using Logistic Regression-------")

    DiabetesPredictor()

if __name__=="__main__":
    main()