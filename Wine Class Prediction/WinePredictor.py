import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

################################################################################################################################
## Function name :  WinePredictor  
## Input :          WinePredictor.csv
## Output :         Accuracy of model
## Description :    To solve Wine Class Predictor case study using KNeighbours Classifer and try to achieve maximum accuracy
## Author :         Girish Pradeep Patil
## Date :           11/03/2022
###############################################################################################################################
def WinePredictor():
    data=pd.read_csv('WinePredictor.csv')
    print(data.head())

    Y=data.Class
    #data.drop('Hue',axis=1,inplace=True)
    data.drop('Color_intensity',axis=1,inplace=True)
    #data.drop('Ash',axis=1,inplace=True)
    X=data.drop('Class',axis=1)
    print(X)

    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3)

    classifier=KNeighborsClassifier(n_neighbors=3)
    classifier.fit(xtrain,ytrain)

    predictor=classifier.predict(xtest)

    Accuracy=accuracy_score(ytest,predictor)
    print("Accuracy using KNN is:",Accuracy*100,"%")

    plt.scatter(ytest,predictor)
    plt.xlabel('Malic_acid%')
    plt.ylabel('Class of Wine')
    plt.show()

    #drop some columns from dataset which are not necessary
    #crosscheck accuracy after dropping particular column
    #display columns which are altered
    #then go for training testing and accuracy

################################################################
## Starter function for WinePredictor function
################################################################
def main():
    print("-"*70)
    print("Girish Patil")
    print("-"*70)

    print("-"*70)
    print("Wine Predictor Case Study Using external csv KNN")
    print("-"*70)

    WinePredictor()

if __name__=="__main__":
    main()
