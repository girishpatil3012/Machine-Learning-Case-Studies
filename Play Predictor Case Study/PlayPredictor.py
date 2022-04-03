from array import array
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

##############################################################################################################
## Function name :  Accuracy  
## Input :          PlayPredictor.csv
## Output :         Accuracy of model
## Description :    To solve Play Predictor case study using KNeighbours Classifer and try to achieve maximum accuracy
## Author :         Girish Pradeep Patil
## Date :           11/03/2022
##############################################################################################################
def Accuracy():
    data=pd.read_csv('PlayPredictor.csv')
    #print(data)

    #print(data.head())
    #print(data.Wether)                             # sunny=2
                                                    #overcast=0
    #Weather=data.Wether                            # rainy=1
    #print(Weather)     
                                                    #hot=1
    #Temp=data.Temperature                          #mild=2
    #print(Temp)                                    #cool=0

                                                    #no=0
                                                    #yes=1

    #features=['Wether','Temperature']          
    #abc=data[features].apply(LabelEncoder().fit_transform)
    #print(data.head())
    
    le=preprocessing.LabelEncoder()
    weather_encoded=le.fit_transform(data.Wether)
    print(weather_encoded)
   
    temp_encoded=le.fit_transform(data.Temperature)
    print(temp_encoded)

    label=le.fit_transform(data.Play)
    print(label)

    features=list(zip(weather_encoded,temp_encoded))

    x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.5)

    classifier=KNeighborsClassifier(n_neighbors=3)
    classifier.fit(x_train,y_train)

    predictor=classifier.predict(x_test)
    print(predictor)

    score=accuracy_score(y_test,predictor)
    return score

################################################################
## Starter function for CalculateAccuracyKNN function
################################################################
def main():

    ret=Accuracy()
    print("Accuracy using KNN Classifier is:",ret*100,"%")

if __name__=="__main__":
    main()
