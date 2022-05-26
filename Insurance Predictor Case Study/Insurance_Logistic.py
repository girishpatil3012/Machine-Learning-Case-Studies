import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

########################################################################################################
## Function name :  InsurancePredictor  
## Input :          insurance_data.csv
## Output :         Accuracy using Linear Regression
## Description :    Accuracy on Training and Testing Data Set using Linear Regression
## Author :         Girish Pradeep Patil
## Date :           17/03/2022
#######################################################################################################

def InsurancePredictor():

    data=pd.read_csv("insurance_data.csv")
    print("_"*50)

    print("Columns of Dataset are:")
    print(data.columns)

    print("First 5 records of dataset are:")
    print(data.head())

    print("Dimension of diabetes data: {}".format(data.shape))

    print("_"*50)
    plt.scatter(data.age,data.bought_insurance,marker='+',color='red')
    plt.savefig('insurance_model')
    plt.show()

    x_train,x_test,y_train,y_test=train_test_split(data[['age']],data.bought_insurance,train_size=0.5)
    print("Independent variabbles for training : ")
    print(x_train)

    print("_"*50)
    print("Dependent variabbles for training : ")
    print(y_train)

    print("_"*50)
    print("Independent variabbles for testing : ")
    print(x_test)

    print("_"*50 )
    print("Dependent variabbles for testing : ")
    print(y_test)

    model=LogisticRegression()
    model.fit(x_train,y_train)

    print("_"*50)
    predictor=model.predict(x_test)
    print(predictor)

    print("_"*50)
    probab=model.predict_proba(x_test)
    print("Probablity of model is : ")
    print(probab)

    print("_"*50)
    print("Classification report of Logistic Regression is : ")
    print(classification_report(y_test,predictor))

    print("_"*50)
    print("Confusion Matrix of Logistic Regression is : ")
    print(confusion_matrix(y_test,predictor))

    print("_"*50)
    print("Accuracy of Logistic Regresion is : ",accuracy_score(y_test,predictor))
    print("_"*50)


########################################################
## Starter function for InsurancePredictor function
######################################################## 
def main():
    print("------Girish Patil-------")
    print("------Logistic Regression on Insurance Dataset--------")
    print("_"*50)
    
    InsurancePredictor()

if __name__=="__main__":
    main()
