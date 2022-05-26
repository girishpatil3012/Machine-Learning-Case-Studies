import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter

########################################################################################################################
## Function name :  DiabetesPredictor    
## Input :          diabetes.csv
## Output :         Accuracy and Feature Importance
## Description :    Accuracy on Training and Testing Data Set and Feature Importances using Random Forest Classifier
## Author :         Girish Pradeep Patil
## Date :           07/03/2022
########################################################################################################################

def DiabetesPredictor():
    simplefilter(action='ignore',category=FutureWarning)

    diabetes=pd.read_csv('diabetes.csv')

    print("Columns of Dataset are:")
    print(diabetes.columns)

    print("First 5 records of dataset are:")
    print(diabetes.head())

    print("Dimension of diabetes data: {}".format(diabetes.shape))

    x_train,x_test,y_train,y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'],stratify= diabetes['Outcome'],random_state=66)

    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(x_train,y_train)
    print("Training set Accuracy using Random Forest Classifier is : {:.3f}".format(rf.score(x_train,y_train)))
    print("Testing set Accuracy using Random Forest Classifier is : {:.3f}".format(rf.score(x_test,y_test)))

    rf1 = RandomForestClassifier(max_depth=3,n_estimators=100, random_state=0)
    rf1.fit(x_train,y_train)
    print("Training set Accuracy using Random Forest Classifier is : {:.3f}".format(rf1.score(x_train,y_train)))
    print("Testing set Accuracy using Random Forest Classifier is : {:.3f}".format(rf1.score(x_test,y_test)))
    
    ############################################################################
    ## Function name :  plot_feature_importances    
    ## Input :          rf
    ## Output :         Feature Importance
    ## Description :    To plot Feature Importance Bar graph using Matplotlib
    ############################################################################
    def plot_feature_importances(model):
        plt.figure(figsize=(8,6))
        n_features = 8
        plt.barh(range(n_features), model.feature_importances_,align='center')
        diabetes_features = [x for i ,x in enumerate(diabetes.columns) if i != 8]
        plt.yticks(np.arange(n_features),diabetes_features)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.ylim(-1, n_features)
        plt.savefig('Random_Forest_model')
        plt.show()
    
    return plot_feature_importances(rf)

########################################################
## Starter function for DiabetesPredictor function
########################################################
def main():
    print("-----Girish Patil-----")
    print("------Diabetes Predictor using Random Forest Classifier-------")

    DiabetesPredictor()

if __name__=="__main__":
    main()