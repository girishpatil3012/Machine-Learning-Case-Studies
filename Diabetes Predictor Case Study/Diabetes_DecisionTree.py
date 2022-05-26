import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

########################################################################################################################
## Function name :  DiabetesPredictor    
## Input :          diabetes.csv
## Output :         Accuracy and Feature Importance using DecisionTreeClassifier
## Description :    Accuracy on Training and Testing Data Set and Feature Importances using Decision Tree Classifier
## Author :         Girish Pradeep Patil
## Date :           07/03/2022
########################################################################################################################

def DiabetesPredictor():

    diabetes = pd.read_csv('diabetes.csv')

    print("Columns of Dataset are:")
    print(diabetes.columns)

    print("First 5 records of dataset are:")
    print(diabetes.head())

    print("Dimension of diabetes data: {}".format(diabetes.shape))

    xtrain,xtest,ytrain,ytest = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'],stratify= diabetes['Outcome'],random_state=66)

    tree= DecisionTreeClassifier(random_state=0)
    tree.fit(xtrain,ytrain)

    print("Accuracy on training set : {:.3f}".format(tree.score(xtrain,ytrain)))
    print("Accuracy on test set : {:.3f}".format(tree.score(xtest,ytest)))

    tree= DecisionTreeClassifier(max_depth=3,random_state=0)
    tree.fit(xtrain,ytrain)

    print("Accuracy on training set after max_depth 3 is : {:.3f}".format(tree.score(xtrain,ytrain)))
    print("Accuracy on test set after max_depth 3 is : {:.3f}".format(tree.score(xtest,ytest)))

    print("Feature importances: \n {}".format(tree.feature_importances_))

    ############################################################################
    ## Function name :  plot_feature_importances    
    ## Input :          tree
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
        plt.show()
    
    return plot_feature_importances(tree)


########################################################
## Starter function for DiabetesPredictor function
########################################################  
def main():
    print("-----Girish Patil-----")

    print("------Diabetes Predictor using Decision Tree Classifier")

    DiabetesPredictor()

if __name__=="__main__":
    main()
