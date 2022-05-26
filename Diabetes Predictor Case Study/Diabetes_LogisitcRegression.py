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

    #logreg = LogisticRegression()
    #logreg.fit(x_train,y_train)

    #print("Training set Accuracy using Logistic Regression is : {:.3f}".format(logreg.score(x_train,y_train)))
    #print("Testing set Accuracy using Logistic Regression is : {:.3f}".format(logreg.score(x_test,y_test)))

    training_accuracy=[]
    test_accuracy=[]

    c_range=(0.01,0.10)
    for i in c_range:
        # Build the Model
        logreg01 =  LogisticRegression(C=i)
        logreg01.fit(x_train,y_train)

        # Record training set accuracy
        training_accuracy.append(logreg01.score(x_train,y_train))

        # Record test set accuracy
        test_accuracy.append(logreg01.score(x_test,y_test))

        #print(training_accuracy)       for finding best accuracy
        #print(test_accuracy)

    ################################################################################################
    ##
    ## Function name :  plot_best_c    
    ## Input :          values of c in range(0.01,0.10)
    ## Output :         Best value of c and Accuracy of Model
    ## Description :    To find best value of c and calculate Accuracy on Training and Test Dataset
    ##
    #################################################################################################
    def plot_best_c():
        plt.plot(c_range, training_accuracy, label= "Training Accuracy")
        plt.plot(c_range, test_accuracy, label="Test Accuracy")
        plt.xlabel("c_range")
        plt.ylabel("Regularization")
        plt.legend()
        plt.savefig('Logisitc_compare_model')
        plt.show()
        logreg01 = LogisticRegression(C=0.05)
        logreg01.fit(x_train,y_train)
        print("Training set Accuracy using Logistic Regression is : {:.3f}".format(logreg01.score(x_train,y_train)))
        print("Testing set Accuracy using Logistic Regression is : {:.3f}".format(logreg01.score(x_test,y_test)))
    
    return plot_best_c()   

########################################################
## Starter function for DiabetesPredictor function
########################################################
def main():

    print("-----Girish Patil-----")
    print("------Diabetes Predictor using Logistic Regression-------")

    DiabetesPredictor()

if __name__=="__main__":
    main()