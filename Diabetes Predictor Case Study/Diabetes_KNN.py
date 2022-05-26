import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

########################################################################################################
##
## Function name :  DiabetesPredictor    
## Input :          diabetes.csv
## Output :         Accuracy and best value of k using KNeighborsClassifier
## Description :    Accuracy on Training and Testing Data Set and finding the appropriate value of k
## Author :         Girish Pradeep Patil
## Date :           07/03/2022
##
########################################################################################################

def DiabetesPredictor():
    
    diabetes = pd.read_csv('diabetes.csv')

    print("Columns of Dataset are:")
    print(diabetes.columns)

    print("First 5 records of dataset are:")
    print(diabetes.head())

    print("Dimension of diabetes data: {}".format(diabetes.shape))

    x_train,x_test,y_train,y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'],stratify= diabetes['Outcome'],random_state=66)

    training_accuracy = []
    test_accuracy = []

    #try n_neighbors from 1 to 10
    neighbors_settings = range(1,11)

    for n_neighbors in neighbors_settings:
        # Build the Model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_train,y_train)

        # Record training set accuracy
        training_accuracy.append(knn.score(x_train,y_train))

        # Record test set accuracy
        test_accuracy.append(knn.score(x_test,y_test))
    
    ################################################################################################
    ##
    ## Function name :  plot_best_k    
    ## Input :          values of k in range(1,11)
    ## Output :         Best value of k and Accuracy of Model
    ## Description :    To find best value of k and calculate Accuracy on Training and Test Dataset
    ##
    #################################################################################################
    def plot_best_k():
        plt.plot(neighbors_settings, training_accuracy, label= "Training Accuracy")
        plt.plot(neighbors_settings, test_accuracy, label="Test Accuracy")
        plt.xlabel("n_neighbors")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('knn_compare_model')
        plt.show()
        knn = KNeighborsClassifier(n_neighbors=9)
        knn.fit(x_train,y_train)

        print("Accuracy of KNN Classifier on Training Set is : {:.2f}".format(knn.score(x_train,y_train)))

        print("Accuracy of KNN Classifier on Testing Set is : {:.2f}".format(knn.score(x_test,y_test)))

    return plot_best_k()
 
########################################################
## Starter function for DiabetesPredictor function
########################################################
def main():

    print("-----Girish Patil-----")
    print("------Diabetes Predictor using K Neighbors Classifier-------")

    DiabetesPredictor()

if __name__=="__main__":
    main()
