from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

####################################################################################
## Function name :  CalculateAccuracyDecisionTree   
## Input :          iris dataset
## Output :         Accuracy of model
## Description :    To solve iris case study  and try to achieve maximum accuracy
## Author :         Girish Pradeep Patil
## Date :           10/03/2022
####################################################################################
def CalculateAccuracyDecisionTree():
    iris=load_iris()

    data = iris.data
    target = iris.target

    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(data_train,target_train)

    predictor = classifier.predict(data_test)

    Accuracy=accuracy_score(target_test,predictor)

    return Accuracy

################################################################
## Starter function for CalculateAccuracyDecisionTree function
################################################################
def main():
    Accuracy=CalculateAccuracyDecisionTree()
    print("Accuracy of classification with Decision Tree Classifier is",Accuracy*100,"%")

if __name__=="__main__":
    main()
