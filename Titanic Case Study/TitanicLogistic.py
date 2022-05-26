import pandas as pd
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

########################################################################################################################
## Function name :  TitanicLogistic   
## Input :          Titanic.csv
## Output :         Accuracy using Logistic Regression
## Description :    Accuracy on Training and Testing Data Set using LogisticRegression
## Author :         Girish Pradeep Pati
## Date :           15/03/2022
########################################################################################################################

def TitanicLogistic():

    #Step 1 : Load Data
    titanic_data=pd.read_csv('MarvellousTitanic.csv')

    print("First 5 entries from the dataset")
    print(titanic_data.head())

    print("Number of passengers arae",str(len(titanic_data)))

    #Step 2 : Analyze data
    print("Visualization : Survived and non survived passengers")
    figure()
    target = "Survived"

    countplot(data=titanic_data,x=target).set_title("Survived and non survived passengers")
    plt.savefig("Plot1")
    show()

    print("Visualization : Survived and non survived passengers based on Gender")
    figure()
    target="Survived"

    countplot(data=titanic_data,x=target,hue="Sex").set_title("Survived and non survived passengers based on Gender")
    plt.savefig("Plot2")
    show()

    print("Visualization : Survived and non survived passengers based on Passenger Class")
    figure()
    target="Survived"

    countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and non survived passengers based on Passenger Class")
    plt.savefig("Plot3")
    show()

    print("Visualization : Survived and non survived passengers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non survived passengers based on Age")
    plt.savefig("Hist")
    show() 

    print("First 5 entries from the loaded dataset after removing column zero")
    print(titanic_data.head(5))

    print("Values of Sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of Sex column after removing one field")
    Sex=print(pd.get_dummies(titanic_data["Sex"],drop_first=True))
    
    print("Values of Pclass column after removing one field")
    Pclass=print(pd.get_dummies(titanic_data["Pclass"],drop_first=True))

    print("Values of data set after concatenating new columns")
    titanic_data=pd.concat([titanic_data,Sex,Pclass],axis=1)
    print(titanic_data.head(5))

    print("Value of data set after removing irrelevant columns")
    titanic_data.drop(["Sex","SibSp","Parch",],axis=1,inplace=True)
    print(titanic_data.head(5))

    updated_df = titanic_data
    updated_df['Age']=updated_df['Age'].fillna(updated_df['Age'].mean())
    print(updated_df.info())

    updated_df = titanic_data
    updated_df['Age']=updated_df['Age'].fillna(updated_df['Age'].mean())
    print(updated_df.info())

    x = titanic_data.drop("Survived",axis=1)
    y = titanic_data["Survived"]

    # Step 4 : Data Training
    xtrain,xtest,ytrain,ytest=train_test_split(x,y, test_size=0.5) 

    logmodel = LogisticRegression()

    logmodel.fit(xtrain,ytrain)

    #Step 4 : Data Testing
    predictor=logmodel.predict(xtest)

    # Step 5 : Accuracy Calulation
    print("Classification report of Logistic Regression is: ")
    print(classification_report(ytest,predictor))

    print("Confusion Matrix of Logistic Regression is: ")
    print(confusion_matrix(ytest,predictor))

    print("Accuracy of Logistic Regression is: ")
    print(accuracy_score(ytest,predictor))    


########################################################
## Starter function for TitanicLogistic function
########################################################
def main():
    print("-----Girish Patil-----")

    print("Logistic Regression on Titanic Data set")

    TitanicLogistic()

if __name__=="__main__":
    main()
