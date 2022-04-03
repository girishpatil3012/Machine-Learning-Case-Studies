import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sb

################################################################################################################################
## Function name :  LinearReg  
## Input :          Advertising.csv
## Output :         R square value
## Description :    To solve Wine Class Predictor case study using Linear Regression and try to achieve maximum R Square value
## Author :         Girish Pradeep Patil
## Date :           11/03/2022
###############################################################################################################################
def LinearReg():
    data=pd.read_csv('Advertising.csv')
    print(data)

    Y=data.sales
    X=data.drop(f'sales',axis=1)

    regressor=LinearRegression()
    regressor.fit(X,Y)

    predictor=regressor.predict(X)


    r2=regressor.score(X,Y)
    print(r2)
    
    plt.scatter(Y,predictor)
    plt.xlabel('TV,Radio,Newspaper')
    plt.ylabel('Sales')
    plt.savefig("Sales_regression")
    plt.show()

    sb.lmplot(x = "TV",
            y = "sales", 
            ci = None,
            data = data)

################################################################
## Starter function for LinearReg function
################################################################
def main():
    print("-"*70)
    print("Girish Patil")
    print("-"*70)

    print("-"*70)
    print("Advertising case study using linear regression")
    print("-"*70)

    LinearReg()

if __name__=="__main__":
    main()
