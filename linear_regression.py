import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.linear_model import LinearRegression

def main():
    data = pd.read_csv('linear_regression_data.csv')
    print(data.head())

    if len(sys.argv) > 1 and sys.argv[1] == "sklearn":
        print("Calculating linear regression using scikit-learn")
        x = data["GPA"]
        y = data["SAT"]
        x_matrix = x.values.reshape(-1,1)
        reg = LinearRegression()
        reg.fit(X=x_matrix, y=y)
        display_sklearn(x, x_matrix, y, reg)
        
    else:
        print("Calculating linear regression using successive derivatives")
        lr = LR(m=245, b=1000, L=0.0001, epochs=1000, data=data)
        lr.linear_regression_implementation()
        lr.display()
    
def display_sklearn(x: np.array, x_matrix: np.ndarray, y: np.array, reg: LinearRegression):
    plt.scatter(x,y)
    print(f"y={reg.coef_[0]}x + {reg.intercept_}")
    yhat = reg.coef_ * x_matrix + reg.intercept_
    plt.plot(x, yhat, lw=4, c='red')
    plt.xlabel('SAT')
    plt.ylabel('GPA')
    plt.show()

class LR:
    def __init__(self, m, b, L, epochs, data):
        self.m = m
        self.b = b
        self.L = L
        self.epochs = epochs
        self.data = data
        self.n = len(data)

    def linear_regression_implementation(self):
        for i in range(self.epochs):
            if i % 100 == 0:
                print(f"Epoch {i}: m: {self.m:.2f}, b: {self.b:.2f}")
            self.gradient_descent()

    def gradient_descent(self):
        m_gradient, b_gradient = 0, 0

        for i in range(self.n):
            x = self.data.iloc[i].GPA
            y = self.data.iloc[i].SAT
            m_gradient += -(2/self.n) * x * (y - (self.m * x + self.b))
            b_gradient += -(2/self.n) * (y - (self.m * x + self.b))

        self.m = self.m - m_gradient * self.L
        self.b = self.b - b_gradient * self.L
    
    def display(self):
        print(f"y = {self.m}x + {self.b}")    
        plt.scatter(x=self.data.GPA, y=self.data.SAT)
        x = np.linspace(start=2.5, stop=3.8)
        plt.xlabel('SAT')
        plt.ylabel('GPA')
        plt.plot(x, (self.m*x)+self.b, lw=4, c='red')
        plt.show()
   
if __name__ == "__main__":
    main()