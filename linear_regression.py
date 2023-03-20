import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    data = pd.read_csv('linear_regression_data.csv')
    m = 500
    b = 0
    L = 0.0001
    epochs = 2000
    
    for i in range(epochs):
        if i % 100 == 0:
            print(f"Epoch {i}: m: {m:.2f}, b: {b:.2f}")
        m, b = gradient_descent(m, b, data, L)

    print(f"y = {m}x + {b}")    
    plt.scatter(x=data.GPA, y=data.SAT, color="black")
    x = np.linspace(start=2.5, stop=3.8)
    plt.plot(x, (m*x)+b, color="red")
    plt.show()

def gradient_descent(m_now, b_now, points, L: float) -> tuple[float, float]:
    m_gradient, b_gradient = 0, 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].GPA
        y = points.iloc[i].SAT
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b
   
    
if __name__ == "__main__":
    main()