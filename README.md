# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 
 

## Program:


Program to implement the linear regression using gradient descent.

Developed by: shivaji.k

RegisterNumber:25018038  

```

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1)*X.T.dot(errors))
    return theta
data=pd.read_csv('50_startups.csv',header=None)
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```

## Output:

<img width="676" height="128" alt="image" src="https://github.com/user-attachments/assets/bbe5e833-83bb-45a5-9aeb-1105475d8fdc" />






<img width="458" height="550" alt="image" src="https://github.com/user-attachments/assets/14be9704-0566-4e71-9ddd-4989d40cdd6f" />






<img width="182" height="545" alt="image" src="https://github.com/user-attachments/assets/7ada450e-cdd3-4ab8-988c-f11ad25e690c" />



















<img width="566" height="540" alt="image" src="https://github.com/user-attachments/assets/3b3ee96e-af4a-4539-af12-1e88b6966f8f" />



















<img width="288" height="784" alt="image" src="https://github.com/user-attachments/assets/00c351e4-f589-478c-a8a9-b597c20d9a27" />














<img width="370" height="44" alt="image" src="https://github.com/user-attachments/assets/77f5aaa8-cdb2-468e-b798-7d32f8e1e7ae" />









## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
