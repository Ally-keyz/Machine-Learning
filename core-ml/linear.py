import numpy as np
import pandas as pd

data = {
    'Experience (years)': [1, 2, 3, 4, 5],
    'Salary (k$)': [30, 35, 40, 45, 50]
}

df = pd.DataFrame(data)

X = df[['Experience (years)']]
Y = df['Salary (k$)']


from sklearn.model_selection import train_test_split


X_train , X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

#model

model = LinearRegression()

#Train the model
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test,y_pred)
print(f"mean squared error:{mse}")
print("R-square score:",  model.score(Y_test,y_pred))

import matplotlib.pyplot as plt

# Scatter plot of actual vs predicted
plt.scatter(X_test, Y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Experience (years)')
plt.ylabel('Salary (k$)')
plt.legend()
plt.show()
