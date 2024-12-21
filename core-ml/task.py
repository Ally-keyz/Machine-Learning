import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt


data = pd.read_csv("house_prices.csv")

#feature and price differenciation

FEACTURES = data[['Size (sq ft)','Bedrooms','Age (years)']]
PRICE = data['Price']

#prepare the data to feed the model

X_train , X_test, y_train , y_test = train_test_split(FEACTURES,PRICE,test_size=0.2,random_state=42)

#scale the data to feed the model 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#build the model
model = LinearRegression()

#train the model
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#evaluation from the model

mse = mean_squared_error(y_test,y_pred)
print(f"mean squared error:{mse}")
print("R-square score:",r2_score(y_test,y_pred))

#Plot from the evaluation

plt.scatter(y_test, y_pred, color='purple')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

