import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt  

data = pd.read_csv('car_prices.csv')

#separate the data
Features =data[[
    'Year', 'Mileage (km)', 'Engine Size (cc)'
]]

Prices = data['Price']

#split the training and testing data

X_train , X_test , y_train , y_test = train_test_split(Features,Prices,test_size=0.2,random_state=42) 

#scale the data
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#model
model = LinearRegression()
#train the model
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#calculate the prediction and the mean sqaure error
mse = mean_squared_error(y_test,y_pred)
print(f"mean squared error:{mse}")
print("Model accuracy:",r2_score(y_test,y_pred))

#plot the graph

plt.scatter(y_test, y_pred, color='purple')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()



