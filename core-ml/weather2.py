from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt  
import pandas as pd

# Load data
data = pd.read_csv("weatherHistory.csv")

# Handle missing values in 'Precip Type' by filling with the most frequent value
data['Precip Type'] = data['Precip Type'].fillna(data['Precip Type'].mode()[0])

# Select categorical columns
COLUMNS = data[['Summary', 'Precip Type', 'Temperature (C)', 'Apparent Temperature (C)',
                'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)', 'Daily Summary']]

# Select numerical columns
NUMERICAL = data[['Wind Bearing (degrees)', 'Loud Cover']]

# Use OrdinalEncoder to encode categorical columns
ordinal_encoder = OrdinalEncoder()
COLUMNS_NUMERICAL = ordinal_encoder.fit_transform(COLUMNS)

# Convert back to DataFrame for better readability
COLUMNS_NUMERICAL = pd.DataFrame(COLUMNS_NUMERICAL, columns=COLUMNS.columns)

# Combine processed categorical and numerical data
DATA = pd.concat([COLUMNS_NUMERICAL, NUMERICAL], axis=1)

# Features and target
X_DATA = DATA[['Summary', 'Precip Type', 'Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 
               'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 
               'Loud Cover', 'Pressure (millibars)']]
y_DATA = DATA['Daily Summary']

# Split the testing data and training data
X_train, X_test, y_train, y_test = train_test_split(X_DATA, y_DATA, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict the target values
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Model Accuracy (R2 Score): {r2}")


plt.scatter(y_test, y_pred, color='purple')

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted ')
plt.show()








