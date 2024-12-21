import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import customtkinter as ctk

# Load the dataset
data = {
    "Date": ["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04", "2024-11-05"],
    "Temperature": [20, 22, 19, 21, 23],
    "Humidity": [65, 70, 60, 75, 68],
    "Wind Speed": [10, 12, 9, 11, 8]
}
df = pd.DataFrame(data)

# Convert date to numerical feature (days since start)
df["Days"] = pd.to_datetime(df["Date"]).map(lambda x: (x - pd.to_datetime("2024-11-01")).days)
X = df[["Days", "Humidity", "Wind Speed"]]
y = df["Temperature"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Model Performance:")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Function for making predictions
def predict_temperature(days, humidity, wind_speed):
    input_data = [[days, humidity, wind_speed]]
    return model.predict(input_data)[0]

# Create the app with CustomTkinter
class WeatherApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Weather Prediction App")
        self.geometry("400x300")
        self.configure(bg="white")

        self.init_ui()

    def init_ui(self):
        # Input fields
        self.label_days = ctk.CTkLabel(self, text="Days since 2024-11-01:")
        self.label_days.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.entry_days = ctk.CTkEntry(self)
        self.entry_days.grid(row=0, column=1, padx=10, pady=10)

        self.label_humidity = ctk.CTkLabel(self, text="Humidity (%):")
        self.label_humidity.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.entry_humidity = ctk.CTkEntry(self)
        self.entry_humidity.grid(row=1, column=1, padx=10, pady=10)

        self.label_wind_speed = ctk.CTkLabel(self, text="Wind Speed (km/h):")
        self.label_wind_speed.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.entry_wind_speed = ctk.CTkEntry(self)
        self.entry_wind_speed.grid(row=2, column=1, padx=10, pady=10)

        # Predict button
        self.predict_button = ctk.CTkButton(self, text="Predict Temperature", command=self.predict_weather)
        self.predict_button.grid(row=3, column=0, columnspan=2, pady=20)

        # Result label
        self.result_label = ctk.CTkLabel(self, text="")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)

    def predict_weather(self):
        try:
            # Get user inputs
            days = int(self.entry_days.get())
            humidity = int(self.entry_humidity.get())
            wind_speed = int(self.entry_wind_speed.get())

            # Predict temperature using the trained model
            predicted_temp = predict_temperature(days, humidity, wind_speed)
            self.result_label.configure(text=f"Predicted Temperature: {predicted_temp:.2f}°C")
        except ValueError:
            self.result_label.configure(text="Invalid input, please enter valid numbers.")

# Run the application
if __name__ == "__main__":
    app = WeatherApp()
    app.mainloop()
