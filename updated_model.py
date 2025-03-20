import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

# Load dataset
# df = pd.read_csv("D:/Hari/EEE/IT learning/semi-colons/archive123/nsw_property_data.csv")  # Replace with actual file
df = df9

# Sort by location and year
df = df.sort_values(by=["address", "settlement_date"])

# Normalize prices for better training
scaler = MinMaxScaler()
df["purchase_price"] = scaler.fit_transform(df[["purchase_price"]])








# Function to create time-series sequences
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Define sequence length (e.g., use past 5 years)
SEQ_LENGTH = 5

X, y = [], []

# Generate sequences for each location
for post_code in df["address"].unique():
    location_data = df[df["address"] == post_code]["purchase_price"].values
    seq_X, seq_y = create_sequences(location_data, SEQ_LENGTH)
    X.extend(seq_X)
    y.extend(seq_y)

X = np.array(X)
y = np.array(y)

# Reshape for simple linear model (flatten sequences)
X = X.reshape(X.shape[0], -1)









# Train Ridge Regression model (acts as a simple RNN alternative)
model = Ridge(alpha=0.1)
model.fit(X, y)

# Predict on training data
y_pred = model.predict(X)

# Calculate Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y, y_pred)
print(f"Mean Absolute Error: ${mae:,.2f}")


def predict_future_prices(post_code, recent_prices):
    # Normalize input
    recent_prices = scaler.transform(np.array(recent_prices).reshape(-1, 1))

    # Flatten for prediction
    recent_prices = np.array(recent_prices).reshape(1, -1)

    # Predict next price
    predicted_price = model.predict(recent_prices)

    # Convert back to original scale
    return scaler.inverse_transform([[predicted_price[0]]])[0][0]


# Example: Predict price for a location in 2025
post_code = "63 BEECH DR, SUFFOLK PARK"
recent_prices = [500000, 520000, 540000, 560000, 580000]  # Last 5 years

predicted_price = predict_future_prices(post_code, recent_prices)
print(f"Predicted Price for {post_code} in 2025: ${predicted_price:,.2f}")
