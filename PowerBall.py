import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Part 1: Download CSV File for PowerBall
# -------------------------
csv_url = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
csv_filename = "powerball.csv"

# Download the CSV file and overwrite any existing file
response = requests.get(csv_url)
with open(csv_filename, "wb") as file:
    file.write(response.content)
print(f"Downloaded CSV file and saved as {csv_filename}")

# -------------------------
# Part 1.5: Sort CSV by Draw Date (most recent first)
# -------------------------
df = pd.read_csv(csv_filename)
# Convert "Draw Date" column to datetime and sort descending
df['Draw Date'] = pd.to_datetime(df['Draw Date'])
df = df.sort_values(by='Draw Date', ascending=False)
# Overwrite the CSV with sorted data
df.to_csv(csv_filename, index=False)
print("Sorted CSV file by Draw Date (most recent first).")

print("\nSample Draw History:")
print(df.head())

# -------------------------
# Part 2: Data Processing
# -------------------------
# The CSV has three columns: Draw Date, Winning Numbers, Multiplier.
# We use only the 'Winning Numbers' column.
draws = []
for index, row in df.iterrows():
    try:
        # "Winning Numbers" is assumed to be a space-separated string, e.g., "3 15 27 37 48"
        winning_numbers_str = row['Winning Numbers']
        numbers = [int(num) for num in winning_numbers_str.strip().split()]
        draws.append(numbers)
    except Exception as e:
        print(f"Error processing row {index}: {e}")

draws = np.array(draws)

# -------------------------
# Part 3: Deep Learning Model (Educational Only)
# -------------------------
# Normalize the data to the [0, 1] range.
scaler = MinMaxScaler(feature_range=(0, 1))
draws_scaled = scaler.fit_transform(draws)

# Create sequences: use the previous 'seq_length' draws to predict the next one.
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 100  # Number of past draws used as input
X, y = create_sequences(draws_scaled, seq_length)

# Build a simple LSTM model using an explicit Input layer.
model = Sequential([
    Input(shape=(seq_length, draws.shape[1])),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(draws.shape[1])
])
model.compile(optimizer='adam', loss='mse')

print("\nModel Summary:")
model.summary()

# Train the model (adjust epochs and batch_size as needed).
print("\nTraining the model...")
model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# Predict the next draw using the most recent sequence.
last_sequence = draws_scaled[-seq_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)
predicted_scaled = model.predict(last_sequence)
predicted = scaler.inverse_transform(predicted_scaled)

# Round the predicted numbers to the nearest whole number.
predicted_rounded = np.round(predicted).astype(int)

print("\nPredicted next draw (winning numbers):")
print(predicted_rounded[0])