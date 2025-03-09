import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Part 1: Download CSV File
# -------------------------
csv_url = "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv"
csv_filename = "euromillions.csv"

# Download the CSV file and overwrite any existing file
response = requests.get(csv_url)
with open(csv_filename, "wb") as file:
    file.write(response.content)
print(f"Downloaded CSV file and saved as {csv_filename}")

# -------------------------
# Part 1.5: Remove Columns I, J, and K
# -------------------------
# Load the CSV file into a DataFrame
df = pd.read_csv(csv_filename)

# Check if the file has at least 11 columns (corresponding to A through K)
if df.shape[1] >= 11:
    # Drop columns at index positions 8, 9, and 10 (Excel columns I, J, and K)
    df.drop(df.columns[[8, 9, 10]], axis=1, inplace=True)
    # Overwrite the CSV file with the updated DataFrame
    df.to_csv(csv_filename, index=False)
    print("Deleted columns I, J, and K from the CSV file.")
else:
    print("CSV file does not have columns I, J, and K. Skipping deletion.")

# Reload the CSV file after modification
df = pd.read_csv(csv_filename)
print("\nSample Draw History:")
print(df.head())

# -------------------------
# Part 2: Data Processing
# -------------------------
# Assuming the CSV has the following columns:
# 'Draw Date', 'Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Lucky Star 1', 'Lucky Star 2'
draws = []
for index, row in df.iterrows():
    try:
        balls = [int(row['Ball 1']), int(row['Ball 2']), int(row['Ball 3']),
                 int(row['Ball 4']), int(row['Ball 5'])]
        lucky_stars = [int(row['Lucky Star 1']), int(row['Lucky Star 2'])]
        draws.append(balls + lucky_stars)
    except Exception as e:
        print(f"Error processing row {index}: {e}")

draws = np.array(draws)

# -------------------------
# Part 3: Deep Learning Model (Educational Only)
# -------------------------
# Normalize the data. The lottery numbers are scaled to the [0, 1] range.
scaler = MinMaxScaler(feature_range=(0, 1))
draws_scaled = scaler.fit_transform(draws)

# Create sequences: use the previous 'seq_length' draws to predict the next one.
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 50  # Number of past draws used as input
X, y = create_sequences(draws_scaled, seq_length)

# Build a simple LSTM model using an explicit Input layer to avoid warnings.
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

print("\nPredicted next draw (balls and Lucky Stars):")
print(predicted_rounded[0])