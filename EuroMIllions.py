import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    "csv_url": "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv",
    "csv_filename": "euromillions.csv",
    "seq_length": 50,  # Number of past draws to use for prediction
    "epochs": 50,
    "batch_size": 16,
}

# -------------------------
# Part 1: Data Acquisition
# -------------------------
def download_data(url, filename):
    """Downloads CSV data from a URL and removes specified columns."""
    print(f"Downloading data from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Saved data to {filename}")

        # Remove columns I, J, and K
        df = pd.read_csv(filename)
        if df.shape[1] >= 11:
            df.drop(df.columns[[8, 9, 10]], axis=1, inplace=True)
            df.to_csv(filename, index=False)
            print("Cleaned unnecessary columns.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return False

# -------------------------
# Part 2: Data Processing
# -------------------------
def preprocess_data(filename):
    """Loads and preprocesses the lottery data."""
    df = pd.read_csv(filename)
    
    # Extract ball and lucky star numbers
    balls = df[['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5']].values
    lucky_stars = df[['Lucky Star 1', 'Lucky Star 2']].values
    
    # **KEY IMPROVEMENT**: Scale balls and lucky stars separately
    # Main balls are 1-50, Lucky Stars are 1-12. Scaling them together is bad practice.
    scaler_balls = MinMaxScaler(feature_range=(0, 1))
    scaler_stars = MinMaxScaler(feature_range=(0, 1))

    balls_scaled = scaler_balls.fit_transform(balls)
    stars_scaled = scaler_stars.fit_transform(lucky_stars)
    
    # Combine scaled data back together
    draws_scaled = np.concatenate((balls_scaled, stars_scaled), axis=1)

    # Create sequences
    X, y = [], []
    for i in range(len(draws_scaled) - CONFIG["seq_length"]):
        X.append(draws_scaled[i:i + CONFIG["seq_length"]])
        y.append(draws_scaled[i + CONFIG["seq_length"]])
        
    return np.array(X), np.array(y), draws_scaled, scaler_balls, scaler_stars

# -------------------------
# Part 3: Model Training
# -------------------------
def build_and_train_model(X_train, y_train):
    """Builds and trains the LSTM model."""
    print("\nBuilding LSTM model...")
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(100, activation='relu', return_sequences=True), # Added a second LSTM layer
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(X_train.shape[2]) # Output layer with 7 neurons
    ])
    model.compile(optimizer='adam', loss='mse')
    print("Model Summary:")
    model.summary()

    print("\nTraining the model...")
    model.fit(
        X_train, y_train, 
        epochs=CONFIG["epochs"], 
        batch_size=CONFIG["batch_size"], 
        verbose=1
    )
    return model

# -------------------------
# Part 4: Prediction
# -------------------------
def make_prediction(model, data_scaled, scaler_balls, scaler_stars):
    """Makes a prediction for the next draw."""
    last_sequence = np.expand_dims(data_scaled[-CONFIG["seq_length"]:], axis=0)
    
    predicted_scaled = model.predict(last_sequence)
    
    # Inverse transform the predictions using the correct scalers
    predicted_balls = scaler_balls.inverse_transform(predicted_scaled[:, :5])
    predicted_stars = scaler_stars.inverse_transform(predicted_scaled[:, 5:])
    
    # Combine and round to the nearest integer
    predicted_draw = np.concatenate((predicted_balls, predicted_stars), axis=1)
    return np.round(predicted_draw).astype(int)[0]

# -------------------------
# Main Execution
# -------------------------
def main():
    """Main function to run the script."""
    if not download_data(CONFIG["csv_url"], CONFIG["csv_filename"]):
        return # Exit if download fails
        
    if not os.path.exists(CONFIG["csv_filename"]):
        print(f"CSV file not found: {CONFIG['csv_filename']}")
        return

    X, y, draws_scaled, scaler_balls, scaler_stars = preprocess_data(CONFIG["csv_filename"])
    
    if len(X) == 0:
        print("Not enough data to create sequences. Try a smaller 'seq_length'.")
        return
        
    model = build_and_train_model(X, y)
    
    predicted_numbers = make_prediction(model, draws_scaled, scaler_balls, scaler_stars)
    
    print("\n---" * 10)
    print("🔮 DISCLAIMER: This is for educational purposes only. Lottery numbers are random.")
    print("The predicted numbers have the same chance of winning as any other combination.")
    print("---\n" * 1)
    
    # Sort the numbers first
    predicted_main_balls = sorted(predicted_numbers[:5])
    predicted_lucky_stars = sorted(predicted_numbers[5:])
    
    # Convert the lists of numbers into comma-separated strings
    main_balls_str = ", ".join(map(str, predicted_main_balls))
    lucky_stars_str = ", ".join(map(str, predicted_lucky_stars))
    
    print(f"Predicted Main Balls: {main_balls_str}")
    print(f"Predicted Lucky Stars: {lucky_stars_str}")
    print("\n---" * 10)

if __name__ == "__main__":
    main()