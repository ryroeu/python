import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

# --- Constants ---
CSV_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv"
SEQ_LENGTH = 50
EPOCHS = 50
BATCH_SIZE = 16

def download_and_prepare_data(url: str) -> pd.DataFrame:
    """
    Downloads EuroMillions draw history, removes unnecessary columns, and returns a DataFrame.

    Args:
        url: The URL to the CSV file.

    Returns:
        A pandas DataFrame with the draw history.
    """
    print("Downloading CSV file...")
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    # Use StringIO to read the content directly into pandas
    csv_content = StringIO(response.text)
    df = pd.read_csv(csv_content)
    
    print("Processing data...")
    # Drop columns at index positions 8, 9, and 10 (Excel columns I, J, and K)
    if df.shape[1] >= 11:
        df.drop(df.columns[[8, 9, 10]], axis=1, inplace=True)
        print("Deleted columns I, J, and K.")
    else:
        print("Columns I, J, and K not found. Skipping deletion.")
        
    print("\nSample Draw History:")
    print(df.head())
    return df

def extract_draws(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts ball and lucky star numbers from the DataFrame into a NumPy array.

    Args:
        df: The DataFrame containing the draw history.

    Returns:
        A NumPy array of the draw numbers.
    """
    draw_columns = ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Lucky Star 1', 'Lucky Star 2']
    
    # Convert draw columns to a NumPy array, handling potential errors
    try:
        draws = df[draw_columns].to_numpy(dtype=int)
    except (KeyError, ValueError) as e:
        print(f"Error extracting draw numbers: {e}")
        # Handle cases where columns are missing or contain non-numeric data
        # For this script, we will exit if the data is not as expected.
        exit(1)
        
    return draws

def create_sequences(data: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of data for training the LSTM model.

    Args:
        data: The scaled draw data.
        seq_length: The length of the input sequences.

    Returns:
        A tuple containing the input sequences (X) and target values (y).
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_and_train_model(X_train: np.ndarray, y_train: np.ndarray, num_features: int) -> tf.keras.Model:
    """
    Builds, compiles, and trains the LSTM model.

    Args:
        X_train: The training input sequences.
        y_train: The training target values.
        num_features: The number of features in the data (number of balls + lucky stars).

    Returns:
        The trained Keras model.
    """
    print("\nBuilding the model...")
    model = Sequential([
        Input(shape=(SEQ_LENGTH, num_features)),
        # Using 'tanh' activation, which is common for LSTMs to help regulate gradients.
        LSTM(50, activation='tanh'),
        Dropout(0.2),
        Dense(num_features)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nTraining the model...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    return model

def predict_next_draw(model: tf.keras.Model, data_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Predicts the next draw using the trained model.

    Args:
        model: The trained LSTM model.
        data_scaled: The full sequence of scaled draw data.
        scaler: The scaler used to normalize the data.

    Returns:
        The predicted next draw as a rounded NumPy array.
    """
    print("\nPredicting the next draw...")
    last_sequence = data_scaled[-SEQ_LENGTH:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    predicted_scaled = model.predict(last_sequence)
    predicted = scaler.inverse_transform(predicted_scaled)
    
    # Round the predicted numbers to the nearest whole number.
    return np.round(predicted).astype(int)

def main():
    """
    Main function to run the lottery prediction script.
    """
    # Part 1: Download and Process Data
    df = download_and_prepare_data(CSV_URL)
    draws = extract_draws(df)
    
    # Part 2: Deep Learning Model (Educational Only)
    # Normalize the data to the [0, 1] range.
    scaler = MinMaxScaler(feature_range=(0, 1))
    draws_scaled = scaler.fit_transform(draws)
    
    # Create sequences for training
    X, y = create_sequences(draws_scaled, SEQ_LENGTH)
    
    if len(X) == 0:
        print("Not enough data to create sequences. Exiting.")
        return

    # Build and train the model
    model = build_and_train_model(X, y, draws.shape[1])
    
    # Predict the next draw
    predicted_draw = predict_next_draw(model, draws_scaled, scaler)
    
    print("\n---")
    print("DISCLAIMER: This prediction is for educational purposes only.")
    print("Lottery draws are random events, and past results do not influence future outcomes.")
    print("---\n")
    print("Predicted next draw (balls and Lucky Stars):")
    print(predicted_draw[0])

if __name__ == "__main__":
    main()
