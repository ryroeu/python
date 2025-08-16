import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    "results_url": "https://eu-dreams.com/results/2025",
    "csv_filename": "eurodreams.csv",
    "seq_length": 50,  # Number of past draws to use for prediction
    "epochs": 50,
    "batch_size": 16,
}

# -------------------------
# Part 1: Data Acquisition (Web Scraping)
# -------------------------
def fetch_and_save_results(url, filename):
    """
    Scrapes EuroDreams results from a URL, formats them, and saves to a CSV file.
    """
    print(f"🌎 Scraping winning numbers from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an error if the page can't be downloaded

        soup = BeautifulSoup(response.content, 'html.parser')
        
        drawings_data = []

        # Find all the containers that hold a single draw's results
        result_sets = soup.find_all('div', class_='results-ball-set')

        if not result_sets:
            print("❌ ERROR: Could not find any results on the page.")
            print("The website's HTML structure may have changed.")
            return False

        for result_set in result_sets:
            # Within each result set, find the main balls and the dream number
            main_balls = [int(b.text) for b in result_set.find_all('div', class_='result-ball')]
            dream_number = int(result_set.find('div', class_='result-lucky-star').text)
            
            if len(main_balls) == 6 and dream_number is not None:
                full_drawing = main_balls + [dream_number]
                drawings_data.append(full_drawing)

        # Create a pandas DataFrame from the scraped data
        column_names = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'boule_6', 'n_dream']
        df = pd.DataFrame(drawings_data, columns=column_names)

        # Reverse the order to have the oldest data first for time-series analysis
        df = df.iloc[::-1]

        df.to_csv(filename, index=False)
        print(f"✅ Successfully saved {len(drawings_data)} drawings to '{filename}'")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Could not download the webpage. {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: An error occurred during scraping. {e}")
        return False

# -------------------------
# Part 2: Data Processing
# -------------------------
def preprocess_data(filename):
    """Loads and preprocesses the EuroDreams lottery data."""
    df = pd.read_csv(filename)
    
    try:
        main_ball_cols = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'boule_6']
        dream_num_col = ['n_dream']
        
        balls = df[main_ball_cols].values
        dream_numbers = df[dream_num_col].values
    except KeyError as e:
        print(f"❌ ERROR: A column name was not found in the CSV: {e}")
        return None, None, None, None, None

    # EuroDreams: 6 balls (1-40), 1 Dream Number (1-5).
    scaler_balls = MinMaxScaler(feature_range=(0, 1))
    scaler_dream = MinMaxScaler(feature_range=(0, 1))

    balls_scaled = scaler_balls.fit_transform(balls)
    dream_scaled = scaler_dream.fit_transform(dream_numbers)
    
    draws_scaled = np.concatenate((balls_scaled, dream_scaled), axis=1)

    X, y = [], []
    for i in range(len(draws_scaled) - CONFIG["seq_length"]):
        X.append(draws_scaled[i:i + CONFIG["seq_length"]])
        y.append(draws_scaled[i + CONFIG["seq_length"]])
        
    return np.array(X), np.array(y), draws_scaled, scaler_balls, scaler_dream

# -------------------------
# Part 3: Model Training
# -------------------------
def build_and_train_model(X_train, y_train):
    """Builds and trains the LSTM model."""
    print("\nBuilding LSTM model...")
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(100, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(X_train.shape[2]) # Output layer with 7 neurons (6 balls + 1 dream number)
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
def make