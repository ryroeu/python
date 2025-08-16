import subprocess
import sys
import importlib.util

def check_and_install_dependencies():
    """
    Checks if required modules are installed and installs them if they are not.
    """
    # A dictionary mapping the import name to the package name for pip
    packages = {
        'kaggle': 'kaggle',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'sklearn': 'scikit-learn'  # Note: the import is 'sklearn', but the package is 'scikit-learn'
    }

    print("Checking for required Python modules...")
    all_installed = True
    for import_name, package_name in packages.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            all_installed = False
            print(f"⚠️ Module '{import_name}' not found. Attempting to install '{package_name}'...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"✅ Successfully installed '{package_name}'.")
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Failed to install '{package_name}'. Please install it manually using 'pip install {package_name}'")
                sys.exit(1) # Exit the script if a crucial dependency fails to install

    if all_installed:
        print("👍 All required modules are installed.")
    print("-" * 30)


# Run the dependency check at the very beginning
check_and_install_dependencies()


# Now, import the libraries for the rest of the script
import kaggle
import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    "kaggle_dataset": "eduardosilva46/eurodreams",
    "kaggle_filename": "Eurodreams From 2023-11-06.csv",
    "csv_filename": "eurodreams.csv", # The name we want the file to have locally
    "seq_length": 50,
    "epochs": 50,
    "batch_size": 16,
}

# -------------------------
# Part 1: Data Acquisition (Kaggle API)
# -------------------------
def download_from_kaggle(dataset, kaggle_file, local_file):
    """
    Downloads a specific file from a Kaggle dataset and renames it.
    (Handles URL-encoded spaces in filenames)
    """
    print(f"📥 Downloading '{kaggle_file}' from Kaggle dataset '{dataset}'...")
    try:
        # Download the file to the current directory
        kaggle.api.dataset_download_file(dataset, file_name=kaggle_file, path='.', force=True)
        print("✅ Download complete.")

        # Create the filename variant with URL-encoded spaces (%20)
        encoded_filename = kaggle_file.replace(' ', '%20')

        # Check if the downloaded file exists with the encoded name
        if os.path.exists(encoded_filename):
            print(f"Found downloaded file: '{encoded_filename}'")
            # Rename the file to our desired local filename
            shutil.move(encoded_filename, local_file)
            print(f"✏️ Renamed '{encoded_filename}' to '{local_file}'.")
            return True
        # As a fallback, check for the original name (in case the API behavior changes)
        elif os.path.exists(kaggle_file):
            print(f"Found downloaded file: '{kaggle_file}'")
            shutil.move(kaggle_file, local_file)
            print(f"✏️ Renamed '{kaggle_file}' to '{local_file}'.")
            return True
        else:
            print(f"❌ ERROR: Expected file was not found after download.")
            print(f"   Checked for '{kaggle_file}' and '{encoded_filename}'.")
            return False

    except Exception as e:
        print(f"❌ ERROR: Failed to download from Kaggle. Please check your setup.")
        print(f"   Ensure 'kaggle.json' is in the correct folder (~/.kaggle/ or C:\\Users\\<user>\\.kaggle)")
        print(f"   Kaggle API error: {e}")
        return False

# -------------------------
# Part 2: Data Processing
# -------------------------
def preprocess_data(filename):
    """Loads and preprocesses the EuroDreams lottery data from the CSV file."""
    df = pd.read_csv(filename)
    
    main_ball_cols = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Number 6']
    dream_num_col = ['Dream Number']
    
    balls = df[main_ball_cols].values
    dream_numbers = df[dream_num_col].values

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
        Dense(X_train.shape[2])
    ])
    model.compile(optimizer='adam', loss='mse')
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
def make_prediction(model, data_scaled, scaler_balls, scaler_dream):
    """Makes a prediction for the next draw."""
    last_sequence = np.expand_dims(data_scaled[-CONFIG["seq_length"]:], axis=0)
    
    predicted_scaled = model.predict(last_sequence)
    
    predicted_balls = scaler_balls.inverse_transform(predicted_scaled[:, :6])
    predicted_dream = scaler_dream.inverse_transform(predicted_scaled[:, 6:])
    
    predicted_draw = np.concatenate((predicted_balls, predicted_dream), axis=1)
    return np.round(predicted_draw).astype(int)[0]

# -------------------------
# Main Execution
# -------------------------
def main():
    """Main function to run the script."""
    if not download_from_kaggle(CONFIG["kaggle_dataset"], CONFIG["kaggle_filename"], CONFIG["csv_filename"]):
        return # Exit if data acquisition fails
        
    processed_data = preprocess_data(CONFIG["csv_filename"])
    
    X, y, draws_scaled, scaler_balls, scaler_dream = processed_data
    
    if len(X) == 0:
        print("Not enough data to create sequences. Try a smaller 'seq_length'.")
        return
        
    model = build_and_train_model(X, y)
    
    predicted_numbers = make_prediction(model, draws_scaled, scaler_balls, scaler_dream)
    
    print("\n" + "---" * 10)
    print("🔮 DISCLAIMER: This is for educational purposes only. Lottery numbers are random.")
    print("The predicted numbers have the same chance of winning as any other combination.")
    print("---" * 10 + "\n")
    
    predicted_main_balls = sorted(predicted_numbers[:6])
    predicted_dream_num = predicted_numbers[6]
    
    main_balls_str = ", ".join(map(str, predicted_main_balls))
    
    print(f"Predicted Main Balls: {main_balls_str}")
    print(f"Predicted Dream Number: {predicted_dream_num}")
    print("\n" + "---" * 10)

if __name__ == "__main__":
    main()