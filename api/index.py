# /api/index.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
import traceback
import os
import requests
from datetime import datetime
import sys

# Point Flask to the correct template and static folders relative to this file's location
app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# --- Load API Key Safely ---
# Load ONLY from environment variable.
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY')

# --- ADDED: Check if the key is missing ---
if not ALPHA_VANTAGE_KEY:
    print("FATAL ERROR: Alpha Vantage API key not set.")
    print("Please set the ALPHA_VANTAGE_KEY environment variable in your deployment environment (e.g., Vercel).")
    # Set key to None to indicate it's unusable
    ALPHA_VANTAGE_KEY = None
else:
    print("Alpha Vantage API Key loaded from environment variable.")


# --- Load Model and Scaler ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "itc_lstm_model.keras")
scaler_path = os.path.join(base_dir, "scaler.pkl")

print(f"Base directory calculated as: {base_dir}")
print(f"Attempting to load model from: {model_path}")
print(f"Attempting to load scaler from: {scaler_path}")

model = None
scaler = None
try:
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found at {model_path}")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    else:
        print(f"Scaler file not found at {scaler_path}")

except Exception as e:
    print(f"FATAL ERROR loading model/scaler: {e}")
    traceback.print_exc()

# --- Routes ---

@app.route("/")
def home():
    if model is None or scaler is None:
         return "Error: Model or scaler failed to load on the server. Check deployment logs.", 500
    print("Serving index.html")
    return render_template("index.html")

# --- Stock Data Route ---
@app.route("/stockdata/<symbol>")
def get_stock_data(symbol):
    # --- MODIFIED: Check if key is None (meaning not set) ---
    if not ALPHA_VANTAGE_KEY:
        print("Error: API Key is missing.")
        return jsonify({"error": "Alpha Vantage API Key is not configured correctly on the server."}), 500
    # --- END MODIFICATION ---

    print(f"Fetching AV data for {symbol}")
    AV_URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={ALPHA_VANTAGE_KEY}'
    try:
        response = requests.get(AV_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "Error Message" in data or "Information" in data or "Time Series (Daily)" not in data:
            # Check specifically for invalid API key message from Alpha Vantage
            if "Invalid API call" in data.get("Error Message", "") or "Invalid API key" in data.get("Information", ""):
                 print("AV API Error: Invalid API Key reported by Alpha Vantage.")
                 return jsonify({"error": "Invalid Alpha Vantage API Key configured on the server."}), 500
            else:
                 print("AV API Error/Limit:", data)
                 return jsonify({"error": data.get("Information", data.get("Error Message", "Unknown API issue"))}), 503

        ts = data["Time Series (Daily)"]
        times = sorted(ts.keys())[-60:] # Get last 60 for indicators
        if len(times) < 35: return jsonify({"error": f"Not enough data ({len(times)})"}), 400

        processed_data = []
        for t in times:
            day_data = ts[t]
            try: processed_data.append({"t": t,"o": float(day_data["1. open"]),"h": float(day_data["2. high"]),"l": float(day_data["3. low"]),"c": float(day_data["4. close"]),"v": float(day_data["5. volume"])})
            except: processed_data.append({"t": t, "o": None, "h": None, "l": None, "c": None, "v": None})

        print(f"AV Fetch OK: {len(processed_data)} points.")
        return jsonify(processed_data)
    except requests.exceptions.Timeout:
        print("AV Request Timeout")
        return jsonify({"error": "Stock API request timed out"}), 504
    except requests.exceptions.RequestException as e:
        print(f"AV Network error: {e}")
        return jsonify({"error": "Network error connecting to stock API"}), 504
    except Exception as e:
        print(f"Error processing stock data: {e}"); traceback.print_exc()
        return jsonify({"error": "Internal error processing stock data"}), 500


# --- Prediction and Status Routes (Keep your existing robust versions) ---
def preprocess_prices(prices, timesteps=30):
    prices_numeric = [p for p in prices if p is not None and not np.isnan(p)]
    if not prices_numeric: return np.zeros((1, timesteps, 1))
    X = np.array(prices_numeric).reshape(-1, 1)
    if X.shape[0] == 0: return np.zeros((1, timesteps, 1))
    if scaler is None: raise ValueError("Scaler not loaded")
    try: X_scaled = scaler.transform(X)
    except Exception as e: print(f"Scaler error: {e}"); return np.zeros((1, timesteps, 1))
    current_len = len(X_scaled);
    if current_len < timesteps: pad = np.zeros((timesteps - current_len, X_scaled.shape[1])); X_scaled = np.vstack([pad, X_scaled])
    elif current_len > timesteps: X_scaled = X_scaled[-timesteps:]
    if X_scaled.shape != (timesteps, 1): print(f"Shape error: {X_scaled.shape}"); return np.zeros((1, timesteps, 1))
    return X_scaled.reshape(1, timesteps, 1)

def calculate_accuracy(prices, model_pred_label):
    prices_clean = [p for p in prices if p is not None and not np.isnan(p)];
    if len(prices_clean) < 2: return 0.0
    actual_label = "Bullish" if prices_clean[-1] > prices_clean[-2] else "Bearish";
    return 100.0 if model_pred_label == actual_label else 0.0

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None: return jsonify({"error": "Model/Scaler not loaded"}), 500
    try:
        data = request.get_json(); prices = data.get("prices", [])
        original_prices = prices[:]
        prices_clean = [p for p in prices if p is not None and not np.isnan(p)]
        if not prices_clean or len(prices_clean) < 2: return jsonify({"error": "Not enough valid data points"}), 400
        print(f"Predict: using {len(prices_clean)} clean prices.")
        X = preprocess_prices(prices_clean)
        if np.all(X == 0) and len(prices_clean) > 0: return jsonify({"error": "Data preprocessing failed"}), 500
        prob = float(model.predict(X, verbose=0)[0][0]); label = "Bullish" if prob >= 0.5 else "Bearish"
        confidence = round(prob*100 if label=="Bullish" else (1-prob)*100, 1)
        accuracy = calculate_accuracy(original_prices, label)
        response = { "label": label, "confidence": confidence, "accuracy": accuracy }; print("Prediction response:", response)
        return jsonify(response)
    except Exception as e: print(f"Critical prediction error: {e}"); traceback.print_exc(); return jsonify({"error": f"Prediction error: {e}"}), 500

@app.route("/status")
def get_status():
    try: status_data = { "model_version": "V1.0", "next_retraining": "2026-01-01" }; return jsonify(status_data)
    except Exception as e: print("Status error:", e); traceback.print_exc(); return jsonify({"error": str(e)}), 500
    
    
# Vercel runs the app, so this is not needed for deployment
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
