# ITC Stock Trend Analysis Dashboard 📈

## Description

This project is a web-based dashboard that provides real-time analysis and trend prediction for the ITC stock (ITC.BSE) listed on the Bombay Stock Exchange. It utilizes an LSTM (Long Short-Term Memory) model trained on historical stock data to predict whether the next day's trend will be Bullish or Bearish, along with a confidence score. The dashboard displays live market data, technical indicators, model predictions, and basic pipeline status information.

The frontend is built with HTML, CSS, and Chart.js for interactive visualizations, while the backend uses Flask (Python) to serve the webpage, fetch live data (via a secure API call), and run the prediction model.

## Features ✨

* **Live Market Data:** Displays the latest closing price and daily change (▲/▼) for ITC stock.
* **Candlestick Chart:** Visualizes recent price action (Open, High, Low, Close).
* **Volume Chart:** Shows trading volume corresponding to the price chart.
* **LSTM Model Prediction:**
    * Predicts the next day's trend (Bullish/Bearish).
    * Displays key indicator values (RSI, MACD, SMA) for the latest data point.
    * Shows model Confidence and Accuracy via semi-circle gauges.
* **Technical Indicators:** Presents charts for:
    * Price Line with SMA and EMA overlays.
    * Relative Strength Index (RSI).
    * Bollinger Bands®.
    * Moving Average Convergence Divergence (MACD) with Signal Line.
* **Data Pipeline Status:** Provides information on API status, data update latency, model version, and next retraining date.
* **Responsive Design:** Adapts layout for different screen sizes (desktop, tablet, mobile).
* **Secure API Key Handling:** Fetches stock data via the backend, keeping the API key hidden from the frontend.

## Tech Stack 🛠️

* **Backend:** Python, Flask, TensorFlow (for Keras), Scikit-learn (for scaler), NumPy, Requests
* **Frontend:** HTML, CSS, JavaScript
* **Charting:** Chart.js with `chartjs-chart-financial` and `chartjs-adapter-luxon` plugins
* **Date/Time:** Luxon.js
* **Deployment:** Vercel (recommended)

## Project Structure 
        ITC-Trend-Analysis/
           ├── api/
           │   └── index.py
           ├── templates/
           │   └── index.html
           ├── itc_lstm_model.keras
           ├── scaler.pkl
           ├── requirements.txt
           ├── vercel.json
           └── .gitignore

## Setup & Installation (Local Development) ⚙️

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Dhinesh-Fedor/ITC-Trend-Analysis.git
    cd ITC-Trend-Dashboard
    ```
2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set Environment Variable:** You need an Alpha Vantage API key. Get a free one [here](https://www.alphavantage.co/support/#api-key). Set it in your terminal **before** running the server:
    ```bash
    # On Linux/macOS:
    export ALPHA_VANTAGE_KEY="YOUR_ACTUAL_API_KEY"
    # On Windows (Command Prompt):
    set ALPHA_VANTAGE_KEY=YOUR_ACTUAL_API_KEY
    # On Windows (PowerShell):
    $env:ALPHA_VANTAGE_KEY="YOUR_ACTUAL_API_KEY"
    ```

## Running Locally 🚀

1.  Make sure your virtual environment is activated and the `ALPHA_VANTAGE_KEY` is set.
2.  Navigate to the project's root directory (`ITC-Trend-Dashboard`).
3.  Run the Flask development server:
    ```bash
    flask --app api/index run --debug
    ```
    *(Alternatively, if you add `app.run(...)` back to `api/index.py`: `python api/index.py`)*
4.  Open your web browser and go to `http://localhost:5000/`.

## Deployment (Vercel) ☁️

This project is configured for easy deployment on Vercel:

1.  **Push to GitHub:** Ensure your project (with the correct structure, `requirements.txt`, and `vercel.json`) is pushed to a GitHub repository.
2.  **Import Project:** Log in to Vercel and import the project from your GitHub repository.
3.  **Configure Environment Variable:** In your Vercel project settings (**Settings -> Environment Variables**), add your `ALPHA_VANTAGE_KEY` with your actual API key value. Ensure it's available for the **Production** environment.
4.  **Deploy:** Trigger a deployment. Vercel will use `vercel.json` and `requirements.txt` to build and deploy your Flask app as a serverless function.
5.  **Visit:** Access your live dashboard via the URL provided by Vercel.

**Note on Deployment Size:** TensorFlow can be large. If you encounter deployment size limits on Vercel's free tier, consider using `tensorflow-cpu` in `requirements.txt` or exploring alternative hosting/model loading strategies.

## File Descriptions 📄

* **`api/index.py`:** Contains the Flask application logic. Handles routing, fetching data from Alpha Vantage (using the environment variable API key), running predictions with the loaded model and scaler, and serving the main HTML page.
* **`templates/index.html`:** The core frontend file. Contains the HTML structure, CSS styling (including responsive design), and JavaScript for initializing charts, fetching data from the Flask backend (`/stockdata`, `/predict`, `/status`), calculating indicators (SMA, EMA, RSI, Bollinger Bands, MACD), and updating the UI elements.
* **`itc_lstm_model.keras`:** The pre-trained TensorFlow Keras LSTM model file used for making trend predictions.
* **`scaler.pkl`:** The saved Scikit-learn MinMaxScaler object used to scale input data before feeding it to the LSTM model.
* **`requirements.txt`:** Lists all necessary Python packages for the backend.
* **`vercel.json`:** Configuration file instructing Vercel how to build the Python backend and route incoming web requests to the Flask application.
* **`.gitignore`:** Specifies files and directories that Git should ignore (e.g., virtual environment).

---
