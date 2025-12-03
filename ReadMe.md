AI Stock Research Copilot (Week 1 MVP)

This tool generates a PDF equity research report containing:

Quantitative Analysis (RSI, SMA).

AI Prediction (30-day Price Forecast via Linear Regression).

Market News (Real-time search).

Charts (Price vs SMA visualization).

Setup Instructions

1. Prerequisites

Python 3.9+ installed.

A Google Gemini API Key (Get it from Google AI Studio).

2. Installation

Create the project folder and navigate inside:

mkdir ai_stock_copilot
cd ai_stock_copilot


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


3. Configuration

Open the .env file.

Paste your API key:

GOOGLE_API_KEY=AIzaSy...


4. Running the Agent

Run the main script:

python main.py


5. Outputs

Enter a ticker (e.g., AAPL).

Wait ~10 seconds.

Check the output/ folder for:

AAPL_chart.png (Technical Chart)

AAPL_Report.pdf (Final Report)