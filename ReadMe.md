# **Prediction and Report Generation**  
An autonomous agent that performs end-to-end equity research.  
It ingests real-time market data, performs advanced time-series forecasting, analyzes news sentiment, and generates a professional PDF report using Google Gemini.

---

##  **Key Features**

### **Advanced Forecasting**
- Uses **Meta's Prophet model** (instead of simple Linear Regression).
- Predicts **30-day future stock prices** with seasonality + trend modeling.

### **Quantitative Analysis**
- Computes key technical indicators:
  - **RSI (14)**
  - **SMA (50)**
  - **SMA (200)**

### **AI Analyst Agent**
- Uses **Google Gemini 2.5 Flash** to synthesize:
  - Market data  
  - Technical indicators  
  - News sentiment  
- Generates a **professional investment thesis**.

### **Multi-Market Support (New)**
- Automatically detects **Indian stocks** (`.NS`, `.BO`) and switches currency formatting to **₹`.
- Supports **US stocks** (`AAPL`, `TSLA`) with **$** formatting.

### **Professional Outputs**
- Generates a **Forecast Chart** (History + Prediction + Confidence Interval).
- Produces a **formatted PDF report** with analyst commentary & tables.

---

##  **Setup Instructions**

### **1. Prerequisites**
- Python **3.9+**
- **C++ Build Tools** (required for Prophet)
  - Windows: Install *"Desktop development with C++"* via Visual Studio Build Tools  
  - Mac/Linux: Install via `xcode-select` or `build-essential`
- A **Google Gemini API Key** (free from Google AI Studio)

---

### **2. Installation**

#### Create project:
```bash
mkdir ai_stock_copilot
cd ai_stock_copilot
```

#### Create a Virtual Environment:
```bash
python -m venv venv
```

##### Windows:
```bash
venv\Scripts\activate
```

##### Mac/Linux:
```bash
source venv/bin/activate
```

#### Install dependencies:
```bash
pip install -r requirements.txt
```

---

### **3. Configuration**

Create a file named **`.env`** in the project root.

Add your API key:
```env
GOOGLE_API_KEY=AIzaSy_Your_Key_Here
```

---

##  **Usage**

### Run the Agent:
```bash
python main.py
```

### Enter a ticker:
- **US Market:** `NVDA`, `TSLA`, `AAPL`  
- **Indian Market:** `RELIANCE.NS`, `TCS.NS`, `INFY.NS`

### Wait for Processing:
- Downloads **3 years of data**
- Runs **Prophet forecasting**
- Fetches **news sentiment**
- Generates the **PDF report**

---

## **Outputs**

Located in the `output/` folder:

- **`TICKER_forecast_chart.png`**  
  Historical price + 30-day prediction cone  
- **`TICKER_Report_FPDF.pdf`**  
  Final 1-page professional research report

---
