# Project Setup Instructions

## Step 1: Clone the Git Repository

```
git clone https://github.com/Nikku831/predict_report
cd predict_report
```

## Step 2: Create a Virtual Environment

```
python -m venv myenv
.\myenv\Scripts\activate
```

## Step 3: Install Dependencies
```
pip install -r requirements.txt
```

## Step 4: Configure Environment Variables
Set your Google API Key in .env file.
```
GOOGLE_API_KEY=your_actual_key_here
```
## Step 5: Run the Application 
```
python main.py
```