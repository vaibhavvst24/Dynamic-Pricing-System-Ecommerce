from flask import Flask, request, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load model
model = load('Dynamic_Sales_Model.pkl')

# Features used by the model
features = [
    'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment', 'Month', 'Year', 'WeekOfYear', 'DayOfWeek'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        input_data = {
            'Store': int(request.form['Store']),
            'Holiday_Flag': int(request.form['Holiday_Flag']),
            'Temperature': float(request.form['Temperature']),
            'Fuel_Price': float(request.form['Fuel_Price']),
            'CPI': float(request.form['CPI']),
            'Unemployment': float(request.form['Unemployment']),
            'Month': int(request.form['Month']),
            'Year': int(request.form['Year']),
            'WeekOfYear': int(request.form['WeekOfYear']),
            'DayOfWeek': int(request.form['DayOfWeek']),
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(input_df)[0]
        prediction_lpa = prediction / 1e5

        return render_template('index.html', result=f"Predicted Sales: ₹{prediction_lpa:.2f} Lakhs/Year")
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
