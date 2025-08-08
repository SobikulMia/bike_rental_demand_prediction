from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Input features
        season = data['season']
        yr = data['yr']
        mnth = data['mnth']
        day = data['day']
        hr = data['hr']
        holiday = data['holiday']
        weekday = data['weekday']
        workingday = data['workingday']
        weathersit = data['weathersit']
        temp = data['temp']
        atemp = data['atemp']
        hum = data['hum']
        windspeed = data['windspeed']
        dayofweek = data['dayofweek']
        is_weekend = data['is_weekend']

        # Derived Features
        year = 2011 if yr == 0 else 2012
        month = mnth
        temp_hum = temp * hum
        wind_season = windspeed * season

        # Prepare Input Array (Total 19 features)
        input_data = [season, yr, mnth, hr, holiday, weekday, workingday, weathersit,
                      temp, atemp, hum, windspeed, year, month, day, dayofweek,
                      is_weekend, temp_hum, wind_season]

        input_scaled = scaler.transform([input_data])
        pred_log = model.predict(input_scaled)
        prediction = np.expm1(pred_log)[0]

        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)
