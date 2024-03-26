from flask import Flask, request, jsonify
from flask_expects_json import expects_json
import joblib
import pickle

app = Flask(__name__)

# Input data schema for temperature prediction
temperature_schema = {
    'type': 'object',
    'properties': {
        'latitude': {'type': 'number'},
        'longitude': {'type': 'number'},
    },
    'required': ['latitude', 'longitude']
}

# Input data schema for earthquake prediction
earthquake_schema = {
    'type': 'object',
    'properties': {
        'location': {'type': 'string'},
    },
    'required': ['location']
}

@app.route("/predict_temperature/", methods=["POST"])
@expects_json(temperature_schema)
def predict_temperature():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']

    model = joblib.load('linear_regression_model.pkl')
    predicted_temperature = model.predict([[latitude, longitude]])[0]
    formatted_temperature = "{:.2f}".format(predicted_temperature)

    return jsonify(latitude=latitude, longitude=longitude, predicted_temperature=formatted_temperature)

@app.route("/predict_earthquake/", methods=["POST"])
@expects_json(earthquake_schema)
def predict_earthquake():
    data = request.get_json()
    location = data['location']

    with open('earthquake_prediction_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    with open('earthquake_label_encoder.pkl', 'rb') as encoder_file:
        loaded_label_encoder = pickle.load(encoder_file)

    encoded_location = loaded_label_encoder.transform([location])
    predicted_magnitude = loaded_model.predict(encoded_location.reshape(-1, 1))
    formatted_magnitude = "{:.2f}".format(predicted_magnitude[0])

    return jsonify(predicted_magnitude=formatted_magnitude)
