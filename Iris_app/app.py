# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # utile si Streamlit est sur un autre port

# Chargement du modèle et du scaler
with open('model/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/iris_species.txt', 'r') as f:
    species_list = f.read().splitlines()

@app.route('/')
def home():
    return jsonify({"message": "API Iris Classification - utilisez /predict (POST)"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Les 4 features attendues
        features = [
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]
        
        # Conversion en array 2D et normalisation
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Prédiction
        pred_idx = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        
        predicted_species = species_list[pred_idx]
        probabilities = {species_list[i]: float(proba[i]) for i in range(len(species_list))}
        
        return jsonify({
            "predicted_species": predicted_species,
            "probabilities": probabilities,
            "features_used": {
                "sepal_length": features[0],
                "sepal_width": features[1],
                "petal_length": features[2],
                "petal_width": features[3]
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)