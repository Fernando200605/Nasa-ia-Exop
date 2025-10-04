from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================
# VARIABLES GLOBALES
# ==========================
model = None
accuracy = None
report = None


# ==========================
# ENTRENAR EL MODELO
# ==========================
@app.route('/train', methods=['POST'])
@cross_origin()
def train_model():
    global model, accuracy, report
    try:
        file = request.files['file']
        data = pd.read_csv(file)

        # Definir columnas (ajústalas a tu dataset real)
        features = ['koi_period', 'koi_duration', 'koi_prad', 'koi_depth']
        target = 'koi_disposition'

        data = data.dropna(subset=features + [target])

        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return jsonify({
            "message": "Modelo entrenado correctamente",
            "accuracy": accuracy,
            "report": report
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# PREDICCIONES
# ==========================
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    global model
    try:
        if model is None:
            return jsonify({"error": "El modelo no está entrenado"}), 400

        data = request.get_json()
        features = [[
            data['koi_period'],
            data['koi_duration'],
            data['koi_prad'],
            data['koi_depth']
        ]]

        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

