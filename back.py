from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.impute import SimpleImputer
import logging
import os
import tempfile
from werkzeug.utils import secure_filename
from collections import Counter
import joblib
import time

# === Importaciones opcionales ===
XGBOOST_AVAILABLE = False
SMOTE_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    pass

# === Configuración ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ORIGINAL_REQUIRED_COLUMNS = [
    'koi_period', 'koi_duration', 'koi_prad', 'koi_depth',
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_teq', 'koi_model_snr'
]
TARGET_COLUMN = 'koi_disposition'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Feature engineering
DERIVED_FEATURES = ['snr_per_period', 'temp_norm', 'prad_srad_ratio']

app = Flask(__name__)
CORS(app)

# Variables globales
model = None
metrics = None
train_data = None
imputer = None


# === Funciones auxiliares ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_target(y_series):
    """Convierte a binario: CONFIRMED = 1, otros = 0"""
    y_clean = y_series.str.upper().str.strip()
    valid_values = ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]
    if not y_clean.isin(valid_values).all():
        invalid = y_clean[~y_clean.isin(valid_values)].unique()
        raise ValueError(f"Valores inválidos en '{TARGET_COLUMN}': {invalid.tolist()}")
    return y_clean.map({"CONFIRMED": 1, "FALSE POSITIVE": 0, "CANDIDATE": 0})


def add_derived_features(df):
    """Agrega características derivadas de manera segura"""
    df = df.copy()
    df['snr_per_period'] = df['koi_model_snr'] / (df['koi_period'].clip(lower=1e-8))
    df['temp_norm'] = (df['koi_steff'] - 5000) / 1000.0
    df['prad_srad_ratio'] = df['koi_prad'] / (df['koi_srad'].clip(lower=1e-8))
    return df


def compute_scale_pos_weight(y):
    counts = Counter(y)
    neg = counts.get(0, 1)
    pos = counts.get(1, 1)
    return max(neg / pos, 1e-3)


# === Entrenamiento RÁPIDO ===
def train_random_forest_fast(X_train, y_train):
    logger.info("Entrenando Random Forest (Fast)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    params_used = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced"
    }
    return model, params_used


def train_xgboost_fast(X_train, y_train):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost no está instalado.")
    logger.info("Entrenando XGBoost (Fast)...")
    scale_pos_weight = compute_scale_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    model.fit(X_train, y_train)
    params_used = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": round(scale_pos_weight, 2)
    }
    return model, params_used


# === Endpoints ===
@app.route('/train', methods=['POST'])
def train_model():
    global model, metrics, train_data, imputer
    start_time = time.time()
    model = metrics = train_data = imputer = None

    if 'files' not in request.files:
        return jsonify({"error": "Falta el campo 'files'"}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename.strip() == '' for f in files):
        return jsonify({"error": "No se subieron archivos válidos"}), 400

    dfs = []
    temp_dir = tempfile.mkdtemp()

    try:
        logger.info("Cargando archivos...")
        for file in files:
            if not file or not allowed_file(file.filename):
                return jsonify({"error": f"Archivo no permitido: {file.filename}"}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(temp_dir, filename)
            file.save(filepath)

            try:
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)
                else:
                    df = pd.read_excel(filepath, engine='openpyxl')
                dfs.append(df)
            except Exception as e:
                return jsonify({"error": f"Error al procesar {filename}: {str(e)[:300]}"}), 400
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        if not dfs:
            return jsonify({"error": "No se procesaron archivos válidos"}), 400

        data = pd.concat(dfs, ignore_index=True)
        train_data = data.copy()

        # Validar columnas
        missing = [col for col in ORIGINAL_REQUIRED_COLUMNS + [TARGET_COLUMN] if col not in data.columns]
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")

        logger.info("Preparando datos...")
        y = prepare_target(data[TARGET_COLUMN])
        X = data[ORIGINAL_REQUIRED_COLUMNS].copy()
        for col in ORIGINAL_REQUIRED_COLUMNS:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        imputer = SimpleImputer(strategy='median')
        X_imputed_array = imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed_array, columns=ORIGINAL_REQUIRED_COLUMNS)
        X_imputed = add_derived_features(X_imputed)

        before = len(X_imputed)
        mask = ~X_imputed.duplicated()
        X_clean = X_imputed[mask].reset_index(drop=True)
        y_clean = y[mask].reset_index(drop=True)
        after = len(X_clean)

        if after == 0:
            return jsonify({"error": "No hay datos válidos tras limpieza"}), 400

        class_balance = y_clean.value_counts(normalize=True).to_dict()
        class_balance = {("CONFIRMED" if k == 1 else "NOT_CONFIRMED"): round(v, 4) for k, v in class_balance.items()}

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )

        smote_applied = False
        if SMOTE_AVAILABLE and len(set(y_train)) > 1:
            try:
                logger.info("Aplicando SMOTE...")
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                smote_applied = True
            except Exception as e:
                logger.warning(f"No se pudo aplicar SMOTE: {e}")

        use_xgboost = request.form.get('use_xgboost', 'false').lower() == 'true'
        if use_xgboost:
            model, best_params = train_xgboost_fast(X_train, y_train)
            model_type = "XGBoost (Fast)"
        else:
            model, best_params = train_random_forest_fast(X_train, y_train)
            model_type = "RandomForest (Fast)"

        logger.info("Evaluando modelo...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        training_time = round(time.time() - start_time, 2)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision_confirmed": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall_confirmed": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_confirmed": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "auc": round(roc_auc_score(y_test, y_proba), 4),
            "rows_before": before,
            "rows_after": after,
            "duplicated_removed": before - after,
            "class_balance": class_balance,
            "best_params": best_params,
            "model_type": model_type,
            "smote_applied": smote_applied,
            "training_time_seconds": training_time,
            "report": classification_report(
                y_test, y_pred,
                target_names=["NOT_CONFIRMED", "CONFIRMED"],
                output_dict=True,
                zero_division=0
            )
        }

        return jsonify({"message": "Modelo entrenado correctamente", **metrics})

    except Exception as e:
        logger.exception("Error en /train")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


@app.route('/predict', methods=['POST'])
def predict():
    global model, imputer
    if model is None or imputer is None:
        return jsonify({"error": "Modelo no entrenado. Llama a /train primero"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON vacío"}), 400

    for col in ORIGINAL_REQUIRED_COLUMNS:
        if col not in data:
            return jsonify({"error": f"Falta la columna: {col}"}), 400

    try:
        input_df = pd.DataFrame([data])[ORIGINAL_REQUIRED_COLUMNS]
        for col in ORIGINAL_REQUIRED_COLUMNS:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        X_imputed_array = imputer.transform(input_df)
        X_imputed_df = pd.DataFrame(X_imputed_array, columns=ORIGINAL_REQUIRED_COLUMNS)
        X_final = add_derived_features(X_imputed_df)

        proba = model.predict_proba(X_final)[0]

        threshold = getattr(model, "best_threshold_", 0.5)
        pred = 1 if proba[1] >= threshold else 0

        prediction = "CONFIRMED" if pred == 1 else "NOT_CONFIRMED"
        confidence = round(proba[pred] * 100, 2)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "threshold_used": threshold,
            "probabilities": {
                "NOT_CONFIRMED": round(proba[0] * 100, 2),
                "CONFIRMED": round(proba[1] * 100, 2)
            }
        })
    except Exception as e:
        logger.exception("Error en /predict")
        return jsonify({"error": f"Error en predicción: {str(e)}"}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    if metrics is None:
        return jsonify({"error": "Entrena el modelo primero con train/"}), 400
    return jsonify(metrics)


@app.route('/save_model', methods=['POST'])
def save_model():
    if model is None or imputer is None:
        return jsonify({"error": "Modelo no entrenado. No hay nada que guardar"}), 400
    try:
        joblib.dump(model, "exoplanet_model.pkl")
        joblib.dump(imputer, "imputer.pkl")
        return jsonify({
            "message": "Modelo guardado correctamente",
            "files": ["exoplanet_model.pkl", "imputer.pkl"]
        })
    except Exception as e:
        logger.exception("Error al guardar modelo")
        return jsonify({"error": f"Error al guardar: {str(e)}"}), 500


@app.route('/load_model', methods=['POST'])
def load_model():
    global model, imputer
    try:
        if not os.path.exists("exoplanet_model.pkl") or not os.path.exists("imputer.pkl"):
            return jsonify({"error": "Archivos del modelo no encontrados"}), 404
        model = joblib.load("exoplanet_model.pkl")
        imputer = joblib.load("imputer.pkl")
        return jsonify({"message": ""})
    except Exception as e:
        logger.exception("Error al cargar modelo")
        return jsonify({"error": f"Error al cargar: {str(e)}"}), 500


@app.route('/clear', methods=['POST'])
def clear_cache():
    global model, metrics, train_data, imputer
    model = metrics = train_data = imputer = None
    return jsonify({"message": "Caché limpiado correctamen"})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "model_trained": model is not None,
        "xgboost_available": XGBOOST_AVAILABLE,
        "smote_available": SMOTE_AVAILABLE,
        "mode": "FAST (No GridSearch)"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
