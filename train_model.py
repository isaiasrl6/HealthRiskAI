import pandas as pd
import os
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# ========== CONFIGURACIÓN ==========
DATA_PATH = 'data/clean_insurance.csv'
MODEL_DIR = 'model'
LOG_DIR = 'logs'
AUC_THRESHOLD = 0.82 # Subimos la vara

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'train_model.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model_pro():
    logging.info("Iniciando Pipeline de Entrenamiento Pro")

    # 1. Carga de datos
    df = pd.read_csv(DATA_PATH)
    X = df.drop('high_cost', axis=1)
    y = df['high_cost']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # Mantener proporción de clases
    )

    # 2. Definición de Búsqueda de Hiperparámetros
    # Explicación: Buscamos el balance óptimo entre complejidad y precisión
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced'] # Crucial si hay pocos casos de "alto costo"
    }

    # 3. Validación Cruzada
    cv = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='roc_auc', 
        n_jobs=-1 # Usar todos los núcleos del procesador
    )

    logging.info("Ejecutando GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    # 4. Evaluación
    y_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    logging.info(f"Mejor AUC encontrado: {auc:.4f}")
    logging.info(f"Mejores Parámetros: {grid_search.best_params_}")

    if auc < AUC_THRESHOLD:
        raise ValueError(f"Rendimiento insuficiente ({auc:.4f}). Umbral: {AUC_THRESHOLD}")

    # 5. Persistencia Trazable
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name = f'healthrisk_model_{timestamp}.pkl'
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # Guardamos metadatos junto al modelo
    model_meta = {
        'model': best_model,
        'params': grid_search.best_params_,
        'auc': auc,
        'features': list(X.columns),
        'date': timestamp
    }
    
    joblib.dump(model_meta, model_path)
    
    # Visualización de Diagnóstico
    RocCurveDisplay.from_predictions(y_test, y_proba, name="RF Optimized")
    plt.title(f"ROC Curve - AUC: {auc:.4f}")
    plt.savefig(os.path.join(MODEL_DIR, f'roc_{timestamp}.png'))
    plt.close()

    print(f"Modelo Pro guardado: {model_name} con AUC {auc:.4f}")

if __name__ == "__main__":
    try:
        train_model_pro()
    except Exception as e:
        logging.error(f"Falla en entrenamiento: {e}")