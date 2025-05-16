import pandas as pd
import os
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
import matplotlib.pyplot as plt

# ========== CONFIGURACIÓN ==========
DATA_PATH = 'data/clean_insurance.csv'
MODEL_DIR = 'model'
LOG_DIR = 'logs'
REPORT_PATH = os.path.join(LOG_DIR, 'training_report.csv')
AUC_THRESHOLD = 0.80  # Nivel mínimo aceptable

# Crear carpetas si no existen
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configurar logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'train_model.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========== ENTRENAMIENTO ==========
def train_model():
    logging.info("🚀 Iniciando entrenamiento del modelo")

    df = pd.read_csv(DATA_PATH)
    X = df.drop('high_cost', axis=1)
    y = df['high_cost']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    logging.info(f"AUC del modelo: {auc:.4f}")

    if auc < AUC_THRESHOLD:
        logging.warning(f"AUC menor al umbral aceptado ({AUC_THRESHOLD})")
        raise ValueError("❌ Modelo no cumple con el rendimiento mínimo esperado.")

    # Guardar el modelo con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_path = os.path.join(MODEL_DIR, f'healthrisk_model_{timestamp}.pkl')
    joblib.dump(model, model_path)
    logging.info(f"✅ Modelo guardado en {model_path}")

    # Guardar curva ROC
    roc_plot_path = os.path.join(MODEL_DIR, f'roc_curve_{timestamp}.png')
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.savefig(roc_plot_path)
    logging.info(f"📈 Curva ROC guardada en {roc_plot_path}")
    plt.close()

    # Guardar reporte en CSV
    metrics = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(metrics).transpose()
    report_df['auc'] = auc
    report_df['timestamp'] = timestamp
    report_df.to_csv(REPORT_PATH, index=True)
    logging.info(f"📋 Reporte de entrenamiento guardado en {REPORT_PATH}")

    # Consola
    print("✅ Entrenamiento finalizado con éxito")
    print(f"Modelo guardado en: {model_path}")
    print(f"AUC: {auc:.4f}")

# ========== EJECUCIÓN ==========
if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logging.error(f"❌ Error durante el entrenamiento: {e}")
        print(f"🚨 Error: {e}")
