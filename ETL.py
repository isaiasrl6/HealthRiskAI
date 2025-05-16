import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import logging
from datetime import datetime

# ========== Configuración de rutas y logging ==========
DATA_PATH = 'insurance.csv'
OUTPUT_DIR = 'data'
LOG_DIR = 'logs'
REPORT_PATH = os.path.join(LOG_DIR, 'etl_report.csv')

# Crear carpetas si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configurar logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'etl_healthrisk.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========== Funciones del pipeline ==========

def extract_data(filepath):
    if not os.path.exists(filepath):
        logging.error(f"No se encontró el archivo: {filepath}")
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    logging.info("Datos extraídos correctamente.")
    return pd.read_csv(filepath)

def validate_data(df):
    issues = []

    if df.isnull().sum().any():
        issues.append("❗ Existen valores nulos en el dataset.")

    if df['age'].min() < 0 or df['age'].max() > 120:
        issues.append("❗ Valores fuera de rango en 'age'.")

    if df['bmi'].min() < 10 or df['bmi'].max() > 60:
        issues.append("❗ Valores fuera de rango en 'bmi'.")

    if df['children'].min() < 0 or df['children'].max() > 10:
        issues.append("❗ Valores fuera de rango en 'children'.")

    for col, valid_values in {
        'sex': ['male', 'female'],
        'smoker': ['yes', 'no'],
        'region': ['northeast', 'northwest', 'southeast', 'southwest']
    }.items():
        if not df[col].isin(valid_values).all():
            issues.append(f"❗ Valores inválidos en '{col}'.")

    if issues:
        for issue in issues:
            logging.warning(issue)
        raise ValueError("❌ Falló la validación de calidad de datos.")
    
    logging.info("✅ Validación de calidad de datos superada.")
    return df

def transform_data(df, threshold=15000):
    df['high_cost'] = (df['charges'] > threshold).astype(int)
    y = df['high_cost']
    df = df.drop(['charges'], axis=1)

    df_encoded = pd.get_dummies(df, drop_first=True)

    scaler = StandardScaler()
    cols_to_scale = ['age', 'bmi', 'children']
    df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])

    df_encoded['high_cost'] = y

    logging.info("✅ Transformación completada.")
    return df_encoded

def load_data(df_transformed):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'clean_insurance.csv'
    output_path = os.path.join(OUTPUT_DIR, filename)
    df_transformed.to_csv(output_path, index=False)
    logging.info(f"✅ Datos guardados en {output_path}")
    return output_path

def generate_etl_report(df, path=REPORT_PATH):
    report = {
        'Total registros': [df.shape[0]],
        'Nulos detectados': [df.isnull().sum().sum()],
        'Edad promedio': [round(df['age'].mean(), 2)],
        '% fumadores': [round(df[df['smoker'] == 'yes'].shape[0] / df.shape[0] * 100, 2)],
        'Fecha de ejecución': [datetime.now().strftime('%Y-%m-%d %H:%M')]
    }
    pd.DataFrame(report).to_csv(path, index=False)
    logging.info(f"📋 Reporte generado en {path}")

def notify_completion():
    logging.info("✅ Proceso ETL completado exitosamente.")
    print("📢 ETL finalizado correctamente. Ver logs y datos generados.")

# ========== Orquestador ==========

def run_etl():
    logging.info("🟡 Iniciando proceso ETL")
    try:
        raw_df = extract_data(DATA_PATH)
        validated_df = validate_data(raw_df)
        transformed_df = transform_data(validated_df)
        output_file = load_data(transformed_df)
        generate_etl_report(raw_df)
        notify_completion()
    except Exception as e:
        logging.error(f"❌ Error en el ETL: {e}")
        print(f"🚨 Error: {e}")

# ========== Ejecución ==========
if __name__ == "__main__":
    run_etl()
