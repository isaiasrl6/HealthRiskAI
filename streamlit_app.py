import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import plotly.express as px
from PIL import Image

# ========================
# CARGA DEL MODELO
# ========================
model_files = sorted(glob.glob('model/healthrisk_model_*.pkl'), reverse=True)
latest_model_path = model_files[0] if model_files else None

if latest_model_path:
    model = joblib.load(latest_model_path)
    st.sidebar.success(f"Modelo cargado: {os.path.basename(latest_model_path)}")
else:
    st.sidebar.error("❌ No se encontró un modelo entrenado.")
    st.stop()

# ========================
# INTERFAZ
# ========================
st.title("💡 HealthRisk AI")
st.caption("Predicción de gasto médico alto en afiliados utilizando Machine Learning")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Predicción",
    "📈 Métricas del Modelo",
    "ℹ️ Acerca del Proyecto",
    "📊 Dashboard Exploratorio"
])


# ========================
# TAB 4 - DASHBOARD
# ========================
with tab4:
    st.subheader("Análisis Exploratorio del Dataset")

    try:
        df_raw = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        st.error("❌ No se encontró 'insurance.csv'")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribución de edades:**")
        age_count = df_raw['age'].value_counts().sort_index().reset_index()
        age_count.columns = ['age', 'count']
        fig1 = px.bar(age_count, x='age', y='count', title='Distribución de Edades', color_discrete_sequence=['#C40000'])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**Gasto médico promedio por hijos:**")
        avg_by_children = df_raw.groupby('children')['charges'].mean().reset_index()
        fig2 = px.bar(avg_by_children, x='children', y='charges', title='Promedio de Charges por Hijos', color_discrete_sequence=['#C40000'])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Gasto médico por condición de fumador:**")
    smoker_group = df_raw.groupby('smoker')['charges'].mean().reset_index()
    fig3 = px.bar(smoker_group, x='smoker', y='charges', title='Gasto Médico según Fumador', color_discrete_sequence=['#C40000'])
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Gasto promedio por región:**")
    region_group = df_raw.groupby('region')['charges'].mean().reset_index()
    fig4 = px.bar(region_group, x='region', y='charges', title='Gasto Médico Promedio por Región', color_discrete_sequence=['#C40000'])
    st.plotly_chart(fig4, use_container_width=True)

from fpdf import FPDF
import plotly.io as pio
import tempfile

st.markdown("### 📥 Descargar Dashboard")

if st.button("Descargar en PDF"):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Guardar gráficos en PNG
        fig1_path = os.path.join(tmpdir, "edad.png")
        fig2_path = os.path.join(tmpdir, "hijos.png")
        fig3_path = os.path.join(tmpdir, "fumador.png")
        fig4_path = os.path.join(tmpdir, "region.png")
        
        pio.write_image(fig1, fig1_path, format='png', width=700, height=400)
        pio.write_image(fig2, fig2_path, format='png', width=700, height=400)
        pio.write_image(fig3, fig3_path, format='png', width=700, height=400)
        pio.write_image(fig4, fig4_path, format='png', width=700, height=400)

        # Crear PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "HealthRisk AI - Dashboard Exploratorio", ln=True, align='C')

        for img_path in [fig1_path, fig2_path, fig3_path, fig4_path]:
            pdf.image(img_path, w=190)
            pdf.ln(10)

        pdf_path = os.path.join(tmpdir, "dashboard.pdf")
        pdf.output(pdf_path)

        # Descargar
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Descargar Dashboard (PDF)",
                data=f,
                file_name="HealthRiskAI_Dashboard.pdf",
                mime="application/pdf"
            )


# ========================
# TAB 1 - PREDICCIÓN
# ========================
with tab1:
    st.subheader("Completa los datos del afiliado")

    age = st.slider("Edad", 18, 80, 30)
    sex = st.selectbox("Sexo", ["male", "female"])
    bmi = st.number_input("Índice de masa corporal (BMI)", min_value=10.0, max_value=60.0, value=25.0)
    children = st.selectbox("Cantidad de hijos", [0,1,2,3,4,5])
    smoker = st.selectbox("¿Fuma?", ["yes", "no"])
    region = st.selectbox("Región", ["northeast", "northwest", "southeast", "southwest"])

    def preprocess(age, sex, bmi, children, smoker, region):
        data = {
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex_male': [1 if sex == "male" else 0],
            'smoker_yes': [1 if smoker == "yes" else 0],
            'region_northwest': [1 if region == "northwest" else 0],
            'region_southeast': [1 if region == "southeast" else 0],
            'region_southwest': [1 if region == "southwest" else 0]
        }
        return pd.DataFrame(data)

    if st.button("🔍 Predecir gasto médico alto"):
        input_df = preprocess(age, sex, bmi, children, smoker, region)
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"🔴 Riesgo alto de gasto médico: {proba:.2%}")
        else:
            st.success(f"🟢 Riesgo bajo de gasto médico: {proba:.2%}")
        st.markdown("---")

# ========================
# TAB 2 - MÉTRICAS
# ========================
with tab2:
    st.subheader("Curva ROC del modelo")
    roc_curves = sorted(glob.glob("model/roc_curve_*.png"), reverse=True)
    if roc_curves:
        st.image(Image.open(roc_curves[0]), caption="Curva ROC", use_column_width=True)
    else:
        st.warning("No se encontró la curva ROC generada.")

    st.subheader("Importancia de variables")
    try:
        feature_importances = pd.Series(model.feature_importances_,
                                        index=input_df.columns).sort_values(ascending=False)
        st.bar_chart(feature_importances)
    except:
        st.warning("El modelo no tiene atributos de importancia disponibles.")

# ========================
# TAB 3 - ACERCA
# ========================
with tab3:
    st.subheader("Sobre HealthRisk AI")
    st.markdown("""
    **HealthRisk AI** es una herramienta de inteligencia artificial diseñada para predecir si un afiliado a una aseguradora tendrá un gasto médico elevado.
    
    **Tecnologías utilizadas:**
    - Python (pandas, scikit-learn, joblib)
    - Streamlit para visualización interactiva
    - Validaciones de calidad de datos (Data Quality)
    - Entrenamiento con validación AUC y logging

    **Autor:** Isaías Josué Rosario Luciano
    **Versión del modelo actual:** `""" + os.path.basename(latest_model_path).replace(".pkl", "") + "`"
    )


