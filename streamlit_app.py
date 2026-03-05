import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ========================
# CONFIGURACIÓN DE PÁGINA
# ========================
st.set_page_config(
    page_title="HealthRisk AI | by Isaias Rosario",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# COLORES CORPORATIVOS SENASA (Base)
SENASA_GREEN = "#39B54A"
SENASA_RED = "#C40000"

# Estilo CSS Dinámico (Soporta Dark & Light Theme)
st.markdown(f"""
    <style>
    /* Uso de variables de Streamlit para adaptabilidad */
    .main {{ 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    }}
    
    /* Card Style: Usamos 'secondaryBackgroundColor' de Streamlit para el fondo de la tarjeta */
    div[data-testid="stVerticalBlock"] > div:has(div.stForm) {{
        background-color: var(--secondary-background-color);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    /* Botones Pro con gradiente adaptable */
    .stButton>button {{ 
        width: 100%; 
        border-radius: 12px; 
        height: 3.5em; 
        background: linear-gradient(135deg, {SENASA_GREEN} 0%, #2e8b3a 100%);
        color: white !important; 
        border: none;
        font-weight: 600;
    }}
    
    /* Tabs: Quitamos fondos blancos fijos */
    .stTabs [data-baseweb="tab-list"] {{ 
        gap: 8px; 
    }}
    .stTabs [data-baseweb="tab"] {{ 
        border-radius: 10px 10px 0 0; 
        padding: 12px 25px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: {SENASA_GREEN} !important; 
        color: white !important; 
    }}

    /* Ajuste de métricas para que el texto no se pierda */
    [data-testid="stMetricValue"] {{
        color: {SENASA_GREEN};
    }}
    </style>
    """, unsafe_allow_html=True)

# ========================
# CARGA DEL MODELO
# ========================
@st.cache_resource
def load_latest_model():
    files = sorted(glob.glob('model/healthrisk_model_*.pkl'), reverse=True)
    if not files: return None
    return joblib.load(files[0]), os.path.basename(files[0])

model_data = load_latest_model()
if model_data:
    model, model_name = model_data
    # Sidebar con color adaptable
    st.sidebar.success(f"📌 **Modelo Activo:**\n{model_name}")
else:
    st.error("No se detectó motor de IA.")
    st.stop()

# ========================
# ENCABEZADO
# ========================
with st.container():
    c1, c2 = st.columns([1, 6])
    with c1:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) 
    with c2:
        st.title("HealthRisk AI")
        st.markdown("##### Plataforma de Análisis Predictivo de Siniestralidad")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Evaluación", "📊 Inteligencia de Datos", "⚙️ Auditoría IA", "📖 Documentación"])

# ========================
# TAB 1 - EVALUACIÓN
# ========================
with tab1:
    col_form, col_res = st.columns([1.2, 1])
    
    with col_form:
        st.subheader("📋 Perfil del Afiliado")
        with st.form("main_form", border=False):
            c_a, c_b = st.columns(2)
            with c_a:
                age = st.number_input("Edad Actual", 18, 100, 30)
                sex = st.selectbox("Sexo Biológico", ["male", "female"])
                bmi = st.slider("IMC (Índice Masa Corporal)", 10.0, 50.0, 24.5)
            with c_b:
                smoker = st.radio("¿Fuma?", ["no", "yes"], horizontal=True)
                children = st.number_input("Dependientes", 0, 10, 0)
                region = st.selectbox("Región", ["northeast", "northwest", "southeast", "southwest"])
            
            submitted = st.form_submit_button("GENERAR DIAGNÓSTICO")

    with col_res:
        if submitted:
            input_data = pd.DataFrame({
                'age': [age], 'bmi': [bmi], 'children': [children],
                'sex_male': [1 if sex == "male" else 0],
                'smoker_yes': [1 if smoker == "yes" else 0],
                'region_northwest': [1 if region == "northwest" else 0],
                'region_southeast': [1 if region == "southeast" else 0],
                'region_southwest': [1 if region == "southwest" else 0]
            })
            
            proba = model.predict_proba(input_data)[0][1]
            
            st.subheader("🩺 Diagnóstico")
            
            # Gauge Chart Adaptable (Usamos 'template' para que Plotly detecte el Dark Mode)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba * 100,
                number = {'suffix': "%", 'font': {'color': SENASA_GREEN if proba < 0.5 else SENASA_RED}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': SENASA_GREEN if proba < 0.5 else SENASA_RED},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(57, 181, 74, 0.1)"},
                        {'range': [50, 100], 'color': "rgba(196, 0, 0, 0.1)"}]
                }
            ))
            fig_gauge.update_layout(
                height=250, 
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)', # Fondo transparente
                plot_bgcolor='rgba(0,0,0,0)',
                font = {'color': "gray"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            if proba > 0.5:
                st.warning(f"**RIESGO ELEVADO:** Probabilidad de alto costo detectada.")
            else:
                st.success(f"**RIESGO BAJO:** Perfil dentro de parámetros normales.")
        else:
            st.info("Esperando datos para diagnóstico...")

# ========================
# TAB 2 - DASHBOARD
# ========================
with tab2:
    try:
        df = pd.read_csv("insurance.csv")
        st.subheader("🔬 Insights Poblacionales")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Costo Total", f"${df['charges'].sum()/1e6:.1f}M")
        m2.metric("Siniestralidad Promedio", f"${df['charges'].mean():,.0f}")
        m3.metric("Casos Críticos (IA)", "18.4%")

        c1, c2 = st.columns(2)
        with c1:
            # Gráficos con template="none" para que hereden el tema del navegador
            fig_scatter = px.scatter(df, x="age", y="charges", color="smoker", 
                                    color_discrete_map={"yes": SENASA_RED, "no": SENASA_GREEN},
                                    template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark")
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)
        with c2:
            fig_box = px.box(df, x="region", y="charges", color="region",
                            template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark")
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, use_container_width=True)
    except:
        st.error("Archivo 'insurance.csv' no encontrado.")

# ========================
# TAB 3 & 4
# ========================
with tab3:
    st.subheader("📊 Importancia de Variables")
    try:
        importances = pd.Series(model.feature_importances_, index=['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_NW', 'region_SE', 'region_SW']).sort_values()
        fig_imp = px.bar(importances, orientation='h', color_discrete_sequence=[SENASA_GREEN])
        fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)
    except: st.write("Datos de auditoría no disponibles.")

with tab4:
    st.markdown(f"**Autor:** {st.session_state.get('user_name', 'Isaías Rosario Luciano')}")
    st.write("Versión 2.0 - Optimización 2026.")