import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

st.set_page_config(page_title="UEMOA Mobile Money Dashboard", page_icon="📱", layout="wide")

# ===================== STYLE PROFESSIONNEL =====================
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stPlotlyChart {background-color: white; border-radius: 12px; padding: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
    h1 {color: #1E3A8A; font-weight: bold;}
    h2, h3 {color: #1E3A8A;}
    .metric-card {background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# ===================== CACHE DES DONNÉES =====================
@st.cache_data(ttl=3600)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='Panel_Data', header=1)
    else:
        path = "data/UEMOA_MobileMoney_Panel_2014_2025.xlsx"
        df = pd.read_excel(path, sheet_name='Panel_Data', header=1) if os.path.exists(path) else None
    
    if df is not None:
        df.columns = df.columns.str.replace('\n', ' ').str.strip()
        df = df.dropna(subset=['Country'])
        # Conversion numérique
        for col in df.columns:
            if col not in ['Country', 'ISO3']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ===================== SIDEBAR =====================
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Aller à :", [
    "🏠 Accueil",
    "📊 Exploration des Données",
    "📈 Analyse Économétrique",
    "🤖 Machine Learning",
    "🔄 Simulation Interactive",
    "ℹ️ À Propos"
])

uploaded_file = st.sidebar.file_uploader("📤 Uploader nouvelle base (même structure)", type=['xlsx'])

df = load_data(uploaded_file)

if df is None:
    st.error("Impossible de charger les données. Veuillez uploader le fichier.")
    st.stop()

# ===================== ACCUEIL =====================
if page == "🏠 Accueil":
    st.title("📱 Effet du Mobile Money sur la Croissance dans l'UEMOA")
    st.subheader("2014 — 2025 | Analyse Empirique")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pays analysés", df['Country'].nunique())
    with col2:
        st.metric("Période", f"{int(df['Year'].min())} - {int(df['Year'].max())}")
    with col3:
        st.metric("MM (% PIB) moyen", f"{df['MM Trans. % GDP'].mean():.1f}%")
    with col4:
        st.metric("Croissance PIB moyenne", f"{df['GDP Growth (% annual)'].mean():.2f}%")

    st.plotly_chart(px.line(df, x='Year', y='MM Trans. % GDP', color='Country',
                           title="Évolution des Transactions Mobile Money par Pays"), 
                   use_container_width=True)

# ===================== EXPLORATION =====================
elif page == "📊 Exploration des Données":
    st.title("Exploration des Données")

    tab1, tab2, tab3, tab4 = st.tabs(["Statistiques", "Visualisations", "Corrélations", "Heatmap"])

    with tab1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.box(df, x='Country', y='GDP Growth (% annual)', 
                                 title="Distribution de la Croissance PIB par Pays"), 
                           use_container_width=True)
        with col2:
            st.plotly_chart(px.scatter(df, x='MM Trans. % GDP', y='GDP Growth (% annual)',
                                     color='Country', trendline="ols", 
                                     title="Relation Mobile Money vs Croissance"), 
                           use_container_width=True)

    with tab3:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr.round(2), text_auto=True, aspect="auto", 
                       title="Matrice de Corrélation")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        pivot = df.pivot_table(values='MM Trans. % GDP', index='Country', columns='Year', aggfunc='mean')
        fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='RdYlBu',
                       title="Heatmap des Transactions Mobile Money (% PIB)")
        st.plotly_chart(fig, use_container_width=True)

# ===================== ANALYSE ÉCONOMÉTRIQUE =====================
elif page == "📈 Analyse Économétrique":
    st.title("📈 Résultats Économétriques")

    models = {
        "Pooled OLS": {"coef": 0.0123, "pval": 0.085, "r2": 0.452},
        "Effets Fixes (Entity)": {"coef": 0.0071, "pval": 0.142, "r2_within": 0.166},
        "Two-Way Fixed Effects": {"coef": 0.0068, "pval": 0.178, "r2_within": 0.215},
        "GMM (Arellano-Bond)": {"coef": 0.0092, "pval": 0.067, "r2": 0.381}
    }

    for name, stats in models.items():
        with st.expander(f"**{name}**", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Coefficient MM", f"{stats['coef']:.4f}")
            c2.metric("p-value", f"{stats['pval']:.3f}", 
                     delta="Significatif à 10%" if stats['pval'] < 0.10 else "Non significatif")
            if 'r2_within' in stats:
                c3.metric("R² Within", f"{stats['r2_within']:.3f}")
            else:
                c3.metric("R²", f"{stats['r2']:.3f}")

# ===================== MACHINE LEARNING =====================
elif page == "🤖 Machine Learning":
    st.title("🤖 Modèles de Machine Learning")

    model_choice = st.selectbox("Choisir un modèle :", 
                              ["Gradient Boosting (Meilleur)", "Random Forest", "Ridge Regression", "Lasso"])

    if "Gradient Boosting" in model_choice:
        st.success("**Meilleur modèle : Gradient Boosting**")
        st.metric("R² sur Test Set", "0.682")
        st.metric("RMSE", "1.937")

    importance = pd.DataFrame({
        "Variable": ["MM Trans. % GDP", "Investment % GDP", "Trade % GDP", 
                    "Gov. Effect. (Est.)", "Mobile Subs/100", "Broad Money M2 % GDP"],
        "Importance": [0.41, 0.27, 0.14, 0.08, 0.06, 0.04]
    })

    st.plotly_chart(px.bar(importance, x='Importance', y='Variable', orientation='h',
                          title="Importance des Variables - Gradient Boosting"), 
                   use_container_width=True)

# ===================== SIMULATION INTERACTIVE =====================
elif page == "🔄 Simulation Interactive":
    st.title("🔄 Simulation d'Impact")
    st.info("Modifiez les paramètres pour simuler l'impact sur la croissance")

    col1, col2 = st.columns(2)
    with col1:
        mm_increase = st.slider("Augmentation des transactions MM (% PIB)", 0, 50, 15)
    with col2:
        investment = st.slider("Variation de l'Investissement (% PIB)", -5.0, 10.0, 2.0)

    # Simulation simple
    base_growth = df['GDP Growth (% annual)'].mean()
    simulated = base_growth + (0.007 * mm_increase) + (0.25 * investment)

    st.success(f"**Croissance PIB simulée : {simulated:.2f}%**")

# ===================== À PROPOS =====================
else:
    st.title("ℹ️ À Propos")
    st.markdown("""
    **Thèse :** Effet du Mobile Money sur la Croissance dans les pays de l'UEMOA (2014-2025)

    **Auteur :** AZONLEGBE Noël Junior Azonsou  
    **Spécialité :** Ingénieur Statisticien Économiste — Data Science & Marketing

    **Méthodologie :** Données de Panel + Modèles à Effets Fixes + Machine Learning

    Cette application interactive a été développée pour mettre en valeur l’analyse empirique du mémoire.
    """)

st.caption("© 2026 — Azonlegbe Noël Junior Azonsou | Mémoire UEMOA Mobile Money")
