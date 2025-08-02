import streamlit as st
import pandas as pd
import mlflow
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
import boto3
from botocore.exceptions import ClientError
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib
import json
from PIL import Image
import plotly.graph_objects as go
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import warnings
import tempfile
import joblib
from functools import lru_cache

# Supprimer les avertissements pour une meilleure lisibilité
warnings.filterwarnings('ignore', category=UserWarning, module='shap')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="X does not have valid feature names")
matplotlib.use('Agg')

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Dashboard Crédit")

# Constantes et variables d'environnement
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
S3_PREFIX_DATA = "input/"
S3_MODEL_KEY = "modele_mlflow_final/model/model.pkl"
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI')

# --- Helpers S3 ---
@st.cache_resource
def init_s3():
    """Initialisation sécurisée de S3"""
    try:
        s3 = boto3.client('s3',
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                          region_name=os.getenv('AWS_REGION', 'eu-north-1'))
        return s3
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de S3: {str(e)}")
        st.info("Vérifiez que les variables d'environnement AWS sont bien définies.")
        st.stop()
        return None

@st.cache_data
def load_s3_parquet(_s3, key):
    """Charge un fichier Parquet depuis S3"""
    if not BUCKET_NAME:
        st.error("Variable d'environnement 'AWS_S3_BUCKET_NAME' non définie.")
        st.stop()
    try:
        obj = _s3.get_object(Bucket=BUCKET_NAME, Key=S3_PREFIX_DATA + key)
        return pd.read_parquet(BytesIO(obj['Body'].read()))
    except ClientError as e:
        st.error(f"Erreur S3 ({key}): {e.response['Error']['Message']}")
        st.info("Vérifiez que le nom du fichier et le chemin S3 sont corrects.")
        st.stop()
        return None

# --- Helpers de chargement du modèle depuis S3 ---
@st.cache_resource
def load_s3_model_pipeline():
    """
    Charge le pipeline MLflow complet depuis S3 en utilisant joblib.
    Utilise un fichier temporaire pour éviter les problèmes de streaming.
    """
    s3_client = init_s3()
    if s3_client is None:
        st.stop()

    try:
        st.info(f"Chargement du modèle depuis s3://{BUCKET_NAME}/{S3_MODEL_KEY}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "model.pkl")

            s3_client.download_file(BUCKET_NAME, S3_MODEL_KEY, temp_file_path)
            pipeline = joblib.load(temp_file_path)

        st.success("Pipeline chargé avec succès depuis S3.")

        feature_names = None
        if isinstance(pipeline, Pipeline):
            try:
                preprocessor = pipeline.named_steps['preprocessor']
                feature_names = preprocessor.get_feature_names_out()
                feature_names = [f.split('__')[-1] for f in feature_names]
            except Exception as e:
                st.warning(f"Impossible de récupérer les noms de features du pipeline. Erreur: {e}")
                feature_names = None

        return pipeline, feature_names

    except ClientError as e:
        st.error(f"Erreur S3 : {e.response['Error']['Message']}")
        st.error("Vérifiez le nom du bucket et le chemin du modèle (S3_MODEL_KEY).")
        st.stop()
        return None, None
    except Exception as e:
        st.error(f"Erreur lors du chargement du pipeline depuis S3: {str(e)}")
        st.stop()
        return None, None

# --- Helpers SHAP ---
@st.cache_resource
def train_explainer_model(_preprocessor, X_raw, y):
    """
    Entraîne un nouveau modèle de régression logistique pour l'explicabilité
    et retourne les données prétraitées avec les noms de features.
    """
    st.info("Entraînement d'un modèle local pour l'explicabilité (SHAP)...")

    # Prétraitement des données
    X_processed = _preprocessor.transform(X_raw)

    # Création du DataFrame avec les noms de features pour SHAP
    feature_names = [f.split('__')[-1] for f in _preprocessor.get_feature_names_out()]
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X_raw.index)

    # Entraînement du modèle
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_processed_df, y.squeeze())

    st.success("Modèle SHAP local et données prétraitées pour l'explainer sont prêts.")
    return model, X_processed_df

@st.cache_resource
def create_explainer(_model, _data):
    """Crée l'explainer SHAP une seule fois."""
    return shap.Explainer(_model, _data)

@st.cache_resource
def generate_global_shap_plots(_explainer, _data):
    """Génère les plots SHAP globaux (Beeswarm et Bar) et retourne les 10 features les plus importantes."""
    shap_values = _explainer(_data)

    # Récupération des 10 features les plus importantes
    if hasattr(shap_values, 'abs'):
        feature_importance_df = pd.DataFrame({
            'feature': _data.columns,
            'importance': np.abs(shap_values.values).mean(0)
        })
        top_10_features = feature_importance_df.sort_values(by='importance', ascending=False)['feature'].head(10).tolist()
    else:
        # Fallback si l'objet shap_values n'est pas un Explainer standard
        top_10_features = _data.columns.tolist()[:10]

    # Génération des images
    shap_images = {}

    # Beeswarm Plot
    fig_beeswarm = plt.figure(figsize=(15, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("Importance globale des features (Beeswarm Plot)")
    plt.tight_layout()
    shap_images['beeswarm_plot'] = fig_beeswarm

    # Bar Plot
    fig_bar = plt.figure(figsize=(15, 8))
    shap.plots.bar(shap_values.abs.mean(0), max_display=20, show=False)
    plt.title("Importance moyenne globale des features (Bar Plot)")
    plt.tight_layout()
    shap_images['bar_plot'] = fig_bar

    return shap_images, top_10_features

# --- Main App ---
def main():
    st.title("📊 Analyse de Risque de Crédit")
    st.markdown("---")

    with st.spinner("Initialisation de l'application et chargement des ressources..."):
        s3 = init_s3()
        if s3 is None:
            st.stop()

        X_train_raw = load_s3_parquet(s3, "X_train.parquet")
        y_train = load_s3_parquet(s3, "y_train.parquet")
        if X_train_raw is None or y_train is None:
            st.stop()

        prediction_pipeline, feature_names = load_s3_model_pipeline()
        if prediction_pipeline is None:
            st.stop()

        preprocessor = prediction_pipeline.named_steps.get('preprocessor')
        if preprocessor is None:
            st.error("Le pipeline MLflow n'a pas d'étape 'preprocessor'.")
            st.stop()

        explainer_model, X_train_processed_df = train_explainer_model(preprocessor, X_train_raw, y_train)

        explainer = create_explainer(explainer_model, X_train_processed_df)
        expected_value = explainer.expected_value

        # Récupération des plots et des 10 features les plus importantes
        shap_images, top_10_features = generate_global_shap_plots(explainer, X_train_processed_df)

    st.success("Toutes les ressources sont chargées avec succès !")
    st.markdown("---")

    # --- UI ---
    tab1, tab2 = st.tabs(["🔍 Analyse Client", "🧮 Simulateur"])

    with tab1:
        st.subheader("Analyse d'un client existant")
        client_id_list = X_train_raw.index.tolist()
        client_id = st.sidebar.selectbox("Sélectionnez un ID client:", client_id_list, index=0)

        client_data_raw = X_train_raw.loc[[client_id]]

        if st.button("Calculer le risque pour ce client"):
            with st.spinner("Calcul en cours..."):
                # PRÉDICTION : Utilise le pipeline MLflow
                proba = prediction_pipeline.predict_proba(client_data_raw)[0][1]

                # EXPLICATION : Utilise le modèle local de l'explainer
                client_data_processed = preprocessor.transform(client_data_raw)
                client_data_processed_df = pd.DataFrame(client_data_processed, columns=X_train_processed_df.columns)
                shap_values = explainer(client_data_processed_df)

                st.subheader("Résultat de la prédiction")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilité de défaut", f"{proba:.2%}")
                    threshold = 0.5
                    if proba > threshold:
                        st.error("❌ Décision : Prêt refusé")
                    else:
                        st.success("✅ Décision : Prêt accordé")

                with col2:
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number", value=proba * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Score de Risque de Défaut"},
                            gauge={'axis': {'range': [None, 100]},
                                   'bar': {'color': "darkblue"},
                                   'steps': [{'range': [0, threshold * 100], 'color': "lightgreen"},
                                             {'range': [threshold * 100, 100], 'color': "salmon"}],
                                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold * 100}}
                        )
                    )
                    st.plotly_chart(fig_gauge)

                st.markdown("---")
                st.subheader("Explication de la prédiction (SHAP)")

                st.write("**Explication locale (Waterfall Plot)**")
                fig_shap_waterfall = plt.figure(figsize=(12, 6))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig_shap_waterfall, bbox_inches='tight')

                st.markdown("---")
                global_plots_tab1, global_plots_tab2 = st.tabs(["Explication Globale (Beeswarm)", "Explication Globale (Barres)"])
                with global_plots_tab1:
                    st.pyplot(shap_images['beeswarm_plot'], bbox_inches='tight')

                with global_plots_tab2:
                    st.pyplot(shap_images['bar_plot'], bbox_inches='tight')

    with tab2:
        st.subheader("Simulation personnalisée")
        with st.form("simulation_form"):
            st.write("Modifiez les valeurs des 10 features les plus importantes pour simuler un nouveau client:")

            # Initialisation du dictionnaire avec les valeurs par défaut
            default_values = X_train_raw.iloc[0].to_dict()

            editable_features = {}
            for col in top_10_features:
                if col in X_train_raw.columns:
                    # Assurez-vous que la valeur par défaut est du même type que le min/max du slider
                    default_value_for_slider = float(default_values.get(col, X_train_raw[col].mean()))
                    min_val = float(X_train_raw[col].min())
                    max_val = float(X_train_raw[col].max())

                    editable_features[col] = st.slider(
                        col.replace('_', ' ').title(),
                        min_val, max_val, default_value_for_slider
                    )

            if st.form_submit_button("Prédire le risque"):
                with st.spinner("Simulation en cours..."):
                    # Création du DataFrame de simulation avec les valeurs par défaut
                    sim_data_raw = pd.DataFrame([default_values])

                    # Mise à jour des valeurs avec les sliders
                    for feat, val in editable_features.items():
                        sim_data_raw.loc[0, feat] = val
                    
                    # Correction des types de colonnes
                    sim_data_raw = sim_data_raw.astype(X_train_raw.dtypes)

                    # PRÉDICTION
                    proba = prediction_pipeline.predict_proba(sim_data_raw)[0][1]
                    st.success(f"**Probabilité de défaut estimée : {proba:.2%}**")

                    # EXPLICATION SHAP
                    sim_data_processed = preprocessor.transform(sim_data_raw)
                    sim_data_processed_df = pd.DataFrame(sim_data_processed, columns=X_train_processed_df.columns)

                    shap_values = explainer(sim_data_processed_df)

                    st.write("**Explication de la prédiction pour le client simulé (Waterfall Plot)**")
                    fig_shap_waterfall_sim = plt.figure(figsize=(12, 6))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig_shap_waterfall_sim, bbox_inches='tight')

                    st.markdown("---")
                    global_plots_tab1, global_plots_tab2 = st.tabs(["Explication Globale (Beeswarm)", "Explication Globale (Barres)"])
                    with global_plots_tab1:
                        st.pyplot(shap_images['beeswarm_plot'], bbox_inches='tight')

                    with global_plots_tab2:
                        st.pyplot(shap_images['bar_plot'], bbox_inches='tight')

                    st.markdown("---")
                    
                    # AJOUT D'UN SPINNER EXPLICITE POUR CETTE OPÉRATION LONGUE
                    with st.spinner("Génération du rapport de dérive des données (Evidently)... Cela peut prendre quelques dizaines de secondes."):
                        st.write("**Analyse de Data Drift (Evidently AI)**")

                        reference_data_for_drift = X_train_processed_df
                        current_data_for_drift = sim_data_processed_df

                        # La logique pour dupliquer la ligne est toujours nécessaire si les données actuelles n'ont qu'une seule ligne
                        if len(current_data_for_drift) == 1:
                            current_data_for_drift = pd.concat([current_data_for_drift] * 5, ignore_index=True)

                        data_drift_report = Report(metrics=[DataDriftPreset()])
                        
                        # --- MODIFICATION CLÉ : RÉDUCTION DE L'ÉCHANTILLON À 100 LIGNES ---
                        data_drift_report.run(
                            reference_data=reference_data_for_drift.sample(n=min(100, len(reference_data_for_drift))),
                            current_data=current_data_for_drift)

                        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmpfile:
                            data_drift_report.save_html(tmpfile.name)

                        try:
                            with open(tmpfile.name, 'r', encoding='utf-8') as f:
                                report_html_content = f.read()

                            st.components.v1.html(report_html_content, height=500, scrolling=True)
                        finally:
                            os.remove(tmpfile.name)

if __name__ == "__main__":
    main()