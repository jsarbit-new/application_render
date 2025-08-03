import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
import boto3
from botocore.exceptions import ClientError
from sklearn.pipeline import Pipeline
import matplotlib
import plotly.graph_objects as go
import warnings
import tempfile
import joblib
from functools import lru_cache

# Supprimer les avertissements
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
matplotlib.use('Agg')

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Dashboard Crédit")

# Constantes et variables d'environnement
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
S3_PREFIX_DATA = "input/"
S3_MODEL_KEY = "modele_mlflow_final/model/model.pkl"
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI')

# URLs vers les images SHAP et le rapport Evidently sur S3
S3_SHAP_BEESWARM_URL = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION', 'eu-north-1')}.amazonaws.com/reports/shap_beeswarm.png"
S3_SHAP_BAR_URL = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION', 'eu-north-1')}.amazonaws.com/reports/shap_bar.png"
S3_EVIDENTLY_REPORT_URL = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION', 'eu-north-1')}.amazonaws.com/reports/evidently_report_10_lignes.html"

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
        return pipeline
    except ClientError as e:
        st.error(f"Erreur S3 : {e.response['Error']['Message']}")
        st.error("Vérifiez le nom du bucket et le chemin du modèle (S3_MODEL_KEY).")
        st.stop()
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du pipeline depuis S3: {str(e)}")
        st.stop()
        return None

# --- Main App ---
def main():
    st.title("📊 Analyse de Risque de Crédit")
    st.markdown("---")

    with st.spinner("Initialisation de l'application et chargement des ressources..."):
        s3 = init_s3()
        if s3 is None:
            st.stop()
        
        # Chargement des données de référence (pour le client existant et la simulation)
        X_train_raw = load_s3_parquet(s3, "X_train.parquet")
        if X_train_raw is None:
            st.stop()
        
        # Chargement du pipeline de prédiction
        prediction_pipeline = load_s3_model_pipeline()
        if prediction_pipeline is None:
            st.stop()

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
                st.subheader("Explications et analyses (Pré-calculées)")
                st.info("Les graphiques ci-dessous sont pré-calculés et chargés depuis S3 pour garantir la rapidité de l'application.")

                global_plots_tab1, global_plots_tab2 = st.tabs(["Explication Globale (Beeswarm)", "Explication Globale (Barres)"])
                with global_plots_tab1:
                    st.image(S3_SHAP_BEESWARM_URL, caption="Beeswarm Plot (Global)")

                with global_plots_tab2:
                    st.image(S3_SHAP_BAR_URL, caption="Bar Plot (Global)")

    with tab2:
        st.subheader("Simulation personnalisée")
        with st.form("simulation_form"):
            st.write("Modifiez les valeurs des 10 features ci-dessous pour simuler un nouveau client:")
            simulation_features = {
                "AMT_CREDIT": "Montant du prêt",
                "AMT_ANNUITY": "Montant Annuité",
                "PAYMENT_RATE": "Ratio Crédit/Annuité",
                "DAYS_EMPLOYED": "Ancienneté emploi (jours)",
                "REGION_POPULATION_RELATIVE": "Taux de population région",
                "EXT_SOURCE_1": "Source Extérieure 1",
                "EXT_SOURCE_2": "Source Extérieure 2",
                "EXT_SOURCE_3": "Source Extérieure 3",
                "CNT_CHILDREN": "Nombre d'enfants",
                "DAYS_BIRTH": "Âge client (jours)"
            }
            default_values = X_train_raw.iloc[0].to_dict()
            editable_features = {}
            for col, label in simulation_features.items():
                if col in X_train_raw.columns:
                    min_val = X_train_raw[col].min()
                    max_val = X_train_raw[col].max()
                    default_value = default_values.get(col, X_train_raw[col].mean())
                    st.write(f"**{label}**")
                    st.write(f"  *Plage de valeurs possibles : de {min_val:.2f} à {max_val:.2f}*")
                    if X_train_raw[col].dtype in ['int64', 'int32']:
                        value_input = st.number_input(
                            'Entrez une valeur',
                            value=int(default_value),
                            min_value=int(min_val),
                            max_value=int(max_val),
                            key=col
                        )
                    else:
                        value_input = st.number_input(
                            'Entrez une valeur',
                            value=float(default_value),
                            min_value=float(min_val),
                            max_value=float(max_val),
                            key=col
                        )
                    editable_features[col] = value_input

            if st.form_submit_button("Prédire le risque"):
                with st.spinner("Simulation en cours..."):
                    sim_data_raw = pd.DataFrame([default_values])
                    for feat, val in editable_features.items():
                        sim_data_raw.loc[0, feat] = val
                    sim_data_raw = sim_data_raw.astype(X_train_raw.dtypes)

                    proba = prediction_pipeline.predict_proba(sim_data_raw)[0][1]
                    st.success(f"**Probabilité de défaut estimée : {proba:.2%}**")

                    st.markdown("---")
                    st.subheader("Explications et analyses (Pré-calculées)")
                    st.info("Les graphiques ci-dessous sont pré-calculés et chargés depuis S3 pour garantir la rapidité de l'application.")

                    global_plots_tab1, global_plots_tab2 = st.tabs(["Explication Globale (Beeswarm)", "Explication Globale (Barres)"])
                    with global_plots_tab1:
                        st.image(S3_SHAP_BEESWARM_URL, caption="Beeswarm Plot (Global)")

                    with global_plots_tab2:
                        st.image(S3_SHAP_BAR_URL, caption="Bar Plot (Global)")

                    st.markdown("---")
                    st.write("**Analyse de Data Drift (Evidently AI)**")
                    st.info("Le rapport de dérive des données est généré séparément pour optimiser les ressources. Vous pouvez le consulter en cliquant sur le lien ci-dessous.")
                    st.markdown(f"**[Accéder au rapport Evidently]({S3_EVIDENTLY_REPORT_URL})**")

if __name__ == "__main__":
    main()