import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
import warnings
import tempfile
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib

# Supprimer les avertissements
warnings.filterwarnings('ignore', category=UserWarning, module='shap')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="X does not have valid feature names")
matplotlib.use('Agg')

# --- Configuration ---
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
S3_PREFIX_DATA = "input/"
S3_MODEL_KEY = "modele_mlflow_final/model/model.pkl"

# Fichiers de sortie pour les plots SHAP
S3_SHAP_BEESWARM_KEY = "reports/shap_beeswarm.png"
S3_SHAP_BAR_KEY = "reports/shap_bar.png"

# --- Helpers S3 (Réutilisation des fonctions existantes) ---
def init_s3():
    try:
        s3 = boto3.client('s3',
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                          region_name=os.getenv('AWS_REGION', 'eu-north-1'))
        return s3
    except Exception as e:
        print(f"Erreur lors de l'initialisation de S3: {str(e)}")
        return None

def load_s3_parquet(s3, key):
    if not BUCKET_NAME:
        print("Erreur: Variable d'environnement 'AWS_S3_BUCKET_NAME' non définie.")
        return None
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_PREFIX_DATA + key)
        print(f"Fichier {key} chargé avec succès.")
        return pd.read_parquet(BytesIO(obj['Body'].read()))
    except ClientError as e:
        print(f"Erreur S3 ({key}): {e.response['Error']['Message']}")
        return None

def load_s3_model_pipeline(s3):
    try:
        print(f"Chargement du modèle depuis s3://{BUCKET_NAME}/{S3_MODEL_KEY}")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "model.pkl")
            s3.download_file(BUCKET_NAME, S3_MODEL_KEY, temp_file_path)
            pipeline = joblib.load(temp_file_path)
        print("Pipeline chargé avec succès depuis S3.")
        return pipeline
    except ClientError as e:
        print(f"Erreur S3 : {e.response['Error']['Message']}")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement du pipeline depuis S3: {str(e)}")
        return None

def upload_file_to_s3(s3_client, file_path, bucket_name, object_name):
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Fichier {object_name} téléchargé avec succès sur S3.")
        return True
    except ClientError as e:
        print(f"Erreur lors du téléchargement vers S3 : {e.response['Error']['Message']}")
        return False

# --- Fonctions SHAP (adaptées pour un script externe) ---
def train_explainer_model(_preprocessor, X_raw, y):
    print("Entraînement d'un modèle local pour l'explicabilité (SHAP)...")
    X_processed = _preprocessor.transform(X_raw)
    feature_names = [f.split('__')[-1] for f in _preprocessor.get_feature_names_out()]
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X_raw.index)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_processed_df, y.squeeze())
    print("Modèle SHAP local et données prétraitées sont prêts.")
    return model, X_processed_df

def create_explainer(_model, _data):
    return shap.Explainer(_model, _data)

# --- Fonction de génération des plots ---
def generate_and_upload_shap_plots(s3_client, bucket_name, explainer, data):
    print("Génération des plots SHAP globaux...")
    shap_values = explainer(data)

    # Beeswarm Plot
    fig_beeswarm = plt.figure(figsize=(15, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("Importance globale des features (Beeswarm Plot)")
    plt.tight_layout()
    # Sauvegarde et téléchargement du fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        tmp_filename = tmpfile.name
        fig_beeswarm.savefig(tmp_filename, bbox_inches='tight')
    plt.close(fig_beeswarm) # Ferme la figure pour libérer le descripteur de fichier
    upload_file_to_s3(s3_client, tmp_filename, bucket_name, S3_SHAP_BEESWARM_KEY)
    os.remove(tmp_filename) # Supprime le fichier une fois le téléchargement terminé

    # Bar Plot
    fig_bar = plt.figure(figsize=(15, 8))
    shap.plots.bar(shap_values.abs.mean(0), max_display=20, show=False)
    plt.title("Importance moyenne globale des features (Bar Plot)")
    plt.tight_layout()
    # Sauvegarde et téléchargement du fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        tmp_filename = tmpfile.name
        fig_bar.savefig(tmp_filename, bbox_inches='tight')
    plt.close(fig_bar) # Ferme la figure pour libérer le descripteur de fichier
    upload_file_to_s3(s3_client, tmp_filename, bucket_name, S3_SHAP_BAR_KEY)
    os.remove(tmp_filename) # Supprime le fichier une fois le téléchargement terminé

    print("Plots SHAP globaux générés et téléchargés sur S3.")

# --- Fonction principale du script ---
def main():
    print("Début du script de génération de plots SHAP.")
    s3 = init_s3()
    if s3 is None: return

    X_train_raw = load_s3_parquet(s3, "X_train.parquet")
    y_train = load_s3_parquet(s3, "y_train.parquet")
    if X_train_raw is None or y_train is None: return

    prediction_pipeline = load_s3_model_pipeline(s3)
    if prediction_pipeline is None: return

    preprocessor = prediction_pipeline.named_steps.get('preprocessor')
    if preprocessor is None:
        print("Erreur: Le pipeline n'a pas d'étape 'preprocessor'.")
        return

    explainer_model, X_train_processed_df = train_explainer_model(preprocessor, X_train_raw, y_train)
    explainer = create_explainer(explainer_model, X_train_processed_df)

    generate_and_upload_shap_plots(s3, BUCKET_NAME, explainer, X_train_processed_df)
    
    print("Script terminé.")

if __name__ == "__main__":
    main()