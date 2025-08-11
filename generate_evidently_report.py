import os
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from io import BytesIO
import warnings
import tempfile
import joblib

# Supprimer les avertissements pour une meilleure lisibilité
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
S3_PREFIX_DATA = "input/"
S3_MODEL_KEY = "modele_mlflow_final/model/model.pkl"
S3_REPORT_KEY = "reports/evidently_report_10_lignes.html"  # Nouveau nom pour le fichier

# --- Helpers S3 ---
def init_s3():
    """Initialisation sécurisée de S3"""
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
    """Charge un fichier Parquet depuis S3"""
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
    """
    Charge le pipeline MLflow complet depuis S3.
    """
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
    """Télécharge un fichier local vers un bucket S3."""
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Fichier {object_name} téléchargé avec succès sur S3.")
        return True
    except ClientError as e:
        print(f"Erreur lors du téléchargement vers S3 : {e.response['Error']['Message']}")
        return False

# --- Fonction principale du script ---
def generate_and_upload_report():
    print("Début du script de génération de rapport Evidently (10 lignes).")

    # 1. Initialisation S3 et chargement des données
    s3 = init_s3()
    if s3 is None:
        return

    X_train_raw = load_s3_parquet(s3, "X_train.parquet")
    if X_train_raw is None:
        return

    pipeline = load_s3_model_pipeline(s3)
    if pipeline is None:
        return

    preprocessor = pipeline.named_steps.get('preprocessor')
    if preprocessor is None:
        print("Erreur: Le pipeline n'a pas d'étape 'preprocessor'.")
        return

    # 2. Prétraitement des données pour le rapport (échantillon de 10 lignes)
    print("Prétraitement d'un échantillon de 10 lignes pour le rapport...")
    
    # Échantillon de 10 lignes pour les données de référence
    reference_data = X_train_raw.sample(n=10, random_state=42)
    reference_data_processed = preprocessor.transform(reference_data)
    feature_names = [f.split('__')[-1] for f in preprocessor.get_feature_names_out()]
    reference_data_for_report = pd.DataFrame(reference_data_processed, columns=feature_names, index=reference_data.index)

    # Création d'un échantillon de "données actuelles" pour la comparaison
    # Ici, nous dupliquons simplement les données de référence. 
    # Pour une simulation plus réaliste, vous pourriez modifier quelques valeurs.
    current_data_for_report = reference_data_for_report.copy()
    
    # 3. Génération du rapport Evidently
    print("Génération du rapport de dérive des données Evidently...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference_data_for_report,
        current_data=current_data_for_report
    )

    # 4. Sauvegarde temporaire du rapport
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmpfile:
        data_drift_report.save_html(tmpfile.name)
        report_file_path = tmpfile.name
    
    print(f"Rapport HTML généré localement : {report_file_path}")

    # 5. Téléchargement du rapport sur S3
    upload_file_to_s3(s3, report_file_path, BUCKET_NAME, S3_REPORT_KEY)

    # 6. Nettoyage du fichier temporaire
    os.remove(report_file_path)
    print(f"Fichier temporaire {report_file_path} supprimé.")

    print("Script terminé.")

if __name__ == "__main__":
    generate_and_upload_report()