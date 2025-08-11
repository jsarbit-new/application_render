import pytest
import pandas as pd
import numpy as np
import os
import joblib
import tempfile
from io import BytesIO
import boto3
from botocore.exceptions import ClientError
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys

# Ajoutez le chemin du répertoire parent pour importer app.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importez les fonctions que nous voulons tester depuis app.py
from app import load_s3_model_pipeline, load_s3_parquet, init_s3

# --- Fixtures Pytest pour la préparation de l'environnement ---

@pytest.fixture(scope="session")
def s3_client():
    """Fixture pour initialiser le client S3 une seule fois."""
    return init_s3()

@pytest.fixture(scope="session")
def prediction_pipeline():
    """Fixture pour charger le pipeline de prédiction une seule fois."""
    pipeline = load_s3_model_pipeline()
    if pipeline is None:
        pytest.fail("Impossible de charger le pipeline de prédiction.")
    return pipeline

@pytest.fixture(scope="session")
def sample_data(s3_client):
    """Fixture pour charger les données de test une seule fois."""
    data = load_s3_parquet(s3_client, "X_test.parquet")
    if data is None:
        pytest.fail("Impossible de charger les données de test.")
    return data.iloc[:10]  # Utiliser un échantillon de 10 lignes pour le test

# --- Tests ---

def test_pipeline_has_predict_proba(prediction_pipeline):
    """
    Teste que le pipeline chargé a bien une méthode `predict_proba`.
    Cela vérifie que le pipeline est complet (avec un modèle final).
    """
    assert hasattr(prediction_pipeline, 'predict_proba'), "Le pipeline chargé n'a pas de méthode predict_proba. Vérifiez l'enregistrement MLflow."

def test_prediction_pipeline_works(prediction_pipeline, sample_data):
    """
    Teste que le pipeline de prédiction renvoie des probabilités valides
    pour des données d'entrée.
    """
    print("\n[TEST] Exécution du test de prédiction...")
    
    # 1. Vérifier que les fixtures sont bien chargées
    assert prediction_pipeline is not None
    assert isinstance(sample_data, pd.DataFrame)
    assert not sample_data.empty

    # 2. Effectuer la prédiction
    try:
        probabilities = prediction_pipeline.predict_proba(sample_data)
    except Exception as e:
        pytest.fail(f"Erreur lors de la prédiction avec le pipeline : {e}")

    # 3. Vérifier le format et les valeurs des résultats
    print(f"Probabilités obtenues : {probabilities}")
    
    # Le résultat doit être un tableau numpy
    assert isinstance(probabilities, np.ndarray)
    
    # Pour 10 échantillons, le résultat doit avoir une forme de (10, 2)
    assert probabilities.shape == (10, 2)
    
    # Les probabilités doivent être entre 0 et 1
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    
    # La somme des probabilités de chaque ligne doit être très proche de 1
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)
    
    print("[TEST] Le test de prédiction a réussi.")