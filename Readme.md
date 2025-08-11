Fichier Readme.md

# Projet 7 : Implémentation d'un Modèle de Scoring

1. feature-eng-local-et-cloud-3.py

Ce script est dédié à la préparation des données et à la gestion du modèle de scoring.

    Traitement des données : Il effectue le feature engineering en local, générant les fichiers intermédiaires X_test.parquet, X_train.parquet, Y_test.parquet, Y_train.parquet et metadata.json. En raison d'une mauvaise connexion initiale, ces fichiers ont été chargés manuellement sur un bucket AWS S3.

    Gestion du modèle : Le pipeline de modélisation (feature engineering + encodage + imputation + régression logistique) est enregistré sur MLflow. Par souci de simplicité pour le déploiement, le modèle final a été extrait de MLflow et également chargé manuellement dans le bucket S3.

2. streamit app-AWS-eslatic-success-2-10lignes-drift.-interface-fixe.py

Ce script est la version complète et en temps réel de l'application Streamlit.

    Fonctionnalités : Il effectue tous les calculs dynamiquement, incluant :

        La probabilité de défaut pour un client existant ou pour la simulation d'un nouveau client.

        Le calcul des valeurs SHAP pour expliquer la prédiction du modèle.

        La génération d'un rapport de dérive (drift) avec Evidently pour 10 lignes de données.

    Exécution : Ce script n'est pas conçu pour les services cloud gratuits. Il doit être exécuté via une console Python sur un ordinateur local, avec les secrets de connexion au bucket AWS S3 correctement définis.

3. app-AWS-eslatic-success-2-10lignes-drift-html.-interface-fixe-et-shap-static.py

Ce script est une version optimisée pour le déploiement sur la plateforme gratuite RENDER. Pour surmonter les limitations de l'offre gratuite, certaines fonctionnalités ont été pré-calculées.

    Stratégie d'optimisation :

        Les graphiques SHAP ont été générés en amont à l'aide du script generate_shap_plots.py et les images résultantes ont été stockées sur le bucket AWS S3.

        Le rapport de dérive Evidently a aussi été généré en local et mis à disposition via un lien de téléchargement directement dans l'application.

