import pandas as pd
import numpy as np
import os
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
import gc
import time
import contextlib
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from io import BytesIO  # Ajout de cette ligne pour corriger l'erreur BytesIO


# --- Configuration ---
INPUT_DIR = r"C:\Projet-7-OC\input"
OUTPUT_DIR = r"C:\Projet-7-OC\processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration MLflow
MLFLOW_URI = "http://16.170.254.32:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Home_Credit_Feature_Engineering_Local")

# NOTE: CHEMIN CORRIGÉ
DATA_PATH = r"C:\Projet-7-OC\input"

# --- FONCTIONS D'INGÉNIERIE DES CARACTÉRISTIQUES (Version réelle) ---

@contextlib.contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f}s')

def one_hot_encoder(df, nan_as_category = True):
    """
    Applique l'encodage one-hot aux colonnes de type 'object' d'un DataFrame.
    Gère également les NaN comme une catégorie si nan_as_category est True.
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def application_train_test(num_rows = None, nan_as_category = False):
    """
    Prétraite les fichiers application_train.csv et application_test.csv.
    """
    with timer("Application Train/Test Preprocessing"):
        df = pd.read_csv(os.path.join(DATA_PATH, 'application_train.csv'), nrows= num_rows)
        test_df = pd.read_csv(os.path.join(DATA_PATH, 'application_test.csv'), nrows= num_rows)
        print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))

        df = pd.concat([df, test_df], ignore_index=True)
        df = df.reset_index(drop=True)
        del test_df
        gc.collect()

        df = df[df['CODE_GENDER'] != 'XNA']

        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])
        df, cat_cols = one_hot_encoder(df, nan_as_category)

        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

        df['INCOME_CREDIT_PERC'] = np.where(df['AMT_CREDIT'] == 0, np.nan, df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'])
        df['INCOME_PER_PERSON'] = np.where(df['CNT_FAM_MEMBERS'] == 0, np.nan, df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'])
        df['ANNUITY_INCOME_PERC'] = np.where(df['AMT_INCOME_TOTAL'] == 0, np.nan, df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'])
        df['PAYMENT_RATE'] = np.where(df['AMT_CREDIT'] == 0, np.nan, df['AMT_ANNUITY'] / df['AMT_CREDIT'])

        return df

def bureau_and_balance(num_rows = None, nan_as_category = True):
    """
    Prétraite les fichiers bureau.csv et bureau_balance.csv.
    """
    with timer("Bureau and Bureau Balance Preprocessing"):
        bureau = pd.read_csv(os.path.join(DATA_PATH, 'bureau.csv'), nrows = num_rows)
        bb = pd.read_csv(os.path.join(DATA_PATH, 'bureau_balance.csv'), nrows = num_rows)

        bb, bb_cat = one_hot_encoder(bb, nan_as_category)
        bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}

        for col in bb_cat:
            bb_aggregations[col] = ['mean']

        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

        new_bb_agg_cols = {}
        for col in bb_agg.columns:
            original_bb_cat_name = col.replace('_MEAN', '')
            if original_bb_cat_name in bb_cat:
                new_bb_agg_cols[col] = col + '_BB'
            elif any(s in col for s in [cat.replace('_MEAN', '') for cat in bb_cat]):
                    if '_BB_MEAN' not in col:
                        new_bb_agg_cols[col] = col.replace('_MEAN', '_BB_MEAN') if '_MEAN' in col else col + '_BB_MEAN'

        bb_agg = bb_agg.rename(columns=new_bb_agg_cols)

        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
        del bb, bb_agg
        gc.collect()

        num_aggregations = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }
        cat_aggregations = {}
        for cat in bureau_cat: cat_aggregations[cat] = ['mean']

        for col in bureau.columns:
            if '_BB_MEAN' in col:
                cat_aggregations[col] = ['mean']

        final_aggregations = {**num_aggregations, **cat_aggregations}

        bureau_agg = bureau.groupby('SK_ID_CURR').agg(final_aggregations)
        bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
        del active, active_agg
        gc.collect()

        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
        del closed, closed_agg, bureau
        gc.collect()
        return bureau_agg

def previous_applications(num_rows = None, nan_as_category = True):
    """
    Prétraite le fichier previous_application.csv.
    """
    with timer("Previous Applications Preprocessing"):
        prev = pd.read_csv(os.path.join(DATA_PATH, 'previous_application.csv'), nrows = num_rows)
        prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)

        prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
        prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
        prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
        prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
        prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

        prev['APP_CREDIT_PERC'] = np.where(prev['AMT_CREDIT'] == 0, np.nan, prev['AMT_APPLICATION'] / prev['AMT_CREDIT'])

        num_aggregations = {
            'AMT_ANNUITY': ['min', 'max', 'mean'],
            'AMT_APPLICATION': ['min', 'max', 'mean'],
            'AMT_CREDIT': ['min', 'max', 'mean'],
            'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']

        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        del refused, refused_agg, approved, approved_agg, prev
        gc.collect()
        return prev_agg

def pos_cash(num_rows = None, nan_as_category = True):
    """
    Prétraite le fichier POS_CASH_balance.csv.
    """
    with timer("POS-CASH Balance Preprocessing"):
        pos = pd.read_csv(os.path.join(DATA_PATH, 'POS_CASH_balance.csv'), nrows = num_rows)
        pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)

        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']

        pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

        pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
        del pos
        gc.collect()
        return pos_agg

def installments_payments(num_rows = None, nan_as_category = True):
    """
    Prétraite le fichier installments_payments.csv.
    """
    with timer("Installments Payments Preprocessing"):
        ins = pd.read_csv(os.path.join(DATA_PATH, 'installments_payments.csv'), nrows = num_rows)
        ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)

        ins['PAYMENT_PERC'] = np.where(ins['AMT_INSTALMENT'] == 0, np.nan, ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT'])
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
            'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
        del ins
        gc.collect()
        return ins_agg

def credit_card_balance(num_rows = None, nan_as_category = True):
    """
    Prétraite le fichier credit_card_balance.csv.
    """
    with timer("Credit Card Balance Preprocessing"):
        cc = pd.read_csv(os.path.join(DATA_PATH, 'credit_card_balance.csv'), nrows = num_rows)
        cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

        cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)

        num_aggregations = {
            'MONTHS_BALANCE': ['min', 'max', 'mean', 'sum', 'var'],
            'AMT_BALANCE': ['min', 'max', 'mean', 'sum', 'var'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'sum'],
            'AMT_RECIVABLE': ['min', 'max', 'mean', 'sum'],
            'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
            'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
            'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
            'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
            'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'sum'],
            'SK_DPD': ['min', 'max', 'mean', 'sum'],
            'SK_DPD_DEF': ['min', 'max', 'mean', 'sum']
        }
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean', 'sum']

        cc_agg = cc.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        del cc
        gc.collect()
        return cc_agg

def feature_engineering_pipeline(num_rows=None):
    """Orchestre le pipeline de feature engineering et retourne le dataframe final."""
    with timer("Feature Engineering Pipeline"):
        df = application_train_test(num_rows)
        
        with timer("Process bureau and bureau_balance"):
            bureau = bureau_and_balance(num_rows)
            df = df.merge(bureau.reset_index(), on='SK_ID_CURR', how='left')
            del bureau
            gc.collect()

        with timer("Process previous_applications"):
            prev = previous_applications(num_rows)
            df = df.merge(prev.reset_index(), on='SK_ID_CURR', how='left')
            del prev
            gc.collect()

        with timer("Process POS-CASH balance"):
            pos = pos_cash(num_rows)
            df = df.merge(pos.reset_index(), on='SK_ID_CURR', how='left')
            del pos
            gc.collect()

        with timer("Process installments payments"):
            ins = installments_payments(num_rows)
            df = df.merge(ins.reset_index(), on='SK_ID_CURR', how='left')
            del ins
            gc.collect()

        with timer("Process credit card balance"):
            cc = credit_card_balance(num_rows)
            df = df.merge(cc.reset_index(), on='SK_ID_CURR', how='left')
            del cc
            gc.collect()
            
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Correction pour gérer les colonnes dupliquées après le merge
    df = df.loc[:,~df.columns.duplicated()]
    
    return df

def find_optimal_fbeta_threshold(y_true, y_pred_proba, beta):
    """
    Trouve le seuil de classification optimal qui maximise le F-beta score.
    """
    thresholds = np.linspace(0, 1, 100)
    fbeta_scores = [fbeta_score(y_true, (y_pred_proba >= t).astype(int), beta=beta, zero_division=0) for t in thresholds]
    optimal_idx = np.argmax(fbeta_scores)
    return thresholds[optimal_idx]

def create_and_tune_model(X_train, y_train):
    """
    Crée un pipeline de prétraitement et de modélisation, puis utilise GridSearchCV
    pour trouver les meilleurs paramètres pour la régression logistique en optimisant le F-beta score.
    """
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    log_reg = LogisticRegression(solver='liblinear', n_jobs=-1, random_state=42, max_iter=1000, class_weight='balanced')
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', log_reg)
    ])
    
    param_grid = {
        'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1],
    }

    BETA_FOR_OPTIMAL_THRESHOLD = np.sqrt(10)
    fbeta_scorer = make_scorer(fbeta_score, beta=BETA_FOR_OPTIMAL_THRESHOLD, zero_division=0)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=fbeta_scorer,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=1,
        n_jobs=-1
    )

    print("Démarrage de la recherche sur grille...")
    grid_search.fit(X_train, y_train)

    print("\n--- Résultats de GridSearchCV ---")
    print(f"Meilleurs hyperparamètres: {grid_search.best_params_}")
    print(f"Meilleur score F-beta sur CV: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def save_artifacts(X_train, X_test, y_train, y_test, pipeline):
    """
    Sauvegarde les jeux de données d'entraînement/test et les métadonnées localement.
    """
    X_train.to_parquet(os.path.join(OUTPUT_DIR, "X_train.parquet"))
    X_test.to_parquet(os.path.join(OUTPUT_DIR, "X_test.parquet"))
    y_train.to_frame().to_parquet(os.path.join(OUTPUT_DIR, "y_train.parquet"))
    y_test.to_frame().to_parquet(os.path.join(OUTPUT_DIR, "y_test.parquet"))
    
    # --- AJOUT DE CETTE LIGNE POUR SAUVEGARDER LA LISTE DES NOMS D'ORIGINE ---
    original_feature_names = X_train.columns.tolist()
    
    try:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        feature_names = feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names)
    except Exception as e:
        print(f"Warning: Impossible de récupérer les feature_names - {str(e)}")
        feature_names = X_train.columns.tolist()
    
    metadata = {
        'feature_names': feature_names,
        'original_feature_names': original_feature_names, # Sauvegarde de la liste des noms d'origine
        'date_processed': datetime.now().isoformat(),
        'data_shape': {
            'rows': int(X_train.shape[0]),
            'columns': int(X_train.shape[1])
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    
    print(f"Artefacts sauvegardés dans {OUTPUT_DIR}")

def main():
    """Fonction principale pour exécuter le pipeline complet."""
    
    # --- AJOUT DE CETTE LIGNE POUR GÉRER L'ÉCHANTILLONNAGE ---
    NUM_ROWS_TO_PROCESS = 10000 
    
    with mlflow.start_run(run_name="Local_Feature_Engineering"):
        try:
            print(f"🚀 Démarrage du pipeline en mode échantillonnage avec {NUM_ROWS_TO_PROCESS} lignes...")
            
            df_full = feature_engineering_pipeline(num_rows=NUM_ROWS_TO_PROCESS)
            
            train_df_full = df_full[df_full['TARGET'].notnull()].copy()
            test_df_prediction = df_full[df_full['TARGET'].isnull()].copy()

            if train_df_full.empty:
                raise ValueError("Le DataFrame d'entraînement est vide après la séparation TARGET.")

            X_train = train_df_full.drop(columns=['TARGET', 'SK_ID_CURR'])
            y_train = train_df_full['TARGET']
            X_test = test_df_prediction.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
            y_test = pd.Series([np.nan] * len(X_test), index=X_test.index, name='TARGET')

            pipeline = create_and_tune_model(X_train, y_train)
            
            mlflow.log_params({
                'total_features': X_train.shape[1],
                'model': 'LogisticRegression (GridSearchCV Fbeta)',
                'imputer_numeric': 'median',
                'scaler_numeric': 'MinMaxScaler',
                'gridsearch_best_params': pipeline.named_steps['classifier'].get_params(),
                'sample_size_rows': NUM_ROWS_TO_PROCESS
            })
            
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=X_train.iloc[[0]]
            )
            
            save_artifacts(X_train, X_test, y_train, y_test, pipeline)
            
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, "metadata.json"))
            
            print("\n✅ Prétraitement et entraînement terminés avec succès!")
            print(f"✅ Modèle enregistré sous l'artefact 'model' dans MLflow.")
            print(f"🔗 Accès au run MLflow: {MLFLOW_URI}")
            
        except Exception as e:
            print(f"\n❌ Erreur lors du traitement: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

if __name__ == "__main__":
    main()