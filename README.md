# FraudShield AI

Détection de fraude bancaire par IA explicable — projet de stage Data Science pour **LUMERIA BANK** (entreprise fictive).

## Stack technique
Python · Pandas · Scikit-Learn · XGBoost · LightGBM · SHAP · MLflow · Streamlit · Power BI · Docker

## Méthodologie
CRISP-DM

## Dataset
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle)
- `train_transaction.csv` — 590 540 lignes × 394 colonnes
- `train_identity.csv`

## Structure du projet
```
data/
  raw/         # données brutes, non modifiées
  processed/   # données nettoyées / features
notebooks/     # exploration (EDA, prototypage)
src/           # code source réutilisable (pipeline, modèles, explicabilité)
models/        # modèles entraînés sérialisés
dashboard/     # Power BI / reporting
app/           # application Streamlit
docs/          # documentation de cadrage
tests/         # tests unitaires
```

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Preparation des donnees (genere data/processed/train_clean.parquet + category_mappings.pkl)
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_preparation.ipynb

# 2. Entrainement du modele baseline (genere models/lightgbm_baseline.pkl)
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb

# 3. Lancer l'application de demo (page principale + page Dashboard)
streamlit run app/streamlit_app.py
```

## Tests

```bash
pytest tests/
```
