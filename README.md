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
