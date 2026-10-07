import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. Chargement des données (identique au notebook)
df = pd.read_csv("Projet_Data/processed/iot_preprocessed.csv")
df = df.sort_values("time").reset_index(drop=True)

targets = ['target_temp', 'target_humidity', 'target_co2']
excluded_columns = ["time", "deveui"] + targets
features = [col for col in df.columns if col not in excluded_columns]

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()

# 2. Création et ajustement des scalers sur le jeu d'entraînement
x_scaler = StandardScaler()
x_scaler.fit(train_df[features].values.astype(np.float32))

y_scaler = StandardScaler()
y_scaler.fit(train_df[targets].values.astype(np.float32))

# 3. Sauvegarde
os.makedirs("Projet_Data/models_dl", exist_ok=True)
joblib.dump(x_scaler, "Projet_Data/models_dl/x_scaler.joblib")
joblib.dump(y_scaler, "Projet_Data/models_dl/y_scaler.joblib")

print(f"Scalers sauvegardés ! Nombre de features (N_FEATURES) : {len(features)}")
