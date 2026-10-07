import torch
import torch.nn as nn
import numpy as np
import joblib
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DÉFINITION DU MODÈLE ---
class TemporalCNN(nn.Module):
    def __init__(self, n_features, hidden=128, dropout=0.2):
        super().__init__()
        n_outputs = 3  # Température, Humidité, CO2
        channels = [n_features, hidden, hidden, hidden]
        dilations = [1, 2, 4]
        layers = []
        for i, d in enumerate(dilations):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(channels[i + 1]), nn.GELU(), nn.Dropout(dropout)
            ]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_outputs)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.head(self.conv(x))

# --- 2. CONFIGURATION ET CHARGEMENT ---
N_FEATURES = 26 

model = TemporalCNN(n_features=N_FEATURES, hidden=128, dropout=0.2).to(DEVICE)

# Chargement des poids
MODEL_PATH = "Projet_Data/models_dl/temporal_cnn.pt"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
else:
    print(f"⚠️ Attention : Fichier {MODEL_PATH} introuvable.")

# Chargement des Scalers
X_SCALER_PATH = "Projet_Data/models_dl/x_scaler.joblib"
Y_SCALER_PATH = "Projet_Data/models_dl/y_scaler.joblib"

x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# --- 3. FONCTION D'INFÉRENCE ---
def predict_next_30m(recent_data_array):
    """
    Ingère les 24 derniers relevés (lookback) et retourne la prédiction à +30min.
    recent_data_array : liste ou numpy array de shape (24, 26).
    """
    # Étape A : Normalisation
    recent_data_scaled = x_scaler.transform(recent_data_array).astype(np.float32)
    
    # Étape B : Inférence
    x_tensor = torch.tensor(recent_data_scaled).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()
        
    # Étape C : Retour à l'échelle d'origine
    pred_future = y_scaler.inverse_transform(pred_scaled)[0]
    
    return {
        "temperature": round(float(pred_future[0]), 1),
        "humidity": round(float(pred_future[1]), 1),
        "co2": round(float(pred_future[2]), 0)
    }

if __name__ == "__main__":
    print(f"Démarrage du test d'inférence (Device: {DEVICE})...")
    # Test avec de fausses données de capteurs (24 séquences, 26 features)
    dummy_data = np.random.rand(24, N_FEATURES)
    result = predict_next_30m(dummy_data)
    print("Prédiction (Temp, Hum, CO2) :", result)
