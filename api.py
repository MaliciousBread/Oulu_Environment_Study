import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# Importation de ton modèle DL et de tes agents
from forecasting.inference import predict_next_30m
from multi_agents.campus_orchestrator import smart_campus_app

app = FastAPI(
    title="Smart Campus API",
    description="API d'évaluation des risques thermodynamiques et sanitaires",
    version="1.0"
)

PROCESSED_CSV = "Projet_Data/processed/iot_preprocessed.csv"
if os.path.exists(PROCESSED_CSV):
    df_global = pd.read_csv(PROCESSED_CSV)
    # Exclure les colonnes non numériques/cibles pour garder les 26 features
    targets = ['target_temp', 'target_humidity', 'target_co2']
    excluded_columns = ["time", "deveui"] + targets
    FEATURE_COLS = [col for col in df_global.columns if col not in excluded_columns]
else:
    df_global = None

# --- SCHÉMAS D'ENTRÉE (Pydantic) ---
class EvaluationRequest(BaseModel):
    current_metrics: Dict[str, float]
    user_profile: Dict[str, str]
    sensor_history: Optional[List[List[float]]] = None  # Matrice 24x26 pour le Temporal CNN

# --- ROUTES ---
@app.post("/evaluate")
async def evaluate_environment(request: EvaluationRequest):
    try:
        # 1. Traitement des données historiques
        if request.sensor_history is None or len(request.sensor_history) != 24:
            if df_global is not None and len(df_global) >= 24:
                history_array = df_global[FEATURE_COLS].head(24).values.astype(np.float32)
            else:
                raise HTTPException(status_code=400, detail="Dataset introuvable.")
        else:
            history_array = np.array(request.sensor_history, dtype=np.float32)

        if history_array.shape != (24, 26):
            # Si l'historique fourni ne fait pas 24x26, on bascule par défaut sur le dataset réel
            if df_global is not None:
                history_array = df_global[FEATURE_COLS].head(24).values.astype(np.float32)
            else:
                raise ValueError(f"Dimensions invalides: {history_array.shape}") 
            
        # 2. Inférence Deep Learning (CNN)
        predicted_metrics = predict_next_30m(history_array)

        # 3. Préparation de l'état partagé (StateGraph)
        state = {
            "current_metrics": request.current_metrics,
            "predicted_metrics": predicted_metrics,
            "user_profile": request.user_profile
        }
        
        # 4. Orchestration Multi-Agents (LangGraph)
        final_state = smart_campus_app.invoke(state)
        
        # 5. Restitution structurée
        return {
            "predictions_t30": predicted_metrics,
            "analysis": {
                "environment": final_state.get("env_analysis"),
                "health": final_state.get("health_context")
            },
            "risk_assessment": final_state.get("risk_assessment"),
            "recommendation": final_state.get("final_response", {"status": "OK", "message": "Environnement adapté."})
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
