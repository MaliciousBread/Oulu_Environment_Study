import streamlit as st
import requests
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Smart Campus IA", layout="wide")
API_URL = "http://localhost:8000/evaluate"

st.title("Smart Campus - Interface Cognitive de Régulation")
st.markdown("---")

# --- BARRE LATÉRALE : PROFIL UTILISATEUR ---
with st.sidebar:
    st.header("Profil de l'Occupant")
    age = st.selectbox("Tranche d'âge", ["18-25", "26-40", "41-60", "60+"])
    condition = st.selectbox("Condition Chronique", ["Aucune", "Asthme", "Maladie Cardiovasculaire", "Allergies Sévères"])
    sensitivity = st.select_slider("Sensibilité Environnementale", options=["Faible", "Modérée", "Élevée"])

# --- SECTION 1 : MESURES ACTUELLES (SIMULATION) ---
st.header("Tableau de Bord Environnemental (T0)")
col1, col2, col3 = st.columns(3)
temp = col1.slider("Température (°C)", 15.0, 35.0, 25.0)
hum = col2.slider("Humidité (%)", 20.0, 80.0, 60.0)
co2 = col3.slider("CO2 (ppm)", 400, 2500, 800)

if st.button("Lancer le Diagnostic IA (Multi-Agents)", use_container_width=True):
    with st.spinner("Analyse cognitive et interrogation des normes en cours..."):
        
        # Préparation de la requête
        payload = {
            "current_metrics": {"temperature": temp, "humidity": hum, "co2": co2},
            "user_profile": {"age_range": age, "condition": condition, "environmental_sensitivity": sensitivity},
            "sensor_history": None # Simulation de l'historique pour le CNN
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # --- SECTION 2 : FORECASTING ---
            st.markdown("---")
            st.header("Prévisions à +30 Minutes (Temporal CNN)")
            pred = data["predictions_t30"]
            
            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("Température", f"{pred['temperature']} °C", f"{round(pred['temperature'] - temp, 1)} °C")
            p_col2.metric("Humidité", f"{pred['humidity']} %", f"{round(pred['humidity'] - hum, 1)} %")
            p_col3.metric("CO2", f"{pred['co2']} ppm", f"{round(pred['co2'] - co2, 0)} ppm")
            
            # --- SECTION 3 : ANALYSE DES AGENTS ---
            st.markdown("---")
            st.header("Diagnostic du Système Multi-Agents")
            
            risk_data = data["risk_assessment"]
            risk_level = risk_data["risk_level"]
            
            # Feu tricolore d'adéquation
            if risk_level == "Adapté":
                st.success(f"🟢 **Adéquation : {risk_level}**")
            elif risk_level == "À surveiller":
                st.warning(f"🟠 **Adéquation : {risk_level}**")
            else:
                st.error(f"🔴 **Adéquation : {risk_level}**")
                
            # Détails de l'analyse
            with st.expander("Voir le raisonnement détaillé des agents", expanded=True):
                st.markdown("**Agent Environnemental :**")
                # Gestion du format de texte retourné par Gemini
                env_text = data["analysis"]["environment"][0]["text"] if isinstance(data["analysis"]["environment"], list) else data["analysis"]["environment"]
                st.write(env_text)
                
                st.markdown("**Agent Santé :**")
                health_text = data["analysis"]["health"][0]["text"] if isinstance(data["analysis"]["health"], list) else data["analysis"]["health"]
                st.write(health_text)
                
                st.markdown("**Facteurs de Risque :**")
                for r in risk_data["reason"]:
                    st.write(f"- {r}")
            
            # --- SECTION 4 : RECOMMANDATIONS & RAG ---
            rec_data = data.get("recommendation", {})
            if rec_data.get("status") != "OK":
                st.markdown("---")
                st.header("Plan d'Action GTB (Agent Recommandation)")
                for rec in rec_data.get("recommendations", []):
                    st.info(f"{rec}")
                    
                st.caption("Sources normatives (RAG) ayant justifié cette décision :")
                for src in rec_data.get("sources", []):
                    st.caption(f"- {src}")
                    
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API : {e}. Assurez-vous qu'Uvicorn tourne sur le port 8000.")
