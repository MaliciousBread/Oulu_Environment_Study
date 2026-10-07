import os
from dotenv import load_dotenv
load_dotenv()
import json
from typing import List, Dict, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

# Initialisation du LLM local (Qwen 2.5 3B via Ollama)
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    base_url="http://localhost:11434"
)

# --- MODELES PYDANTIC (Garantissent le format JSON décrit en 4.11) ---

class RiskSchema(BaseModel):
    status: str = Field(description="Risk status: normal, warning, or critical")
    risk_score: float = Field(description="Score between 0.0 and 1.0")
    recommendations: list[str] = Field(description="List of action items")

structured_llm = llm.with_structured_output(RiskSchema)

class RiskAssessmentOutput(BaseModel):
    risk_level: Literal["Adapté", "À surveiller", "Potentiellement inadapté"] = Field(
        description="Niveau de risque global basé sur l'environnement et la santé."
    )
    reason: List[str] = Field(
        description="Liste des facteurs justifiant ce niveau de risque."
    )

class RecommendationOutput(BaseModel):
    status: str = Field(description="Statut de l'alerte (ex: warning, critical).")
    risk_score: float = Field(description="Score de risque normalisé entre 0 et 1.")
    factors: List[str] = Field(description="Facteurs environnementaux et médicaux déclencheurs.")
    recommendations: List[str] = Field(description="Actions GTB ou recommandations usager (ex: améliorer ventilation).")
    sources: List[str] = Field(description="Sources normatives issues du RAG (traçabilité).")

# --- ETAT DU GRAPHE (La mémoire partagée des agents) ---

class SmartCampusState(TypedDict):
    current_metrics: Dict[str, float]
    predicted_metrics: Dict[str, float]
    user_profile: Dict[str, str]
    
    env_analysis: str
    health_context: str
    rag_documents: List[str]
    
    risk_assessment: dict
    final_response: dict

def setup_contextual_rag():
    """
    Charge la base vectorielle ChromaDB avec le modèle d'embedding local dédié (Nomic).
    """
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    vectorstore = Chroma(
        persist_directory="Projet_Data/rag/chroma_db",
        embedding_function=embeddings,
        collection_name="campus_norms"
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = setup_contextual_rag()

def env_analysis_agent(state: SmartCampusState):
    """Agent Environnemental (4.3) : Interprète les deltas."""
    prompt = f"""
    Compare les mesures actuelles et les prédictions à t+30min :
    Actuel : {state['current_metrics']}
    Prédictions : {state['predicted_metrics']}
    Identifie les tendances critiques en 2 phrases maximum.
    """
    response = llm.invoke([SystemMessage(content="Tu es un expert en thermodynamique des bâtiments."), HumanMessage(content=prompt)])
    return {"env_analysis": response.content}

def health_context_agent(state: SmartCampusState):
    """Agent Santé (4.4) : Interprète les vulnérabilités de l'occupant."""
    prompt = f"""
    Analyse ce profil utilisateur : {state['user_profile']}.
    L'environnement intérieur prévu à t+30min est : {state['predicted_metrics']}.
    Quels sont les risques physiologiques précis liés EXCLUSIVEMENT à ces niveaux de température, humidité et CO2 pour cette pathologie ? 
    Ne mentionne PAS d'autres polluants externes (pollen, détergents, tabac).
    """
    response = llm.invoke([
        SystemMessage(content="Tu es un médecin du travail expert en qualité de l'air intérieur."), 
        HumanMessage(content=prompt)
    ])
    return {"health_context": response.content}

def rag_knowledge_agent(state: SmartCampusState):
    """Agent RAG (4.6) : Récupère les normes avec filtrage par maladie."""
    condition = state['user_profile']['condition'].lower()
    
    # Catch both French and English UI inputs, fallback to cross_cutting
    disease_filter = "cross_cutting" 
    if "asthme" in condition or "asthma" in condition:
        disease_filter = "asthma"
    elif "cardio" in condition:
        disease_filter = "cardiovascular"
    elif "diabèt" in condition or "diabet" in condition:
        disease_filter = "diabetes"
    elif "peau" in condition or "skin" in condition or "allergie" in condition:
        disease_filter = "skin"

    query = f"Risques environnementaux (température, humidité, CO2) pour {condition}"
    
    docs = retriever.invoke(query, filter={"disease": disease_filter})
    
    formatted_docs = []
    for doc in docs:
        # Extracts the exact title and journal from your metadata.json
        source_cite = f"{doc.metadata.get('title')} ({doc.metadata.get('journal')}, {doc.metadata.get('year')})"
        formatted_docs.append(f"[Source: {source_cite}]\n{doc.page_content}")
        
    return {"rag_documents": formatted_docs}

def risk_assessment_agent(state: SmartCampusState):
    """Agent d'Évaluation des Risques (4.7) : Retourne un JSON structuré."""
    structured_llm_risk = llm.with_structured_output(RiskAssessmentOutput)
    
    prompt = f"""
    Évalue le risque.
    Environnement : {state['env_analysis']}
    Santé : {state['health_context']}
    Normes applicables : {state['rag_documents']}
    
    RÈGLE STRICTE DE CLASSIFICATION : 
    - Si l'environnement présente le moindre danger, favorise les moisissures, ou dépasse les seuils de confort (ex: 30°C, 78% HR, 1200 ppm CO2), tu DOIS IMPÉRATIVEMENT choisir "Potentiellement inadapté".
    - Utilise "À surveiller" pour des dépassements légers.
    - Utilise "Adapté" UNIQUEMENT si l'environnement est parfaitement sain.
    """
    risk_data = structured_llm_risk.invoke([
        SystemMessage(content="Tu es un auditeur de risques clinique intraitable. Tes choix de classification sont absolus."), 
        HumanMessage(content=prompt)
    ])
    
    return {"risk_assessment": risk_data.model_dump()}

def recommendation_agent(state: SmartCampusState):
    """Agent de Recommandation (4.8) : Agit uniquement si nécessaire."""
    structured_llm_rec = llm.with_structured_output(RecommendationOutput)
    
    prompt = f"""
    Génère un plan d'action JSON.
    Niveau de risque détecté : {state['risk_assessment']['risk_level']}
    Facteurs : {state['risk_assessment']['reason']}
    Documents extraits : {state['rag_documents']}
    
    Règle stricte: Dans le champ 'sources', tu DOIS recopier exactement les titres et journaux indiqués entre [Source: ...] dans les documents fournis.
    """ 
    rec_data = structured_llm_rec.invoke([
        SystemMessage(content="Tu es le contrôleur GTB. Propose des actions correctives (ventilation, climatisation)."), 
        HumanMessage(content=prompt)
    ])
    
    return {"final_response": rec_data.model_dump()}

# --- ROUTAGE CONDITIONNEL (Section 4.7.2 Trigger mechanism) ---
def should_recommend(state: SmartCampusState):
    """Décide si l'Agent de Recommandation doit être déclenché."""
    risk_level = state["risk_assessment"]["risk_level"]
    if risk_level == "Adapté":
        return "end"
    return "recommend"

# --- CONSTRUCTION DU GRAPHE ---
workflow = StateGraph(SmartCampusState)

# Ajout des noeuds
workflow.add_node("env_agent", env_analysis_agent)
workflow.add_node("health_agent", health_context_agent)
workflow.add_node("rag_agent", rag_knowledge_agent)
workflow.add_node("risk_agent", risk_assessment_agent)
workflow.add_node("rec_agent", recommendation_agent)

# Définition des arêtes (Flux d'exécution)
workflow.set_entry_point("env_agent")
workflow.add_edge("env_agent", "health_agent")
workflow.add_edge("health_agent", "rag_agent")
workflow.add_edge("rag_agent", "risk_agent")

# Branchement conditionnel
workflow.add_conditional_edges(
    "risk_agent",
    should_recommend,
    {
        "recommend": "rec_agent",
        "end": END
    }
)
workflow.add_edge("rec_agent", END)

# Compilation du graphe
smart_campus_app = workflow.compile()

if __name__ == "__main__":
    test_state = {
        "current_metrics": {"temperature": 26, "humidity": 65, "co2": 800},
        "predicted_metrics": {"temperature": 30, "humidity": 78, "co2": 1200},
        "user_profile": {
            "age_range": "50-60",
            "condition": "asthma",
            "environmental_sensitivity": "high"
        }
    }

    print("=== DÉMARRAGE DU PIPELINE COGNITIF LOCAL (QWEN + NOMIC) ===\n")
    
    final_state = smart_campus_app.invoke(test_state)
    
    print("1. [Agent Environnemental] Synthèse :")
    print(final_state["env_analysis"], "\n")
    
    print("2. [Agent Santé] Profil évalué :")
    print(final_state["health_context"], "\n")
    
    print("3. [Agent RAG] Sources documentaires mobilisées :")
    for doc in final_state["rag_documents"]:
        print(f" - {doc}")
    print("\n")
    
    print("4. [Agent Risque] Évaluation structurée :")
    print(json.dumps(final_state["risk_assessment"], indent=2, ensure_ascii=False), "\n")
    
    if "final_response" in final_state:
        print("5. [Agent Recommandation] Ordre JSON pour la GTB :")
        print(json.dumps(final_state["final_response"], indent=2, ensure_ascii=False))
    else:
        print("5. [Système] Aucun risque majeur, aucune action GTB requise.")
