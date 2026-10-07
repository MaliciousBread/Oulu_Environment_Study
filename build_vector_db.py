import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(dotenv_path = "multi_agents/.env")

# Chemins basés sur ton architecture
DATA_FILE = "Projet_Data/rag/medical_guidelines.txt"
CHROMA_DIR = "Projet_Data/rag/chroma_db"

def build_database():
    print("1. Chargement du document normatif...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Le fichier {DATA_FILE} est introuvable. Crée-le d'abord.")
        
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    documents = loader.load()

    print("2. Découpage en chunks...")
    # Paramètres issus de ton notebook 06_RAG_Evaluation
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"   -> {len(docs)} fragments générés.")

    print("3. Vectorisation et persistance dans ChromaDB...")
    # Utilisation du modèle d'embedding défini dans ton orchestrateur
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="campus_norms"
    )
    
    print(f"Succès ! Base vectorielle sauvegardée dans : {CHROMA_DIR}")

if __name__ == "__main__":
    build_database()
