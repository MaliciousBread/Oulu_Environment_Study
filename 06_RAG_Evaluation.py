import json
import os
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

print("Chargement de la base de connaissances scientifique...")
# Load the rich metadata file
with open("Projet_Data/knowledge_base/metadata.json", "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)

documents = []
for item in knowledge_data:
    # Use the abstract as the core text to embed
    text_content = item.get("abstract", "")
    if not text_content:
        continue
        
    # Attach tracking metadata for the agents
    doc_metadata = {
        "title": item.get("title", "Titre inconnu"),
        "journal": item.get("journal", "Source inconnue"),
        "year": item.get("year", "N/A"),
        "disease": item.get("disease", "general"),
        "factor": item.get("factor", "general")
    }
    documents.append(Document(page_content=text_content, metadata=doc_metadata))

print(f"{len(documents)} documents prêts pour l'embedding.")

# Use Nomic directly via Ollama
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

os.makedirs("Projet_Data/rag/chroma_db", exist_ok=True)
vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embeddings,
    persist_directory="Projet_Data/rag/chroma_db",
    collection_name="campus_norms"
)

print("Base vectorielle construite avec succès avec Nomic et ChromaDB !")

# %%

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Données issues de l'analyse des fichiers JSON et du README
diseases = {'Asthme': 70, 'Diabète': 58, 'Cardiovasculaire': 44, 'Peau': 41, 'Transversal': 35}
factors = {'Température / Chaleur': 103, 'Air Intérieur & CO2': 61, 'Humidité': 59, 'Général': 25}
sources = {'Articles Scientifiques': 230, 'Directives Officielles': 18}

# Configuration d'un style académique (minimaliste, serif)
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams['font.family'] = 'serif'

fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

# Palette en niveaux de gris progressifs
grey_palette = sns.color_palette("Greys_r", n_colors=6)

# 1. Répartition par maladie
bars1 = sns.barplot(x=list(diseases.keys()), y=list(diseases.values()), ax=axes[0], palette=grey_palette[:5], edgecolor='black', linewidth=1.2)
axes[0].set_title('Répartition par Pathologie', fontweight='bold', pad=15)
axes[0].set_ylabel('Nombre de documents', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
# Ajout des étiquettes de données sur les barres
for p in bars1.patches:
    axes[0].annotate(format(int(p.get_height()), 'd'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 8), textcoords='offset points')

# 2. Répartition par facteur environnemental
bars2 = sns.barplot(x=list(factors.keys()), y=list(factors.values()), ax=axes[1], palette=grey_palette[:4], edgecolor='black', linewidth=1.2)
axes[1].set_title('Répartition par Facteur Environnemental', fontweight='bold', pad=15)
axes[1].set_ylabel('Nombre de documents', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
# Ajout des étiquettes de données sur les barres
for p in bars2.patches:
    axes[1].annotate(format(int(p.get_height()), 'd'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 8), textcoords='offset points')

# 3. Répartition par source
axes[2].pie(sources.values(), labels=sources.keys(), autopct='%1.1f%%', 
            colors=['#777777', '#E0E0E0'], startangle=140, explode=[0, 0.1],
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.2},
            textprops={'fontweight': 'bold'})
axes[2].set_title('Typologie des Sources', fontweight='bold', pad=15)

# Nettoyage des bordures (esthétique professionnelle)
sns.despine(ax=axes[0])
sns.despine(ax=axes[1])

plt.suptitle("Figure 5.1 : Analyse de la base de connaissances du RAG Contextuel", fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()

# Sauvegarde de l'image prête pour le document
os.makedirs("Projet_Data/images", exist_ok=True)
plt.savefig("Projet_Data/images/figure_5_1.png", bbox_inches='tight', transparent=False)
print("Graphique sauvegardé sous : Projet_Data/images/figure_5_1.png")
plt.show()

# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import numpy as np

print("Extraction des vecteurs depuis ChromaDB...")

# 1. Connexion à la base de données existante
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vectorstore = Chroma(
    persist_directory="Projet_Data/rag/chroma_db",
    embedding_function=embeddings,
    collection_name="campus_norms"
)

# 2. Récupération complète des données (Vecteurs + Métadonnées)
# Attention: get() nécessite include=['embeddings', 'metadatas']
db_data = vectorstore.get(include=['embeddings', 'metadatas'])
vectors = np.array(db_data['embeddings'])
metadatas = db_data['metadatas']

print(f"Nombre de documents extraits : {vectors.shape[0]}")
print(f"Dimensionnalité des vecteurs : {vectors.shape[1]} (Attendu: 768 pour Nomic)")

# 3. Réduction de dimensionnalité (de 768 vers 2 dimensions avec t-SNE)
print("Réduction de dimensionnalité via t-SNE en cours (cela peut prendre quelques secondes)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# 4. Préparation du DataFrame pour l'affichage
df_vis = pd.DataFrame({
    'Dim 1': vectors_2d[:, 0],
    'Dim 2': vectors_2d[:, 1],
    'Maladie': [meta.get('disease', 'Inconnu').capitalize() for meta in metadatas],
    'Facteur': [meta.get('factor', 'Inconnu').capitalize() for meta in metadatas]
})

# 5. Visualisation Professionnelle (Niveaux de gris et marqueurs)
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams['font.family'] = 'serif'

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Graphique A : Clustering par Maladie
sns.scatterplot(data=df_vis, x='Dim 1', y='Dim 2', hue='Maladie', style='Maladie', 
                ax=axes[0], s=100, palette="Greys_r", edgecolor='black', alpha=0.8)
axes[0].set_title('Topologie Sémantique par Pathologie', fontweight='bold', pad=15)
sns.despine(ax=axes[0])

# Graphique B : Clustering par Facteur Environnemental
sns.scatterplot(data=df_vis, x='Dim 1', y='Dim 2', hue='Facteur', style='Facteur', 
                ax=axes[1], s=100, palette="Greys_r", edgecolor='black', alpha=0.8)
axes[1].set_title('Topologie Sémantique par Facteur Climatique', fontweight='bold', pad=15)
sns.despine(ax=axes[1])

plt.suptitle("Figure 5.2 : Projection t-SNE de l'espace latent des documents normatifs (Embeddings à 768 dimensions)", 
             fontsize=14, fontweight='bold', y=1.05)

plt.tight_layout()
plt.savefig("Projet_Data/images/figure_5_2_tsne.png", bbox_inches='tight')
print("Graphique sauvegardé sous : Projet_Data/images/figure_5_2_tsne.png")
plt.show()
