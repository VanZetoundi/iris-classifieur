# dashboard.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(page_title="Iris Flower Classifier", layout="wide")

st.title("🌸 Classificateur de fleurs Iris")
st.markdown("Entrez les mesures de la fleur et obtenez la prédiction en temps réel via l'API Flask.")

# Chargement des données pour visualisation
@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

df = load_iris_data()

# Formulaire de saisie
with st.sidebar:
    st.header("Mesures de la fleur")
    
    sepal_length = st.slider("Longueur du sépale (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Largeur du sépale (cm)",  2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Longueur du pétale (cm)", 1.0, 7.0, 4.3, 0.1)
    petal_width  = st.slider("Largeur du pétale (cm)",   0.1, 2.5, 1.3, 0.1)
    
    if st.button("Prédire", type="primary"):
        payload = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Espèce prédite : **{result['predicted_species']}**")
                
                probs = result["probabilities"]
                st.subheader("Probabilités")
                for sp, p in probs.items():
                    st.progress(p)
                    st.write(f"{sp}: {p:.1%}")
                
        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")

# Visualisations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Nuage de points interactif")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                    hue='species', palette='deep', ax=ax)
    ax.scatter(petal_length, petal_width, c='red', s=200, marker='*',
               label='Votre mesure')
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Distribution des longueurs de pétales")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x='petal length (cm)', hue='species', multiple='stack', ax=ax2)
    st.pyplot(fig2)

st.markdown("---")
st.caption("API Flask sur http://127.0.0.1:5000 | Modèle : KNN | Données : Iris dataset")