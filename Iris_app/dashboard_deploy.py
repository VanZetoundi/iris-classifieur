import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Configuration de la page
st.set_page_config(
    page_title="Classificateur Iris",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et introduction
st.title("🌸 Classificateur de fleurs Iris")

# Chargement du modèle et du scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('model/iris_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# Liste des espèces (hardcodée car très stable)
SPECIES = ['setosa', 'versicolor', 'virginica']

# Chargement des données Iris pour les visualisations
@st.cache_data
def load_iris_df():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df.columns = ['Sepal Length (cm)', 'Sepal Width (cm)', 
                  'Petal Length (cm)', 'Petal Width (cm)', 'species']
    return df

df = load_iris_df()

# SIDEBAR – Formulaire de saisie
with st.sidebar:
    st.header("Mesures de la fleur")
    st.markdown("Déplacez les curseurs pour simuler une nouvelle observation")

    sepal_length = st.slider("Longueur du sépale (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Largeur du sépale (cm)",   2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Longueur du pétale (cm)",  1.0, 7.0, 4.3, 0.1)
    petal_width  = st.slider("Largeur du pétale (cm)",   0.1, 2.5, 1.3, 0.1)

    predict_button = st.button("Prédire l’espèce", type="primary", use_container_width=True)

# PRÉDICTION (lorsque le bouton est cliqué)
if predict_button:
    # Préparation des données
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)

    # Prédiction
    pred_idx = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    predicted = SPECIES[pred_idx]
    probabilities = dict(zip(SPECIES, proba))

    # Affichage résultat
    st.success(f"**Espèce prédite : {predicted.upper()}**")
    
    st.subheader("Probabilités estimées")
    for sp, p in probabilities.items():
        st.write(f"**{sp.capitalize()}** : {p:.1%}")
        st.progress(p)

# VISUALISATIONS
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Nuage de points – Pétales")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df, 
        x='Petal Length (cm)', 
        y='Petal Width (cm)',
        hue='species', 
        palette='viridis',
        alpha=0.7,
        s=80,
        ax=ax1
    )
    ax1.scatter(petal_length, petal_width, c='red', s=300, marker='*',
                edgecolor='black', linewidth=1.5, label='Votre mesure')
    ax1.legend()
    ax1.set_title("Votre observation (étoile rouge) par rapport aux 150 iris")
    st.pyplot(fig1)

with col2:
    st.subheader("Distribution – Longueur des pétales")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.histplot(
        data=df, 
        x='Petal Length (cm)', 
        hue='species', 
        multiple='stack',
        palette='viridis',
        ax=ax2
    )
    st.pyplot(fig2)

# Pied de page
st.markdown("---")
st.caption("Modèle : KNN (k=5) | Données : Iris dataset | Réalisé par Van Zetoundi")