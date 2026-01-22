# train_and_save_model.py
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Chargement des données
iris = load_iris()
X = iris.data
y = iris.target

# Division
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Sauvegarde
with open('model/iris_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Sauvegarde des noms d'espèces (pour l'affichage)
species = iris.target_names
with open('model/iris_species.txt', 'w') as f:
    f.write('\n'.join(species))

print("Modèle et scaler sauvegardés avec succès.")
print(f"Exactitude sur test : {knn.score(X_test_scaled, y_test):.3f}")