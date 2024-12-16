#%%

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Charger les données à partir du fichier CSV
file_path = 'data.csv'  # Remplacez par le chemin de votre fichier CSV
data = pd.read_csv(file_path)

# Afficher un aperçu des données
print(data.head())

# Vérifier les valeurs manquantes
print(data.isnull().sum())

# Remplacer les valeurs manquantes par la moyenne de chaque colonne (si nécessaire)
data.fillna(data.mean(), inplace=True)

# Sélectionner les caractéristiques pour le clustering
features = data.drop(columns=['Outcome'])

# Normaliser les données
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Déterminer le nombre optimal de clusters en utilisant la méthode du coude
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Tracer la courbe d'inertie
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.show()

# Appliquer K-Means avec le nombre optimal de clusters (par exemple, k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Ajouter les clusters aux données d'origine
data['Cluster'] = clusters

# Réduire les dimensions pour la visualisation avec PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Ajouter les composants principaux aux données
data['PCA1'] = pca_features[:, 0]
data['PCA2'] = pca_features[:, 1]

# Visualiser les clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='viridis')
plt.title('Visualisation des clusters K-Means')
plt.show()
# %%
