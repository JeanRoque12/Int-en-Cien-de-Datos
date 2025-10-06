import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#Exportacion de datos
df_stats = pd.read_csv("tb_lobby_stats_player.csv")

#Preprocessing
df_stats["KillsPerRound"] = df_stats.apply(lambda row: row.qtKill / row.qtRoundsPlayed, axis = 1)
df_stats["DeathsPerRound"] = df_stats.apply(lambda row: row.qtDeath / row.qtRoundsPlayed, axis = 1)
df_stats["AvgDmgPerRound"] = df_stats.apply(lambda row: row.vlDamage / row.qtRoundsPlayed, axis = 1)
df_stats["HS%"] = np.where(
    df_stats["qtKill"] > 0,
    df_stats["qtHs"] / df_stats["qtKill"],
    0  # valor por defecto cuando qtKill = 0
)
df_stats["Accuracy"] = np.where(
    df_stats["qtShots"] > 0,
    df_stats["qtHits"] / df_stats["qtShots"],
    0
)

columnas_a_remover = ["idLobbyGame", "idPlayer", "idRoom", "descMapName", "flWinner", "vlLevel", "qtHitHeadshot", "qtHitChest", "qtHitStomach", "qtHitLeftAtm", "qtHitRightArm", "qtHitLeftLeg", "qtHitRightLeg", "dtCreatedAt"]
df_stats = df_stats.drop(columns=columnas_a_remover)

#Normalizar datos datos
features = ["KillsPerRound","DeathsPerRound","AvgDmgPerRound","HS%","Accuracy","qtClutchWon","qtFirstKill","qtFlashAssist"]
df_stats = df_stats.dropna(subset=features)
X = df_stats[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

separador = "\n\n---------------------------------------------------------------------------------\n\n"

#Información básica
print(df_stats.shape)       # filas y columnas
print(separador)
print(df_stats.info())      # tipos de datos
print(separador)
print(df_stats.describe())  # estadísticas
print(separador)
print(df_stats.head())      # primeras 
print(separador)


#Aplicacion de algoritmo Kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_stats["Cluster"] = clusters
print(df_stats["Cluster"])

#Visualizacion

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df_stats["Cluster"], cmap="tab10", alpha=0.6)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("Clustering de jugadores de CS:GO")
plt.show()

#Interpretacion
df_stats.groupby("Cluster")[features].mean()
