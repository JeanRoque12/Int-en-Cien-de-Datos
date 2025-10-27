import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#Exportacion de datos
df_stats = pd.read_csv("./Popular_Spotify_Songs.csv")


#Estructura y tipos de datos
df_stats.head()
df_stats.info()
df_stats.describe()

#Valores nulos o faltantes
df_stats.isnull().sum()
df_stats = df_stats.dropna(subset=['track_name', 'artist(s)_name'])

#Valores duplicados
df_stats = df_stats.drop_duplicates(subset=['track_name', 'artist(s)_name'])

#Top 10 canciones
df_stats.nlargest(10, 'streams')[['track_name','artist(s)_name','streams']]

#Correlacion entre variables
plt.figure(figsize=(10,6))
sns.heatmap(df_stats[['bpm','energy_%','danceability_%','valence_%','acousticness_%']].corr(), annot=True, cmap='coolwarm')

#Distribución de canciones por año de lanzamiento
plt.figure(figsize=(10,5))
sns.countplot(x='released_year', data=df_stats, palette='viridis')
plt.title('Cantidad de canciones populares por año de lanzamiento')
plt.xticks(rotation=45)
plt.show()

#Artistas con mas canciones
top_artists = df_stats['artist(s)_name'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_artists.values, y=top_artists.index, palette='magma')
plt.title('Top 10 artistas con más canciones populares')
plt.xlabel('Número de canciones')
plt.ylabel('Artista')
plt.show()

#Relacion entre energia y baile
plt.figure(figsize=(8,6))
sns.scatterplot(x='energy_%', y='danceability_%', data=df_stats, alpha=0.6)
plt.title('Relación entre Energía y Bailabilidad')
plt.xlabel('Energía (%)')
plt.ylabel('Bailabilidad (%)')
plt.show()


#Preprocesamiento de datos
metricas = df_stats[['bpm','danceability_%','valence_%','energy_%','acousticness_%','instrumentalness_%','liveness_%','speechiness_%']]
escalador = StandardScaler()
features_scaled = escalador.fit_transform(metricas)

#Modelado basado en contenido
similarity = cosine_similarity(features_scaled)

def recommend(song_name, n=5):
    idx = df_stats[df_stats['track_name'].str.lower() == song_name.lower()].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top = scores[1:n+1]
    recommendations = df_stats.iloc[[i[0] for i in top]][['track_name','artist(s)_name','bpm','danceability_%','energy_%']]
    return recommendations

