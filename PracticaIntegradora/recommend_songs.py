import streamlit as st
from spotify_song_analysis import recommend

#Despligue automatizado

st.title("🎶 Recomendador de Canciones Spotify")

song = st.text_input("Ingresa una canción para obtener recomendaciones:")
if st.button("Recomendar"):
    try:
        recs = recommend(song)
        st.dataframe(recs)
    except:
        st.error("Canción no encontrada en el dataset.")
