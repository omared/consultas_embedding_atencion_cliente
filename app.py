import streamlit as st
import faiss
import numpy as np
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Cargar modelo de embeddings
@st.cache_resource
def load_model():
    #return SentenceTransformer("all-MiniLM-L6-v2") no razona bien
     return SentenceTransformer("paraphrase-mpnet-base-v2")  # Modelo más potente

model = load_model()

# Cargar datos y crear embeddings normalizados
@st.cache_data
def load_data():
    file_name = "tabular-actgan-ajustada.json"  # Asegúrate de que el JSON esté en el mismo repositorio
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Generar embeddings normalizados
    df["embedding"] = df["Cliente"].apply(lambda x: model.encode(x, convert_to_numpy=True))

    # Convertir embeddings a matriz NumPy
    embeddings = np.vstack(df["embedding"].values)
    embeddings = normalize(embeddings, axis=1)  # Normalizar para mejorar FAISS

     # Crear índice FAISS con normalización
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Cambiar a Inner Product para mejor precisión
    index.add(embeddings)

    return df, index

df, index = load_data()

# Función para buscar la mejor respuesta
def buscar_respuesta(pregunta, top_k=3):
    pregunta_embedding = model.encode(pregunta, convert_to_numpy=True).reshape(1, -1)
    pregunta_embedding = normalize(pregunta_embedding, axis=1)
    # Normalizar antes de buscar
    _, indices = index.search(pregunta_embedding, top_k)
    respuestas = df.iloc[indices[0]]["Agente"].values
    return respuestas

# Interfaz con Streamlit
st.title("Asistente de Soporte Bancario con IA 💳🤖")

pregunta = st.text_input("Ingrese la consulta del cliente:")

if st.button("Buscar Respuesta"):
    if pregunta:
        respuesta = buscar_respuesta(pregunta)
        st.success("**Mejores respuestas sugeridas:**")
        for i, respuesta in enumerate(respuestas):
            st.write(f"👉 {i+1}. {respuesta}")

    else:
        st.warning("Por favor ingrese una consulta.")
