import streamlit as st
import faiss
import numpy as np
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Cargar modelo
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-mpnet-base-v2")

model = load_model()

# Cargar datos y crear índice FAISS
@st.cache_data
def load_data():
    file_name = "tu_archivo.json"  # Asegúrate de que el JSON esté en el repositorio
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Generar embeddings normalizados
    df["embedding"] = df["Cliente"].apply(lambda x: model.encode(x, convert_to_numpy=True))
    embeddings = np.vstack(df["embedding"].values)
    embeddings = normalize(embeddings, axis=1)  # Normalizar para mejor búsqueda

    # Crear índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return df, index

df, index = load_data()

# Función para buscar respuesta
def buscar_respuesta(pregunta, top_k=3):
    if not pregunta.strip():  # Validar que la pregunta no esté vacía
        return []

    pregunta_embedding = model.encode(pregunta, convert_to_numpy=True).reshape(1, -1)
    pregunta_embedding = normalize(pregunta_embedding, axis=1)  # Normalizar

    _, indices = index.search(pregunta_embedding, top_k)

    respuestas = df.iloc[indices[0]]["Agente"].values if len(indices[0]) > 0 else []
    return respuestas

# Interfaz Streamlit
st.title("Asistente de Soporte Bancario con IA 💳🤖")

pregunta = st.text_input("Ingrese la consulta del cliente:")

if st.button("Buscar Respuesta"):
    if pregunta:
        respuestas = buscar_respuesta(pregunta)
        if respuestas:
            st.success("**Mejores respuestas sugeridas:**")
            for i, respuesta in enumerate(respuestas):
                st.write(f"👉 {i+1}. {respuesta}")
        else:
            st.warning("No se encontró una respuesta relevante.")
    else:
        st.warning("Por favor ingrese una consulta.")

