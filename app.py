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
    file_name = "tabular-actgan-ajustada.json"  # Asegúrate de que el JSON esté en el repositorio
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Generar embeddings normalizados
    df["embedding"] = df["Cliente"].apply(lambda x: model.encode(x, convert_to_numpy=True))
    embeddings = np.vstack(df["embedding"].values)
    embeddings = normalize(embeddings, axis=1)  # Normalizar para mejorar la precisión

    # Crear índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return df, index

df, index = load_data()

# Función para buscar respuesta
def buscar_respuesta(pregunta, top_k=3):
    if not pregunta or not pregunta.strip():  # Validar que la pregunta no esté vacía
        return []

    pregunta_embedding = model.encode(pregunta, convert_to_numpy=True).reshape(1, -1)
    pregunta_embedding = normalize(pregunta_embedding, axis=1)  # Normalizar

    _, indices = index.search(pregunta_embedding, top_k)

    # Asegurar que indices no esté vacío y que haya resultados válidos
    if len(indices) == 0 or len(indices[0]) == 0:
        return []

    respuestas = df.iloc[indices[0]]["Agente"].values.tolist()
    return respuestas

# Interfaz Streamlit
st.title("Asistente de Soporte Bancario con IA 💳🤖")

pregunta = st.text_input("Ingrese la consulta del cliente:")

if st.button("Buscar Respuesta"):
    if pregunta:
        respuestas = buscar_respuesta(pregunta)

        if isinstance(respuestas, list) and len(respuestas) > 0:  # Validar que sea lista y no esté vacía
            st.success("**Mejores respuestas sugeridas:**")
            for i, respuesta in enumerate(respuestas):
                st.w


