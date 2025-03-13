import streamlit as st
import faiss
import numpy as np
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# Cargar modelo de embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Cargar datos desde el JSON
@st.cache_data
def load_data():
    file_name = "tu_archivo.json"  # Asegúrate de que el JSON esté en el mismo repositorio
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Generar embeddings de las preguntas
    df["embedding"] = df["Cliente"].apply(lambda x: model.encode(x, convert_to_numpy=True))

    # Convertir embeddings a matriz NumPy
    embeddings = np.vstack(df["embedding"].values)

    # Crear y cargar el índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return df, index

df, index = load_data()

# Función para buscar la mejor respuesta
def buscar_respuesta(pregunta, top_k=1):
    pregunta_embedding = model.encode(pregunta, convert_to_numpy=True).reshape(1, -1)
    _, indices = index.search(pregunta_embedding, top_k)
    respuestas = df.iloc[indices[0]]["Agente"].values
    return respuestas

# Interfaz con Streamlit
st.title("Asistente de Soporte Bancario con IA 💳🤖")

pregunta = st.text_input("Ingrese la consulta del cliente:")

if st.button("Buscar Respuesta"):
    if pregunta:
        respuesta = buscar_respuesta(pregunta)[0]
        st.success(f"**Respuesta Sugerida:** {respues
