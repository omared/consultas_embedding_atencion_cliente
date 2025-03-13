import streamlit as st
import faiss
import numpy as np
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# Convertir embeddings almacenados en el DataFrame a una matriz NumPy
embeddings = np.vstack(df["embedding"].values)

# Crear un índice FAISS desde cero
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Cargar modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar datos
file_name = "tu_archivo.json"  # Asegúrate de tener este archivo en el mismo directorio

with open(file_name, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Cargar índice FAISS
index = faiss.read_index("index_faiss.bin")

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
        st.success(f"**Respuesta Sugerida:** {respuesta}")
    else:
        st.warning("Por favor ingrese una consulta.")