"""
🧠 VECTOR ENGINE - Motor de Memoria Semántica
==============================================

Base de datos vectorial casera usando:
- OpenAI Embeddings (text-embedding-3-small) - Barato y potente
- NumPy para cálculos de similitud
- JSON para persistencia

No necesita Chroma, Pinecone ni bases de datos complejas.
"""

import json
import numpy as np
import os
from openai import OpenAI
from datetime import datetime

# Configuración
DB_FILE = "memoria_permanente.json"
# Usamos el modelo 'small' que es muy barato y rápido
EMBEDDING_MODEL = "text-embedding-3-small" 

def get_embedding(text, client):
    """Convierte texto en un vector de 1536 números."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding

def cosine_similarity(v1, v2):
    """Calcula qué tan parecidos son dos vectores (Matemática pura)."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class VectorMemoryDB:
    def __init__(self, client):
        self.client = client
        self.memories = []
        self.load_db()

    def load_db(self):
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r') as f:
                self.memories = json.load(f)
            print(f"📚 VectorDB: {len(self.memories)} recuerdos cargados.")
            
            # Verificación de integridad: ¿Tienen vectores?
            if self.memories and 'vector' not in self.memories[0]:
                print("⚠️ ADVERTENCIA: Memorias antiguas sin vector detectadas.")
                print("   Ejecuta el script de migración o espera a que se generen al vuelo.")

    def save_db(self):
        with open(DB_FILE, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def add_memory(self, text, importance_score):
        """Guarda texto + su vector matemático."""
        vector = get_embedding(text, self.client)
        
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "content": text,
            "score": f"{importance_score:.1f}%",
            "vector": vector  # <--- LA MAGIA
        }
        self.memories.append(entry)
        self.save_db()
        return entry

    def search(self, query, top_k=3):
        """Busca los recuerdos más relevantes semánticamente."""
        if not self.memories:
            return []

        # 1. Vectorizamos la pregunta del usuario
        query_vector = get_embedding(query, self.client)
        
        # 2. Comparamos con TODOS los recuerdos
        scored_memories = []
        for mem in self.memories:
            if 'vector' not in mem: continue # Saltar corruptos
            
            sim = cosine_similarity(query_vector, mem['vector'])
            scored_memories.append((sim, mem))
        
        # 3. Ordenamos por similitud (de mayor a menor)
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # 4. Devolvemos los mejores
        results = []
        print(f"\n🔍 BÚSQUEDA SEMÁNTICA PARA: '{query}'")
        for score, mem in scored_memories[:top_k]:
            print(f"   ➤ [{score:.4f}] {mem['content']}")
            results.append(mem)
            
        return results
