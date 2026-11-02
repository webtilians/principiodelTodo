"""
🔮 INFINITO JARVIS VECTORIAL - Con Búsqueda Semántica RAG
==========================================================

Este es el Jarvis definitivo que combina:
1. Infinito Gate (El Portero) - Decide qué guardar
2. VectorMemoryDB - Almacena con embeddings
3. OpenAI GPT - Genera respuestas inteligentes con contexto RAG

Uso: python infinito_jarvis_vector.py
"""

import torch
import sys
import os

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infinito_memory_keeper import InfinitoDynamicChat, text_to_ids
from vector_engine import VectorMemoryDB

# --- CONFIGURACIÓN ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")

if len(API_KEY) < 10:
    print("❌ Configura tu API Key en .env o variable de entorno OPENAI_API_KEY")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)

class JarvisVectorSystem:
    def __init__(self, keeper_model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Cargar Infinito (El Portero)
        print(f"🧠 Cargando Infinito Gate en {self.device}...")
        # num_layers=2 para coincidir con el modelo entrenado
        self.keeper = InfinitoDynamicChat(vocab_size=256, hidden_dim=64, num_layers=2, use_improved_memory=True).to(self.device)
        checkpoint = torch.load(keeper_model_path, map_location=self.device, weights_only=False)
        # Cargar solo los pesos que coinciden, ignorando las claves de atención guardadas
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.keeper.load_state_dict(state_dict, strict=False)
        self.keeper.eval()
        print("   ✅ Gate cargado")

        # 2. Cargar Base de Datos Vectorial
        print("📚 Cargando VectorDB...")
        self.db = VectorMemoryDB(client)
        print("   ✅ VectorDB listo")

    def _es_pregunta(self, texto):
        """Detecta si el texto es una pregunta."""
        t = texto.lower().strip()
        if '?' in t or '¿' in t: 
            return True
        starts = ['que', 'qué', 'como', 'cómo', 'cual', 'cuál', 'quien', 'quién', 
                  'donde', 'dónde', 'cuando', 'cuándo', 'cuanto', 'cuánto', 'por que', 'por qué']
        words = t.split()
        if words and words[0] in starts: 
            return True
        return False

    def chat(self, user_text):
        # --- FASE 1: FILTRADO (Infinito Keeper) ---
        inp = text_to_ids(user_text).to(self.device)
        with torch.no_grad():
            _, metrics = self.keeper(inp, return_metrics=True)
        
        importance = metrics['gate_value'] * 100
        is_question = self._es_pregunta(user_text)
        should_save = (importance > 50.0) and (not is_question)

        feedback = f" [Gate: {importance:.1f}%"
        
        if should_save:
            self.db.add_memory(user_text, importance)  # Guardar con Vector
            feedback += " 🟢 GUARDADO (Vectorizado)]"
        else:
            feedback += " 🔴 NO GUARDADO]"
        print(feedback)

        # --- FASE 2: RECUPERACIÓN (RAG Vectorial) ---
        # Buscamos recuerdos RELEVANTES para lo que acabas de decir
        relevant_memories = self.db.search(user_text, top_k=3)
        
        # --- FASE 3: GENERACIÓN (OpenAI) ---
        context_block = ""
        if relevant_memories:
            context_block = "CONTEXTO RELEVANTE DE TU MEMORIA:\n"
            for mem in relevant_memories:
                context_block += f"- {mem['content']} (Fecha: {mem['timestamp']})\n"
        else:
            context_block = "No encontré recuerdos relacionados con este tema."

        system_prompt = f"""Eres Infinito, un asistente con memoria perfecta.

{context_block}

Usa el contexto anterior SOLO si es útil para responder.
Si te preguntan algo personal, mira el contexto.
Responde de forma natural y amigable."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    MODEL_PATH = "models/dynamic_chat_detector_v3.pt"  # v3: entrenado con 10k+ datos híbridos
    
    print("\n" + "="*55)
    print("🔮 INFINITO VECTORIAL - Búsqueda Semántica Activada")
    print("="*55)
    
    jarvis = JarvisVectorSystem(MODEL_PATH)
    
    print("\n" + "─"*55)
    print("Escribe 'salir' para terminar")
    print("─"*55)
    
    while True:
        try:
            text = input("\n👤 Tú > ").strip()
            if not text:
                continue
            if text.lower() == 'salir': 
                print("\n👋 ¡Hasta luego!")
                break
            
            print("🔍 Buscando contexto...", end="\r")
            response = jarvis.chat(text)
            print(f"🤖 Infinito > {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
