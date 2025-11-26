"""
🔄 MIGRATE MEMORIES - Migrador de Memoria a Vectores
=====================================================

Este script lee los recuerdos existentes en memoria_permanente.json
y les añade su vector de embedding si no lo tienen.

Uso: python src/migrate_memories.py
"""

import json
import os
import sys

# Añadir el directorio actual al path para imports
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from dotenv import load_dotenv
from vector_engine import get_embedding

# Cargar variables de entorno
load_dotenv()

# --- CONFIGURA TU API KEY ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not API_KEY:
    print("❌ ERROR: No se encontró OPENAI_API_KEY")
    print("   Configura tu .env o variable de entorno")
    exit(1)

DB_FILE = "memoria_permanente.json"

# Cambiar al directorio raíz del proyecto
os.chdir(os.path.dirname(os.path.dirname(__file__)))

if not os.path.exists(DB_FILE):
    print("No hay memoria que migrar.")
    print(f"   Archivo buscado: {os.path.abspath(DB_FILE)}")
    exit()

client = OpenAI(api_key=API_KEY)

print("🔄 Iniciando migración a Vectores...")
with open(DB_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

count = 0
for entry in data:
    if 'vector' not in entry:
        print(f"   Vectorizando: '{entry['content']}'...")
        entry['vector'] = get_embedding(entry['content'], client)
        count += 1

with open(DB_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ ¡Hecho! {count} recuerdos actualizados con inteligencia vectorial.")
