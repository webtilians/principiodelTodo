#!/usr/bin/env python3
"""
🧠 INFINITO RAG Demo - Búsqueda Semántica con Embeddings Conscientes
=====================================================================

Demo interactiva de Retrieval-Augmented Generation usando:
- Embeddings de Capa 11 (mejor discriminación semántica)
- Métricas PHI en tiempo real
- ChromaDB para almacenamiento vectorial

Ejecutar: streamlit run app_rag_demo.py
"""

import streamlit as st
import torch
import os
import sys
import time
import numpy as np
from datetime import datetime

# Añadir paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="INFINITO RAG Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .phi-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }
    .similarity-high { color: #00ff88; font-weight: bold; }
    .similarity-medium { color: #ffcc00; }
    .similarity-low { color: #ff6b6b; }
    .doc-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES ---
MODEL_PATH = "models/infinito_gpt2_spanish_phi.pt"
CHROMADB_PATH = "data/chromadb_rag_demo"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@st.cache_resource
def load_model():
    """Carga el modelo INFINITO (cacheado)."""
    from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver
    
    model = InfinitoGPT2WithObserver()
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_extractor(_model):
    """Carga el extractor de embeddings (cacheado)."""
    from src.core.layer4_embeddings import Layer4Extractor
    return Layer4Extractor(_model, device=DEVICE, use_search_layer=True)


@st.cache_resource
def load_vectordb(_extractor):
    """Carga/crea la base de datos vectorial."""
    from src.core.conscious_vectordb import ConsciousVectorDB
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(CHROMADB_PATH), exist_ok=True)
    
    # Usar el modelo ya cargado
    db = ConsciousVectorDB.__new__(ConsciousVectorDB)
    db.collection_name = "rag_demo"
    db.persist_directory = CHROMADB_PATH
    db.extractor = _extractor
    
    import chromadb
    db.client = chromadb.PersistentClient(path=CHROMADB_PATH)
    db.collection = db.client.get_or_create_collection(
        name="rag_demo",
        metadata={
            "description": "RAG Demo with conscious embeddings",
            "hnsw:space": "cosine"
        }
    )
    
    return db


def generate_response(model, tokenizer, prompt, context, max_length=100):
    """Genera respuesta con contexto RAG."""
    full_prompt = f"""Contexto relevante:
{context}

Pregunta: {prompt}

Respuesta basada en el contexto:"""
    
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=400)
    input_ids = inputs.input_ids.to(DEVICE)
    
    phi_values = []
    generated = input_ids
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs, metrics = model(generated, return_phi=True, use_memory=False)
            phi_values.append(metrics['phi'].mean().item())
            
            next_token_logits = outputs.logits[:, -1, :] / 0.7
            
            top_k = 50
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            decoded = tokenizer.decode(next_token[0])
            if decoded in ['.', '\n', '</s>', '<|endoftext|>']:
                break
    
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # Extraer solo la respuesta
    if "Respuesta basada en el contexto:" in full_text:
        response = full_text.split("Respuesta basada en el contexto:")[-1].strip()
    else:
        response = full_text
    
    return response, np.mean(phi_values) if phi_values else 0.0


# --- INICIALIZACIÓN ---
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []


# --- INTERFAZ PRINCIPAL ---
def main():
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 INFINITO RAG Demo</h1>', unsafe_allow_html=True)
    st.markdown("**Retrieval-Augmented Generation con Embeddings Conscientes (Capa 11)**")
    
    # --- CARGA DE COMPONENTES ---
    with st.spinner("Cargando modelo INFINITO..."):
        try:
            model = load_model()
            st.success(f"✅ Modelo cargado en {DEVICE.upper()}")
        except Exception as e:
            st.error(f"❌ Error cargando modelo: {e}")
            return
    
    with st.spinner("Inicializando extractor..."):
        try:
            extractor = load_extractor(model)
            st.success(f"✅ Extractor Capa {extractor.TARGET_LAYER} listo")
        except Exception as e:
            st.error(f"❌ Error con extractor: {e}")
            return
    
    # --- LAYOUT ---
    col_main, col_sidebar = st.columns([3, 1])
    
    with col_sidebar:
        st.markdown("## 📚 Base de Conocimiento")
        
        # Documentos predefinidos
        default_docs = [
            ("PHI (Φ) es una medida propuesta por Giulio Tononi que cuantifica "
             "el nivel de integración de información en un sistema.", "consciencia"),
            ("Los transformers son una arquitectura de red neuronal que usa "
             "mecanismos de atención para procesar secuencias.", "ia"),
            ("La Capa 4 de GPT-2 muestra el máximo nivel de PHI (8.91), "
             "indicando máxima integración de información.", "infinito"),
            ("El cerebro humano contiene 86 mil millones de neuronas "
             "conectadas por sinapsis.", "neurociencia"),
            ("INFINITO es un framework que integra IIT con transformers "
             "para analizar integración de información.", "infinito"),
        ]
        
        # Añadir documentos predefinidos
        if st.button("📥 Cargar docs de ejemplo", use_container_width=True):
            with st.spinner("Indexando documentos..."):
                for doc, cat in default_docs:
                    if doc not in [d['text'] for d in st.session_state.documents]:
                        emb, phi = extractor.extract_with_phi(doc)
                        st.session_state.documents.append({
                            'text': doc,
                            'embedding': emb,
                            'phi': phi['phi'],
                            'category': cat
                        })
            st.success(f"✅ {len(st.session_state.documents)} documentos indexados")
        
        # Añadir documento manual
        with st.expander("➕ Añadir documento"):
            new_doc = st.text_area("Texto del documento", height=100)
            new_cat = st.selectbox("Categoría", ["general", "consciencia", "ia", "infinito", "neurociencia"])
            
            if st.button("Añadir", use_container_width=True):
                if new_doc:
                    with st.spinner("Calculando embedding..."):
                        emb, phi = extractor.extract_with_phi(new_doc)
                        st.session_state.documents.append({
                            'text': new_doc,
                            'embedding': emb,
                            'phi': phi['phi'],
                            'category': new_cat
                        })
                    st.success(f"✅ Documento añadido (PHI: {phi['phi']:.3f})")
        
        st.divider()
        
        # Estadísticas
        if st.session_state.documents:
            st.metric("📊 Documentos", len(st.session_state.documents))
            avg_phi = np.mean([d['phi'] for d in st.session_state.documents])
            st.metric("Φ PHI Medio", f"{avg_phi:.3f}")
            
            # Categorías
            cats = {}
            for d in st.session_state.documents:
                cats[d['category']] = cats.get(d['category'], 0) + 1
            
            st.markdown("**Por categoría:**")
            for cat, count in cats.items():
                st.caption(f"• {cat}: {count}")
        
        # Limpiar
        if st.button("🗑️ Limpiar base", use_container_width=True):
            st.session_state.documents = []
            st.rerun()
        
        st.divider()
        st.markdown("### 💤 Mantenimiento Cognitivo")
        
        # Configuración del Curator
        with st.expander("⚙️ Configuración del Sueño"):
            sleep_mode = st.radio(
                "Modo de sueño",
                ["🌙 Normal", "🧹 Limpieza Profunda", "🔬 Solo Análisis"],
                horizontal=True,
                help="Normal: fusión + poda ligera | Profunda: poda agresiva de trivialidades | Análisis: no modifica nada"
            )
            
            max_operations = st.slider("Operaciones máximas", 1, 10, 5)
            
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                merge_threshold = st.slider("Umbral fusión", 0.5, 0.95, 0.75, 0.05, help="Similitud mínima para fusionar")
            with col_cfg2:
                phi_threshold = st.slider("Mejora Φ mínima", 0.01, 0.10, 0.02, 0.01, help="% mejora para aceptar fusión")
        
        # Análisis de salud de la memoria
        if st.button("📊 Analizar Salud", use_container_width=True, help="Evalúa redundancia y calidad de la memoria"):
            with st.spinner("Analizando memoria..."):
                try:
                    from src.core.knowledge_curator import KnowledgeCurator
                    
                    # Crear DB temporal con los documentos actuales
                    db = load_vectordb(extractor)
                    
                    # Sincronizar documentos de sesión con DB
                    if st.session_state.documents:
                        existing = db.collection.get()
                        existing_texts = set(existing.get('documents', []))
                        
                        for doc in st.session_state.documents:
                            if doc['text'] not in existing_texts:
                                db.collection.add(
                                    embeddings=[doc['embedding'].tolist()],
                                    documents=[doc['text']],
                                    metadatas=[{'phi': doc['phi'], 'category': doc['category']}],
                                    ids=[f"sync_{hash(doc['text']) % 10000}"]
                                )
                    
                    curator = KnowledgeCurator(model, extractor, db)
                    health = curator.analyze_memory_health()
                    
                    # Mostrar métricas
                    col_h1, col_h2 = st.columns(2)
                    with col_h1:
                        st.metric("📚 Memorias", health['total_memories'])
                        st.metric("🗑️ PHI Bajo", health['low_phi_count'])
                    with col_h2:
                        st.metric("Φ Promedio", f"{health['avg_phi']:.3f}")
                        st.metric("🔄 Redundancia", f"{health['redundancy_score']*100:.1f}%")
                    
                    # Score de salud
                    health_score = health.get('health_score', 0)
                    if health_score > 0.7:
                        st.success(f"✅ Memoria saludable (Score: {health_score:.2f})")
                    elif health_score > 0.4:
                        st.warning(f"⚠️ Memoria necesita optimización (Score: {health_score:.2f})")
                    else:
                        st.error(f"🔴 Memoria degradada (Score: {health_score:.2f})")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("🌙 Iniciar Ciclo de Sueño", use_container_width=True, help="El modelo reflexionará sobre sus memorias para optimizarlas"):
            with st.spinner("Soñando y reestructurando conocimiento..."):
                try:
                    from src.core.knowledge_curator import KnowledgeCurator
                    
                    # Cargar/crear la base de datos con los documentos actuales
                    db = load_vectordb(extractor)
                    
                    # Sincronizar documentos de la sesión con la DB
                    if st.session_state.documents:
                        existing = db.collection.get()
                        existing_texts = set(existing.get('documents', []))
                        
                        for doc in st.session_state.documents:
                            if doc['text'] not in existing_texts:
                                db.collection.add(
                                    embeddings=[doc['embedding'].tolist()],
                                    documents=[doc['text']],
                                    metadatas=[{'phi': doc['phi'], 'category': doc['category']}],
                                    ids=[f"sync_{hash(doc['text']) % 10000}"]
                                )
                    
                    # Configurar Curator según modo seleccionado
                    prune_thresh = 0.3  # Normal
                    if sleep_mode == "🧹 Limpieza Profunda":
                        prune_thresh = 5.0  # Más agresivo (elimina más)
                    
                    curator = KnowledgeCurator(
                        model, extractor, db,
                        phi_improvement_threshold=phi_threshold,
                        similarity_threshold_merge=merge_threshold,
                        prune_threshold=prune_thresh
                    )
                    
                    # Solo análisis no ejecuta cambios
                    if sleep_mode == "🔬 Solo Análisis":
                        health = curator.analyze_memory_health()
                        logs = [f"📊 Análisis completado:\n   - Memorias: {health['total_memories']}\n   - Redundancia: {health['redundancy_score']*100:.1f}%\n   - PHI medio: {health['avg_phi']:.3f}"]
                    else:
                        # Ejecutar ciclo de sueño CON VERBOSE para debug en consola
                        print("\n" + "="*60)
                        print(f"🌙 CICLO DE SUEÑO - Modo: {sleep_mode}")
                        print(f"   Config: merge>{merge_threshold}, phi_improve>{phi_threshold}, prune<{prune_thresh}")
                        print("="*60)
                        logs = curator.sleep_cycle(max_steps=max_operations, verbose=True)
                        print("="*60 + "\n")
                    
                    # Guardar resultados en session_state para que persistan
                    st.session_state.sleep_results = {
                        'logs': logs,
                        'stats': curator.get_stats(),
                        'mode': sleep_mode
                    }
                    
                    # Recargar documentos desde la DB actualizada
                    updated_data = db.collection.get(include=['documents', 'metadatas', 'embeddings'])
                    st.session_state.documents = []
                    
                    # Manejar arrays numpy de forma segura (no usar 'or' con arrays)
                    documents_list = updated_data.get('documents')
                    metadatas_list = updated_data.get('metadatas')
                    embeddings_list = updated_data.get('embeddings')
                    
                    # Convertir None a lista vacía de forma segura
                    if documents_list is None:
                        documents_list = []
                    if metadatas_list is None:
                        metadatas_list = []
                    if embeddings_list is None:
                        embeddings_list = []
                    
                    for i, doc_text in enumerate(documents_list):
                        meta = metadatas_list[i] if i < len(metadatas_list) else {}
                        if meta is None:
                            meta = {}
                        
                        emb = None
                        if i < len(embeddings_list):
                            emb_data = embeddings_list[i]
                            if emb_data is not None:
                                emb = np.array(emb_data)
                        
                        st.session_state.documents.append({
                            'text': doc_text,
                            'embedding': emb,
                            'phi': meta.get('phi', 0) if isinstance(meta, dict) else 0,
                            'category': meta.get('category', 'general') if isinstance(meta, dict) else 'general'
                        })
                    
                except Exception as e:
                    st.error(f"Error en ciclo de sueño: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Mostrar resultados del último ciclo de sueño (persistentes)
        if 'sleep_results' in st.session_state and st.session_state.sleep_results:
            results = st.session_state.sleep_results
            
            st.success("🌅 Último ciclo de sueño")
            
            # Mostrar estadísticas
            stats = results['stats']
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("✨ Fusiones", stats['total_merges'])
            with col_s2:
                st.metric("🗑️ Podas", stats['total_prunes'])
            with col_s3:
                st.metric("Φ Ganado", f"+{stats['phi_gained']:.3f}")
            
            # Mostrar logs detallados
            with st.expander("📝 Diario de Sueño", expanded=True):
                for log in results['logs']:
                    if "✨" in log:
                        st.info(log)
                    elif "🗑️" in log:
                        st.warning(log)
                    elif "📉" in log:
                        st.caption(log)
                    elif "💤" in log:
                        st.success(log)
                    else:
                        st.text(log)
            
            # Botón para limpiar resultados
            if st.button("🧹 Limpiar resultados", key="clear_sleep"):
                st.session_state.sleep_results = None
                st.rerun()
    
    with col_main:
        st.markdown("## 💬 Búsqueda Semántica + Generación")
        
        # Input de búsqueda
        query = st.text_input(
            "🔍 Pregunta",
            placeholder="Ej: ¿Qué es PHI y cómo se relaciona con la consciencia?",
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            top_k = st.slider("Documentos a recuperar", 1, 5, 2)
        with col2:
            generate = st.checkbox("Generar respuesta", value=True)
        
        if st.button("🚀 Buscar", use_container_width=True) and query:
            if not st.session_state.documents:
                st.warning("⚠️ Añade documentos primero")
            else:
                with st.spinner("Buscando..."):
                    # Calcular embedding de la query
                    query_emb, query_phi = extractor.extract_with_phi(query)
                    
                    st.markdown(f"**PHI de la pregunta:** `{query_phi['phi']:.4f}`")
                    
                    # Buscar documentos similares
                    similarities = []
                    for doc in st.session_state.documents:
                        sim = np.dot(query_emb, doc['embedding'])
                        similarities.append((sim, doc))
                    
                    # Ordenar por similitud
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    top_docs = similarities[:top_k]
                    
                    # Mostrar resultados
                    st.markdown("### 📚 Documentos Recuperados")
                    
                    context_text = ""
                    for i, (sim, doc) in enumerate(top_docs, 1):
                        sim_class = "similarity-high" if sim > 0.1 else "similarity-medium" if sim > 0 else "similarity-low"
                        
                        st.markdown(f"""
                        <div class="doc-card">
                            <strong>[{i}]</strong> 
                            <span class="{sim_class}">Sim: {sim:.4f}</span> | 
                            <span class="phi-badge">Φ {doc['phi']:.3f}</span> |
                            <em>{doc['category']}</em>
                            <p style="margin-top: 10px;">{doc['text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        context_text += doc['text'] + " "
                    
                    # Generar respuesta
                    if generate:
                        st.markdown("### 🤖 Respuesta Generada")
                        
                        with st.spinner("Generando con PHI Observer..."):
                            response, response_phi = generate_response(
                                model, model.tokenizer, query, context_text[:500]
                            )
                        
                        st.markdown(f"**PHI de la respuesta:** `{response_phi:.4f}`")
                        st.info(response)
        
        # Historial
        st.divider()
        st.markdown("### 📊 Análisis de Embeddings")
        
        if st.session_state.documents:
            # Matriz de similitud
            if st.checkbox("Mostrar matriz de similitud"):
                n = len(st.session_state.documents)
                sim_matrix = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(n):
                        sim_matrix[i, j] = np.dot(
                            st.session_state.documents[i]['embedding'],
                            st.session_state.documents[j]['embedding']
                        )
                
                import pandas as pd
                # Usar índices numéricos únicos para compatibilidad con Styler
                df = pd.DataFrame(sim_matrix).round(3)
                
                # Mostrar leyenda de categorías
                legend = " | ".join([f"{i}: {d['category'][:10]}" for i, d in enumerate(st.session_state.documents)])
                st.caption(f"📋 {legend}")
                
                # Mostrar matriz con gradiente de color
                st.dataframe(
                    df.style.background_gradient(cmap='RdYlGn', vmin=-0.2, vmax=1.0).format("{:.3f}"),
                    use_container_width=True
                )
            
            # PHI por categoría
            if st.checkbox("PHI por categoría"):
                cats_phi = {}
                for d in st.session_state.documents:
                    cat = d['category']
                    if cat not in cats_phi:
                        cats_phi[cat] = []
                    cats_phi[cat].append(d['phi'])
                
                import pandas as pd
                data = [(cat, np.mean(phis), len(phis)) for cat, phis in cats_phi.items()]
                df = pd.DataFrame(data, columns=['Categoría', 'PHI Medio', 'N'])
                st.bar_chart(df.set_index('Categoría')['PHI Medio'])


if __name__ == "__main__":
    main()
