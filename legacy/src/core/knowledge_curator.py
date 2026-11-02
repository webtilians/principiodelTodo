"""
🧠 INFINITO Knowledge Curator (El Jardinero)
============================================
Agente de mantenimiento cognitivo que estructura la memoria durante
los periodos de inactividad ("sueño").

Funciones:
1. MERGE: Fusiona memorias redundantes en conceptos de mayor densidad (Phi).
2. PRUNE: Elimina memorias de bajo valor (ruido) que no se usan.
3. REINFORCE: Detecta conexiones latentes entre conceptos distantes.

El algoritmo está inspirado en la consolidación de memoria durante el sueño REM,
donde el cerebro reorganiza y fortalece conexiones importantes mientras
elimina información irrelevante.

Uso:
    from src.core.knowledge_curator import KnowledgeCurator
    
    curator = KnowledgeCurator(model, extractor, vectordb)
    logs = curator.sleep_cycle(max_steps=5)
"""

import torch
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class KnowledgeCurator:
    """
    Agente autónomo de mantenimiento cognitivo.
    
    Implementa un "Algoritmo de Sueño" que:
    - Fusiona memorias similares en conceptos de mayor densidad Φ
    - Elimina memorias de bajo valor semántico (ruido)
    - Busca reducir la entropía de la memoria
    """
    
    def __init__(
        self, 
        model, 
        extractor,
        vectordb,
        device: str = None,
        similarity_threshold_merge: float = 0.82,
        phi_improvement_threshold: float = 0.05,
        prune_threshold: float = 0.3
    ):
        """
        Args:
            model: InfinitoGPT2WithObserver con .gpt2 y .tokenizer
            extractor: Layer4Extractor para embeddings y PHI
            vectordb: ConsciousVectorDB o similar con .collection
            device: 'cuda' o 'cpu'
            similarity_threshold_merge: Umbral de similitud para fusionar (0.82 = muy parecidos)
            phi_improvement_threshold: Mejora mínima de PHI para aceptar fusión (5%)
            prune_threshold: PHI mínimo para que un recuerdo sobreviva
        """
        self.model = model
        self.extractor = extractor
        self.db = vectordb
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = model.tokenizer
        
        # Umbrales Heurísticos (Política Inicial del Agente)
        self.similarity_threshold_merge = similarity_threshold_merge
        self.phi_improvement_threshold = phi_improvement_threshold
        self.prune_threshold = prune_threshold
        
        # Estadísticas del agente
        self.stats = {
            'total_merges': 0,
            'total_prunes': 0,
            'total_cycles': 0,
            'phi_gained': 0.0
        }
        
        print("🕵️‍♂️ Curator Agent inicializado y listo para el ciclo de sueño.")
        print(f"   - Umbral fusión: {similarity_threshold_merge}")
        print(f"   - Mejora PHI mínima: {phi_improvement_threshold*100}%")
        print(f"   - PHI mínimo supervivencia: {prune_threshold}")
        
        # Patrones triviales comunes
        self.trivial_patterns = [
            'hola', 'ok', 'vale', 'si', 'no', 'gracias', 'adios', 'bye',
            'hey', 'buenas', 'buenos dias', 'buenas noches', 'hello',
            'jaja', 'jeje', 'lol', 'xd', 'wow', 'oye', 'mira',
            'test', 'prueba', 'asdf', 'qwerty'
        ]

    def _is_trivial(self, text: str) -> bool:
        """
        Detecta si un texto es trivial (bajo valor semántico).
        
        Criterios:
        - Muy corto (< 10 caracteres)
        - Coincide con patrones triviales conocidos
        - Solo emojis o símbolos
        """
        text_clean = text.strip().lower()
        
        # Muy corto
        if len(text_clean) < 10:
            return True
        
        # Coincide con patrones triviales
        for pattern in self.trivial_patterns:
            if text_clean == pattern or text_clean.startswith(pattern + ' '):
                return True
        
        # Pocas palabras únicas (repetición)
        words = text_clean.split()
        if len(words) < 3:
            return True
        
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Muy repetitivo
            return True
            
        return False

    def _calculate_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Coseno similitud entre dos vectores numpy normalizados."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def synthesize_concepts(self, text_a: str, text_b: str) -> str:
        """
        Pide al modelo (LLM) que fusione dos textos en una verdad superior.
        Usa un prompt de 'Reflexión' con temperatura baja para síntesis precisa.
        
        Args:
            text_a: Primer concepto
            text_b: Segundo concepto
            
        Returns:
            Texto sintetizado que combina ambos conceptos
        """
        prompt = f"""Tarea: Sintetizar y abstraer conocimiento.

Concepto A: {text_a}

Concepto B: {text_b}

Instrucción: Genera un único párrafo denso que combine la información esencial de A y B, eliminando redundancias y elevando el nivel técnico. Mantén solo la información más importante.

Síntesis:"""
        
        # Generación determinista (baja temperatura) para síntesis precisa
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.gpt2.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.3,  # Frío y analítico
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extraer solo la síntesis
        if "Síntesis:" in full_text:
            synthesis = full_text.split("Síntesis:")[-1].strip()
            # Limpiar posibles artefactos
            synthesis = synthesis.split("\n\n")[0].strip()
            return synthesis
        
        return full_text.strip()

    def measure_phi(self, text: str) -> float:
        """
        Mide la densidad de información (Phi) de un texto.
        
        Args:
            text: Texto a medir
            
        Returns:
            Valor de PHI (float)
        """
        _, phi_metrics = self.extractor.extract_with_phi(text)
        return phi_metrics['phi']

    def get_memory_snapshot(self) -> Dict:
        """
        Obtiene un snapshot completo de la memoria actual.
        
        Returns:
            Dict con ids, documents, embeddings, metadatas
        """
        try:
            data = self.db.collection.get(
                include=['embeddings', 'documents', 'metadatas']
            )
            return data
        except Exception as e:
            print(f"⚠️ Error obteniendo snapshot: {e}")
            return {'ids': [], 'documents': [], 'embeddings': [], 'metadatas': []}

    def sleep_cycle(self, max_steps: int = 5, verbose: bool = True) -> List[str]:
        """
        Ejecuta un ciclo de mantenimiento cognitivo ("sueño").
        
        El algoritmo:
        1. Obtiene snapshot de toda la memoria
        2. Baraja aleatoriamente los recuerdos (como sueño REM)
        3. Para cada par cercano:
           - Si similitud > umbral_merge → intenta fusionar
           - Si fusión mejora PHI > umbral → consolida
           - Si PHI muy bajo → poda (olvido activo)
        
        Args:
            max_steps: Número máximo de operaciones (merge/prune)
            verbose: Si imprimir logs durante el proceso
            
        Returns:
            Lista de logs describiendo las acciones tomadas
        """
        logs = []
        self.stats['total_cycles'] += 1
        
        if verbose:
            print(f"\n🌙 Iniciando ciclo de sueño #{self.stats['total_cycles']}...")
        
        # 1. Obtener snapshot de la memoria
        data = self.get_memory_snapshot()
        
        ids = data['ids']
        if len(ids) < 2:
            msg = "💤 Memoria insuficiente para soñar (se necesitan +2 recuerdos)."
            logs.append(msg)
            return logs

        embeddings = np.array(data['embeddings'])
        documents = data['documents']
        metadatas = data['metadatas'] or [{}] * len(ids)
        
        if verbose:
            print(f"   📚 Memorias encontradas: {len(ids)}")
            print(f"   📋 Documentos en memoria:")
            for idx, doc in enumerate(documents):
                phi = metadatas[idx].get('phi', 'N/A') if metadatas[idx] else 'N/A'
                is_triv = self._is_trivial(doc)
                print(f"      [{idx}] \"{doc[:40]}{'...' if len(doc)>40 else ''}\" | Φ={phi} | trivial={is_triv}")
        
        step_count = 0
        processed_ids = set()
        
        # 2. Baraja aleatoria (como sueño REM)
        indices = list(range(len(ids)))
        np.random.shuffle(indices)
        
        if verbose:
            print(f"\n   🎲 Orden aleatorio de procesamiento: {indices[:10]}{'...' if len(indices)>10 else ''}")

        for i in range(len(indices) - 1):
            if step_count >= max_steps:
                if verbose:
                    print(f"   ⏹️ Límite de operaciones alcanzado ({max_steps})")
                break
                
            idx_a = indices[i]
            id_a = ids[idx_a]
            doc_a = documents[idx_a]
            
            if id_a in processed_ids:
                continue
            
            if verbose:
                print(f"\n   --- Evaluando documento [{idx_a}]: \"{doc_a[:30]}...\" ---")

            # Buscar el vecino más cercano en este batch aleatorio
            best_sim = -1
            best_idx_b = -1
            
            for j in range(i + 1, min(i + 10, len(indices))):  # Limitar búsqueda local
                idx_b = indices[j]
                id_b = ids[idx_b]
                if id_b in processed_ids:
                    continue
                
                sim = self._calculate_similarity(embeddings[idx_a], embeddings[idx_b])
                if sim > best_sim:
                    best_sim = sim
                    best_idx_b = idx_b
            
            if verbose:
                print(f"      Mejor vecino: [{best_idx_b}] con similitud={best_sim:.3f}")
            
            if best_idx_b == -1:
                if verbose:
                    print(f"      ❌ Sin vecino válido, saltando")
                continue
                
            # --- LÓGICA DE DECISIÓN DEL AGENTE ---
            
            if verbose:
                print(f"      📊 Umbrales: merge>{self.similarity_threshold_merge:.2f}, prune<{self.prune_threshold:.2f}")
            
            # CASO 0: PRUNE PRIORITARIO - Eliminar trivialidades ANTES de considerar merge
            is_trivial_a = self._is_trivial(doc_a)
            if is_trivial_a:
                if verbose:
                    print(f"      🚨 TRIVIALIDAD DETECTADA - Evaluando eliminación prioritaria")
                    print(f"         - Doc: \"{doc_a}\"")
                    print(f"         - is_trivial: {is_trivial_a}")
                
                try:
                    self.db.collection.delete(ids=[id_a])
                    processed_ids.add(id_a)
                    self.stats['total_prunes'] += 1
                    
                    log_msg = (
                        f"🗑️ OLVIDO ACTIVO (Prioritario):\n"
                        f"   Recuerdo: \"{doc_a[:50]}{'...' if len(doc_a) > 50 else ''}\"\n"
                        f"   Razón: trivialidad detectada"
                    )
                    logs.append(log_msg)
                    step_count += 1
                    
                    if verbose:
                        print(f"      ✅ ELIMINADO por trivialidad")
                except Exception as e:
                    if verbose:
                        print(f"      ⚠️ Error eliminando: {e}")
                continue  # Saltar al siguiente documento
            
            # CASO 1: MERGE (Alta redundancia detectada)
            if best_sim > self.similarity_threshold_merge:
                idx_b = best_idx_b
                id_b = ids[idx_b]
                
                doc_b = documents[idx_b]
                
                # Obtener PHI de los metadatos o calcularlo
                phi_a = metadatas[idx_a].get('phi', self.measure_phi(doc_a)) if metadatas[idx_a] else self.measure_phi(doc_a)
                phi_b = metadatas[idx_b].get('phi', self.measure_phi(doc_b)) if metadatas[idx_b] else self.measure_phi(doc_b)
                
                if verbose:
                    print(f"\n   🔍 Candidatos a fusión (Sim: {best_sim:.3f}):")
                    print(f"      A: {doc_a[:50]}...")
                    print(f"      B: {doc_b[:50]}...")
                
                # Intentar sintetizar
                try:
                    synthesis = self.synthesize_concepts(doc_a, doc_b)
                    new_phi = self.measure_phi(synthesis)
                except Exception as e:
                    logs.append(f"⚠️ Error en síntesis: {e}")
                    continue
                
                # Criterio de Aceptación: ¿Es la síntesis mejor que las partes?
                max_prev_phi = max(phi_a, phi_b)
                phi_improvement = (new_phi - max_prev_phi) / max(max_prev_phi, 0.001)
                
                if phi_improvement > self.phi_improvement_threshold:
                    # ¡ÉXITO! Consolidación de memoria
                    try:
                        # Añadir nuevo documento consolidado
                        new_emb, new_phi_metrics = self.extractor.extract_with_phi(synthesis)
                        
                        # Crear ID único para el documento consolidado
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        new_id = f"consolidated_{timestamp}"
                        
                        # Combinar categorías de los originales
                        cat_a = metadatas[idx_a].get('category', 'general') if metadatas[idx_a] else 'general'
                        cat_b = metadatas[idx_b].get('category', 'general') if metadatas[idx_b] else 'general'
                        new_category = cat_a if cat_a == cat_b else f"{cat_a}+{cat_b}"
                        
                        # Añadir documento consolidado
                        self.db.collection.add(
                            embeddings=[new_emb.tolist()],
                            documents=[synthesis],
                            metadatas=[{
                                'phi': new_phi,
                                'category': new_category,
                                'consolidated_from': f"{id_a},{id_b}",
                                'consolidated_at': datetime.now().isoformat(),
                                'phi_gain': new_phi - max_prev_phi
                            }],
                            ids=[new_id]
                        )
                        
                        # Eliminar documentos originales
                        self.db.collection.delete(ids=[id_a, id_b])
                        
                        processed_ids.add(id_a)
                        processed_ids.add(id_b)
                        
                        # Actualizar estadísticas
                        self.stats['total_merges'] += 1
                        self.stats['phi_gained'] += (new_phi - max_prev_phi)
                        
                        log_msg = (
                            f"✨ FUSIÓN EXITOSA (Sim: {best_sim:.2f}):\n"
                            f"   A: {doc_a[:40]}...\n"
                            f"   B: {doc_b[:40]}...\n"
                            f"   → Síntesis: {synthesis[:60]}...\n"
                            f"   → Nuevo Φ: {new_phi:.3f} (+{(new_phi-max_prev_phi):.3f})"
                        )
                        logs.append(log_msg)
                        step_count += 1
                        
                        if verbose:
                            print(f"   ✨ ¡Fusión exitosa! Φ: {max_prev_phi:.3f} → {new_phi:.3f}")
                            
                    except Exception as e:
                        logs.append(f"⚠️ Error guardando fusión: {e}")
                        
                else:
                    log_msg = f"📉 Fusión descartada (Φ {new_phi:.3f} no mejora suficiente sobre {max_prev_phi:.3f})"
                    logs.append(log_msg)
                    if verbose:
                        print(f"   📉 Fusión descartada (mejora insuficiente)")

            # CASO 2: PRUNE (Limpieza de ruido) - solo si no hubo fusión
            elif best_sim < 0.5:  # No muy similar a nada → posible ruido
                phi_a = metadatas[idx_a].get('phi', 0) if metadatas[idx_a] else 0
                
                # Detectar trivialidades por longitud y contenido
                is_trivial = self._is_trivial(doc_a)
                
                if verbose:
                    print(f"      🔍 EVALUANDO PRUNE:")
                    print(f"         - Doc: \"{doc_a}\"")
                    print(f"         - is_trivial: {is_trivial}")
                    print(f"         - phi_a: {phi_a}")
                    print(f"         - prune_threshold: {self.prune_threshold}")
                    print(f"         - Condición trivial: {is_trivial}")
                    print(f"         - Condición phi bajo: {phi_a < self.prune_threshold and phi_a > 0}")
                
                if is_trivial or (phi_a < self.prune_threshold and phi_a > 0):
                    if verbose:
                        print(f"      ✅ SERÁ ELIMINADO")
                    try:
                        self.db.collection.delete(ids=[id_a])
                        processed_ids.add(id_a)
                        
                        self.stats['total_prunes'] += 1
                        
                        reason = "trivialidad detectada" if is_trivial else f"bajo Φ ({phi_a:.3f})"
                        log_msg = (
                            f"🗑️ OLVIDO ACTIVO:\n"
                            f"   Recuerdo: \"{doc_a[:50]}{'...' if len(doc_a) > 50 else ''}\"\n"
                            f"   Razón: {reason}"
                        )
                        logs.append(log_msg)
                        step_count += 1
                        
                        if verbose:
                            print(f"   🗑️ Recuerdo eliminado - Razón: {reason}")
                            
                    except Exception as e:
                        logs.append(f"⚠️ Error en poda: {e}")
                else:
                    if verbose:
                        print(f"      ❌ NO ELIMINADO - No cumple criterios de poda")
            
            # CASO 3: Ni merge ni prune - similitud intermedia
            else:
                if verbose:
                    print(f"      ⏭️ SALTADO - Similitud intermedia ({best_sim:.2f}), no merge ni prune")

        # Resumen final
        if not logs:
            logs.append("💤 Sueño tranquilo. No se requirieron cambios estructurales.")
        
        if verbose:
            print(f"\n   📊 Ciclo completado:")
            print(f"      - Fusiones: {self.stats['total_merges']}")
            print(f"      - Podas: {self.stats['total_prunes']}")
            print(f"      - Φ ganado total: {self.stats['phi_gained']:.3f}")
            
        return logs

    def get_stats(self) -> Dict:
        """Retorna estadísticas del agente."""
        return self.stats.copy()
    
    def analyze_memory_health(self) -> Dict:
        """
        Analiza la salud general de la memoria.
        
        Returns:
            Dict con métricas de salud:
            - total_memories: Número total de recuerdos
            - avg_phi: PHI promedio
            - entropy: Entropía estimada (dispersión de similitudes)
            - redundancy_score: Proporción de pares muy similares
            - low_phi_count: Recuerdos con PHI bajo
        """
        data = self.get_memory_snapshot()
        
        if len(data['ids']) == 0:
            return {
                'total_memories': 0,
                'avg_phi': 0,
                'entropy': 0,
                'redundancy_score': 0,
                'low_phi_count': 0
            }
        
        embeddings = np.array(data['embeddings'])
        metadatas = data['metadatas'] or [{}] * len(data['ids'])
        
        # PHI promedio
        phis = [m.get('phi', 0) for m in metadatas if m]
        avg_phi = np.mean(phis) if phis else 0
        
        # Contar PHI bajo
        low_phi_count = sum(1 for p in phis if p < self.prune_threshold)
        
        # Calcular redundancia (pares con alta similitud)
        n = len(embeddings)
        high_sim_pairs = 0
        total_pairs = 0
        similarities = []
        
        for i in range(min(n, 50)):  # Limitar para rendimiento
            for j in range(i + 1, min(n, 50)):
                sim = self._calculate_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
                total_pairs += 1
                if sim > self.similarity_threshold_merge:
                    high_sim_pairs += 1
        
        redundancy_score = high_sim_pairs / max(total_pairs, 1)
        
        # Entropía aproximada (varianza de similitudes)
        entropy = np.std(similarities) if similarities else 0
        
        return {
            'total_memories': n,
            'avg_phi': float(avg_phi),
            'entropy': float(entropy),
            'redundancy_score': float(redundancy_score),
            'low_phi_count': low_phi_count,
            'health_score': min(1.0, avg_phi * (1 - redundancy_score))
        }


# --- FUNCIONES DE UTILIDAD ---

def run_curator_demo():
    """Demo standalone del curator."""
    import os
    import sys
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from src.core.layer4_embeddings import Layer4Extractor
    from src.core.conscious_vectordb import ConsciousVectorDB
    
    print("🧠 INFINITO Knowledge Curator - Demo")
    print("=" * 50)
    
    # Crear DB de prueba
    db = ConsciousVectorDB(
        collection_name="curator_demo",
        persist_directory="data/chromadb_curator_demo",
        force_recreate=True
    )
    
    # Añadir documentos de prueba (algunos redundantes)
    test_docs = [
        "El cielo es azul debido a la dispersión de Rayleigh.",
        "El color azul del cielo se debe a la dispersión de la luz solar.",
        "Los transformers usan mecanismos de atención para procesar secuencias.",
        "La arquitectura transformer emplea atención multi-cabeza para procesar texto.",
        "PHI mide la integración de información en sistemas complejos.",
        "Hola mundo",  # Bajo valor semántico
    ]
    
    print("\n📚 Añadiendo documentos de prueba...")
    for doc in test_docs:
        db.add_documents([doc], metadatas=[{'category': 'test'}])
    
    print(f"   Total: {db.collection.count()} documentos")
    
    # Crear curator
    curator = KnowledgeCurator(
        model=db.model,
        extractor=db.extractor,
        vectordb=db
    )
    
    # Analizar salud
    print("\n📊 Salud de la memoria ANTES:")
    health = curator.analyze_memory_health()
    for k, v in health.items():
        print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")
    
    # Ejecutar ciclo de sueño
    print("\n" + "=" * 50)
    logs = curator.sleep_cycle(max_steps=5, verbose=True)
    print("=" * 50)
    
    # Mostrar logs
    print("\n📝 Diario de sueño:")
    for log in logs:
        print(f"   {log}")
    
    # Analizar salud después
    print("\n📊 Salud de la memoria DESPUÉS:")
    health = curator.analyze_memory_health()
    for k, v in health.items():
        print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")
    
    print(f"\n✅ Demo completada. Documentos finales: {db.collection.count()}")


if __name__ == "__main__":
    run_curator_demo()
