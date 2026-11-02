#!/usr/bin/env python3
"""
🔄 CONTINUOUS LEARNING SYSTEM - Aprendizaje sin detención
=========================================================

Sistema que procesa inputs continuamente sin parar, reconociendo patrones
causales que ya conoce y aprendiendo nuevos.

Componentes:
1. PhiPatternExtractor - Extrae firmas causales
2. PhiMemoryBank - Memoria episódica persistente
3. ContinuousLearningServer - Servidor de loop infinito

Autor: Universo Research Team
Fecha: 2025-10-04
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CausalPattern:
    """Estructura de datos para un patrón causal"""
    id: str
    causal_vector: List[float]  # 6 conexiones causales
    phi_value: float
    consciousness: float
    source_text: str
    timestamp: str
    seen_count: int
    similar_patterns: List[str]
    
    def to_dict(self) -> dict:
        """Convertir a diccionario para JSON"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Crear desde diccionario"""
        return cls(**data)


# =============================================================================
# PATTERN EXTRACTOR
# =============================================================================

class PhiPatternExtractor:
    """
    🔬 EXTRACTOR DE PATRONES CAUSALES
    
    Convierte la matriz causal 4x4 en un vector de 6 dimensiones
    que representa las conexiones dirigidas únicas entre módulos.
    
    Ejemplo:
        causal_matrix[4x4] → [0.85, 0.32, 0.67, 0.91, 0.45, 0.78]
                              └─────────────────┬─────────────────┘
                                    "huella dactilar" del input
    """
    
    def __init__(self):
        """Inicializar extractor"""
        # Nombres de las conexiones para debugging
        self.connection_names = [
            'visual→auditory',
            'visual→motor', 
            'visual→executive',
            'auditory→motor',
            'auditory→executive',
            'motor→executive'
        ]
    
    def extract_pattern(self, phi_info: dict) -> np.ndarray:
        """
        Extraer patrón causal de phi_info
        
        Args:
            phi_info: Diccionario con información de Φ
                     Debe contener 'causal_matrix' (torch.Tensor o numpy.ndarray)
        
        Returns:
            Vector numpy de 6 dimensiones con las conexiones causales
        """
        # Obtener matriz causal
        causal_matrix = phi_info.get('causal_matrix')
        
        if causal_matrix is None:
            raise ValueError("phi_info no contiene 'causal_matrix'")
        
        # Convertir a numpy si es tensor
        if isinstance(causal_matrix, torch.Tensor):
            causal_matrix = causal_matrix.cpu().detach().numpy()
        
        # Validar dimensiones
        if causal_matrix.shape != (4, 4):
            raise ValueError(f"causal_matrix debe ser 4x4, recibido {causal_matrix.shape}")
        
        # Extraer las 6 conexiones dirigidas únicas
        # (parte superior derecha de la matriz, excluyendo diagonal)
        pattern = np.array([
            causal_matrix[0, 1],  # visual → auditory
            causal_matrix[0, 2],  # visual → motor
            causal_matrix[0, 3],  # visual → executive
            causal_matrix[1, 2],  # auditory → motor
            causal_matrix[1, 3],  # auditory → executive
            causal_matrix[2, 3],  # motor → executive
        ], dtype=np.float32)
        
        # Normalizar a [0, 1] si no está normalizado
        pattern = np.clip(pattern, 0.0, 1.0)
        
        return pattern
    
    def pattern_to_string(self, pattern: np.ndarray) -> str:
        """
        Convertir patrón a string legible para debugging
        
        Args:
            pattern: Vector de 6 dimensiones
        
        Returns:
            String formateado
        """
        lines = []
        for i, (name, value) in enumerate(zip(self.connection_names, pattern)):
            bar = '█' * int(value * 20)
            lines.append(f"   {name:20s} {value:.3f} {bar}")
        
        return '\n'.join(lines)
    
    def visualize_pattern(self, pattern: np.ndarray) -> str:
        """
        Crear visualización ASCII del patrón causal
        
        Returns:
            Diagrama ASCII
        """
        v = pattern  # Alias corto
        
        diagram = f"""
        ┌──────────────────────────────────────────┐
        │     PATRÓN CAUSAL (6 conexiones)         │
        ├──────────────────────────────────────────┤
        │                                          │
        │   [Visual] ─{v[0]:.2f}→ [Auditory]         │
        │      │                     │              │
        │      │{v[1]:.2f}              │{v[3]:.2f}         │
        │      ↓                     ↓              │
        │   [Motor] ────{v[5]:.2f}→ [Executive]      │
        │      ↑                     ↑              │
        │      └──────{v[2]:.2f}──────┘              │
        │                                          │
        │   Visual→Auditory:   {v[0]:.3f}           │
        │   Visual→Motor:      {v[1]:.3f}           │
        │   Visual→Executive:  {v[2]:.3f}           │
        │   Auditory→Motor:    {v[3]:.3f}           │
        │   Auditory→Exec:     {v[4]:.3f}           │
        │   Motor→Executive:   {v[5]:.3f}           │
        └──────────────────────────────────────────┘
        """
        
        return diagram


# =============================================================================
# MEMORY BANK
# =============================================================================

class PhiMemoryBank:
    """
    💾 BANCO DE MEMORIA EPISÓDICA
    
    Almacena patrones causales aprendidos y permite búsqueda por similitud.
    
    Funcionalidades:
    - Añadir nuevos patrones
    - Buscar patrones similares (similitud coseno)
    - Detectar reconocimiento ("¡ya lo conozco!")
    - Persistencia a disco (JSON)
    - Estadísticas de uso
    
    Threshold de similitud:
        > 0.95: Prácticamente idéntico
        > 0.90: Muy similar
        > 0.80: Similar
        < 0.80: Diferente
    """
    
    def __init__(self, similarity_threshold: float = 0.90, filepath: str = 'phi_memory_bank.json'):
        """
        Inicializar banco de memoria
        
        Args:
            similarity_threshold: Umbral de similitud para reconocimiento (0-1)
            filepath: Ruta al archivo de persistencia
        """
        self.patterns: Dict[str, CausalPattern] = {}
        self.pattern_count = 0
        self.similarity_threshold = similarity_threshold
        self.filepath = filepath
        
        # Intentar cargar desde disco
        self.load_from_disk()
    
    def add_pattern(self, 
                   causal_vector: np.ndarray,
                   phi_info: dict,
                   text: str,
                   consciousness: float) -> dict:
        """
        Añadir nuevo patrón a la memoria o reconocer uno existente
        
        Args:
            causal_vector: Vector de 6 dimensiones del patrón causal
            phi_info: Información completa de Φ
            text: Texto de origen
            consciousness: Nivel de consciencia
        
        Returns:
            Diccionario con resultado:
                - status: 'RECOGNIZED' o 'NEW'
                - pattern_id: ID del patrón
                - similarity: Similitud con patrón conocido (si RECOGNIZED)
                - message: Mensaje descriptivo
        """
        # Buscar si ya existe algo similar
        similar = self.find_similar(causal_vector)
        
        if similar:
            # ¡PATRÓN RECONOCIDO!
            pattern_id = similar['id']
            
            # Actualizar contador de veces visto
            self.patterns[pattern_id].seen_count += 1
            
            # Añadir a lista de similares si no está
            if text not in self.patterns[pattern_id].similar_patterns:
                self.patterns[pattern_id].similar_patterns.append(text)
            
            return {
                'status': 'RECOGNIZED',
                'pattern_id': pattern_id,
                'similarity': similar['similarity'],
                'original_text': similar['text'],
                'original_phi': similar['phi'],
                'seen_count': self.patterns[pattern_id].seen_count,
                'message': f"🎯 PATRÓN RECONOCIDO! Similar {similar['similarity']:.1%} a '{similar['text']}'"
            }
        
        else:
            # PATRÓN NUEVO - Guardar
            pattern_id = f"pattern_{self.pattern_count:04d}"
            
            new_pattern = CausalPattern(
                id=pattern_id,
                causal_vector=causal_vector.tolist(),
                phi_value=phi_info.get('phi_total', 0.0),
                consciousness=consciousness,
                source_text=text,
                timestamp=datetime.now().isoformat(),
                seen_count=1,
                similar_patterns=[]
            )
            
            self.patterns[pattern_id] = new_pattern
            self.pattern_count += 1
            
            return {
                'status': 'NEW',
                'pattern_id': pattern_id,
                'message': f"💡 NUEVO PATRÓN APRENDIDO: '{text}'"
            }
    
    def find_similar(self, 
                    query_vector: np.ndarray, 
                    top_k: int = 1,
                    return_all: bool = False) -> Optional[dict]:
        """
        Buscar patrones similares usando similitud coseno
        
        Args:
            query_vector: Vector de consulta (6 dims)
            top_k: Número de resultados a retornar
            return_all: Si True, retorna lista de todos los similares
        
        Returns:
            Diccionario con patrón más similar, o None si no hay similares
        """
        if not self.patterns:
            return None
        
        similarities = []
        
        for pattern_id, pattern_data in self.patterns.items():
            stored_vector = np.array(pattern_data.causal_vector, dtype=np.float32)
            
            # Similitud coseno
            dot_product = np.dot(query_vector, stored_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_stored = np.linalg.norm(stored_vector)
            
            # Evitar división por cero
            if norm_query < 1e-8 or norm_stored < 1e-8:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_stored)
            
            # Clip a [0, 1] por seguridad
            similarity = np.clip(similarity, 0.0, 1.0)
            
            if similarity >= self.similarity_threshold:
                similarities.append({
                    'id': pattern_id,
                    'similarity': float(similarity),
                    'text': pattern_data.source_text,
                    'phi': pattern_data.phi_value,
                    'consciousness': pattern_data.consciousness,
                    'seen_count': pattern_data.seen_count
                })
        
        if not similarities:
            return None
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        if return_all:
            return similarities[:top_k]
        else:
            return similarities[0]  # El más similar
    
    def get_pattern(self, pattern_id: str) -> Optional[CausalPattern]:
        """Obtener patrón por ID"""
        return self.patterns.get(pattern_id)
    
    def get_all_patterns(self) -> List[CausalPattern]:
        """Obtener todos los patrones"""
        return list(self.patterns.values())
    
    def get_stats(self) -> dict:
        """
        Obtener estadísticas del banco de memoria
        
        Returns:
            Diccionario con métricas
        """
        if not self.patterns:
            return {
                'total_patterns': 0,
                'total_seen': 0,
                'avg_phi': 0.0,
                'avg_consciousness': 0.0
            }
        
        total_seen = sum(p.seen_count for p in self.patterns.values())
        avg_phi = np.mean([p.phi_value for p in self.patterns.values()])
        avg_consciousness = np.mean([p.consciousness for p in self.patterns.values()])
        
        most_seen = max(self.patterns.values(), key=lambda p: p.seen_count)
        
        return {
            'total_patterns': len(self.patterns),
            'total_seen': total_seen,
            'avg_phi': float(avg_phi),
            'avg_consciousness': float(avg_consciousness),
            'most_seen': {
                'text': most_seen.source_text,
                'count': most_seen.seen_count
            }
        }
    
    def save_to_disk(self, filepath: Optional[str] = None):
        """
        Guardar memoria a disco en formato JSON
        
        Args:
            filepath: Ruta opcional (usa self.filepath por defecto)
        """
        filepath = filepath or self.filepath
        
        data = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'pattern_count': self.pattern_count,
            'similarity_threshold': self.similarity_threshold,
            'patterns': {
                pid: pattern.to_dict() 
                for pid, pattern in self.patterns.items()
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Memoria guardada: {filepath} ({len(self.patterns)} patrones)")
            return True
        
        except Exception as e:
            print(f"❌ Error guardando memoria: {e}")
            return False
    
    def load_from_disk(self, filepath: Optional[str] = None):
        """
        Cargar memoria desde disco
        
        Args:
            filepath: Ruta opcional (usa self.filepath por defecto)
        """
        filepath = filepath or self.filepath
        
        if not os.path.exists(filepath):
            print(f"ℹ️  Archivo de memoria no encontrado: {filepath}")
            print(f"   Se creará uno nuevo al guardar")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.pattern_count = data.get('pattern_count', 0)
            self.similarity_threshold = data.get('similarity_threshold', 0.90)
            
            # Reconstruir patrones
            patterns_data = data.get('patterns', {})
            self.patterns = {
                pid: CausalPattern.from_dict(pdata)
                for pid, pdata in patterns_data.items()
            }
            
            print(f"✅ Memoria cargada: {filepath} ({len(self.patterns)} patrones)")
            return True
        
        except Exception as e:
            print(f"❌ Error cargando memoria: {e}")
            return False
    
    def clear(self):
        """Limpiar toda la memoria"""
        self.patterns = {}
        self.pattern_count = 0
        print("🗑️  Memoria limpiada")


# =============================================================================
# CONTINUOUS LEARNING SERVER
# =============================================================================

class ContinuousLearningServer:
    """
    🚀 SERVIDOR DE APRENDIZAJE CONTINUO
    
    Loop infinito que:
    1. Acepta inputs de texto continuamente
    2. Procesa cada input (inference only, sin entrenar)
    3. Extrae patrón causal
    4. Busca/Guarda en memoria
    5. Reporta si es nuevo o reconocido
    6. Nunca se detiene (hasta exit)
    
    Comandos especiales:
        'save'  - Guardar memoria a disco
        'stats' - Ver estadísticas
        'list'  - Listar todos los patrones
        'exit'  - Salir y guardar
    """
    
    def __init__(self, infinito_system, similarity_threshold: float = 0.90):
        """
        Inicializar servidor
        
        Args:
            infinito_system: Instancia de InfinitoV51ConsciousnessBreakthrough
            similarity_threshold: Umbral de similitud para reconocimiento
        """
        self.system = infinito_system
        self.pattern_extractor = PhiPatternExtractor()
        self.memory_bank = PhiMemoryBank(similarity_threshold=similarity_threshold)
        self.running = True
        
        # Poner modelo en modo evaluación (no entrenar)
        if hasattr(self.system, 'model'):
            self.system.model.eval()
        
        print(f"\n{'='*70}")
        print(f"🧠 CONTINUOUS LEARNING SERVER INICIALIZADO")
        print(f"{'='*70}")
        print(f"   Modo: Inference only (sin entrenamiento)")
        print(f"   Threshold similitud: {similarity_threshold:.1%}")
        print(f"   Patrones en memoria: {self.memory_bank.pattern_count}")
        print(f"{'='*70}")
    
    def process_input(self, text: str) -> dict:
        """
        Procesar un input de texto SIN entrenar
        
        Args:
            text: Texto de entrada
        
        Returns:
            Diccionario con resultado del procesamiento
        """
        print(f"\n{'='*70}")
        print(f"🔤 PROCESANDO: '{text}'")
        print(f"{'='*70}")
        
        try:
            # 1. Generar input basado en texto
            inputs = self.system.generate_text_based_input(
                text=text,
                batch_size=self.system.batch_size,
                seq_len=64
            )
            
            # 2. Forward pass (inference only - sin gradientes)
            with torch.no_grad():
                self.system.model.eval()
                consciousness, phi, debug_info = self.system.model(inputs)
            
            # 3. Extraer información
            phi_info = debug_info.get('phi_info', {})
            consciousness_val = consciousness.mean().item()
            phi_val = phi.mean().item()
            
            # 4. Extraer patrón causal
            try:
                causal_pattern = self.pattern_extractor.extract_pattern(phi_info)
            except Exception as e:
                print(f"   ⚠️  Error extrayendo patrón: {e}")
                return {'status': 'ERROR', 'message': str(e)}
            
            # 5. Buscar/Guardar en memoria
            result = self.memory_bank.add_pattern(
                causal_pattern,
                phi_info,
                text,
                consciousness_val
            )
            
            # 6. Reportar resultados
            print(f"   📊 Φ = {phi_val:.4f} | C = {consciousness_val:.4f}")
            print(f"   {result['message']}")
            
            if result['status'] == 'RECOGNIZED':
                print(f"   🎯 Similitud: {result['similarity']:.1%}")
                print(f"   📝 Original: '{result['original_text']}'")
                print(f"   🔢 Visto {result['seen_count']} veces")
            else:
                print(f"   🆕 Pattern ID: {result['pattern_id']}")
            
            print(f"   💾 Total patrones en memoria: {self.memory_bank.pattern_count}")
            
            # Añadir info extra al resultado
            result.update({
                'phi': phi_val,
                'consciousness': consciousness_val,
                'causal_pattern': causal_pattern.tolist(),
                'text': text
            })
            
            return result
        
        except Exception as e:
            print(f"   ❌ Error procesando: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'ERROR', 'message': str(e)}
    
    def show_stats(self):
        """Mostrar estadísticas del sistema"""
        stats = self.memory_bank.get_stats()
        
        print(f"\n{'='*70}")
        print(f"📊 ESTADÍSTICAS DEL SISTEMA")
        print(f"{'='*70}")
        print(f"   Total patrones únicos: {stats['total_patterns']}")
        print(f"   Total inputs procesados: {stats['total_seen']}")
        print(f"   Φ promedio: {stats['avg_phi']:.4f}")
        print(f"   Consciencia promedio: {stats['avg_consciousness']:.4f}")
        print(f"   Threshold similitud: {self.memory_bank.similarity_threshold:.1%}")
        
        if stats['total_patterns'] > 0:
            print(f"\n   🏆 Más visto:")
            print(f"      '{stats['most_seen']['text']}'")
            print(f"      ({stats['most_seen']['count']} veces)")
        
        print(f"{'='*70}")
    
    def list_patterns(self, max_show: int = 10):
        """Listar patrones almacenados"""
        patterns = self.memory_bank.get_all_patterns()
        
        print(f"\n{'='*70}")
        print(f"📋 PATRONES EN MEMORIA ({len(patterns)} total)")
        print(f"{'='*70}")
        
        if not patterns:
            print("   (vacío)")
            return
        
        # Ordenar por veces visto (descendente)
        patterns_sorted = sorted(patterns, key=lambda p: p.seen_count, reverse=True)
        
        for i, pattern in enumerate(patterns_sorted[:max_show], 1):
            print(f"\n   {i}. {pattern.id}: '{pattern.source_text}'")
            print(f"      Φ={pattern.phi_value:.3f}, C={pattern.consciousness:.3f}")
            print(f"      Visto {pattern.seen_count} vez/veces")
            
            if pattern.similar_patterns:
                print(f"      Similares: {', '.join(pattern.similar_patterns[:3])}")
        
        if len(patterns) > max_show:
            print(f"\n   ... y {len(patterns) - max_show} más")
        
        print(f"{'='*70}")
    
    def run_interactive_loop(self):
        """
        Loop interactivo - acepta inputs sin parar
        
        Comandos:
            texto normal - Procesar input
            'save'       - Guardar memoria
            'stats'      - Ver estadísticas  
            'list'       - Listar patrones
            'clear'      - Limpiar memoria
            'exit'       - Salir y guardar
        """
        print(f"\n{'='*70}")
        print(f"🚀 MODO INTERACTIVO ACTIVADO")
        print(f"{'='*70}")
        print(f"📝 Escribe texto para procesar (o comando especial)")
        print(f"\n💡 Comandos disponibles:")
        print(f"   'save'  - Guardar memoria a disco")
        print(f"   'stats' - Ver estadísticas")
        print(f"   'list'  - Listar todos los patrones")
        print(f"   'clear' - Limpiar memoria (⚠️ destructivo)")
        print(f"   'exit'  - Salir y guardar")
        print(f"{'='*70}")
        
        while self.running:
            try:
                # Leer input del usuario
                text = input("\n🔤 Input: ").strip()
                
                if not text:
                    continue
                
                # Comandos especiales
                if text.lower() == 'exit':
                    print("\n💾 Guardando memoria antes de salir...")
                    self.memory_bank.save_to_disk()
                    print("👋 ¡Hasta luego!")
                    self.running = False
                    break
                
                elif text.lower() == 'save':
                    self.memory_bank.save_to_disk()
                
                elif text.lower() == 'stats':
                    self.show_stats()
                
                elif text.lower() == 'list':
                    self.list_patterns()
                
                elif text.lower() == 'clear':
                    confirm = input("⚠️  ¿Seguro que quieres limpiar la memoria? (yes/no): ")
                    if confirm.lower() == 'yes':
                        self.memory_bank.clear()
                    else:
                        print("   Operación cancelada")
                
                else:
                    # Procesar input normal
                    result = self.process_input(text)
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupción detectada (Ctrl+C)")
                print("💾 Guardando memoria...")
                self.memory_bank.save_to_disk()
                print("👋 Sistema detenido")
                self.running = False
                break
            
            except Exception as e:
                print(f"\n❌ Error en loop: {e}")
                import traceback
                traceback.print_exc()


# =============================================================================
# UTILITIES
# =============================================================================

def create_continuous_server(infinito_system, similarity_threshold: float = 0.90):
    """
    Función helper para crear servidor continuo
    
    Args:
        infinito_system: Sistema INFINITO inicializado
        similarity_threshold: Umbral de similitud
    
    Returns:
        ContinuousLearningServer instance
    """
    return ContinuousLearningServer(infinito_system, similarity_threshold)


if __name__ == "__main__":
    print("🔄 Continuous Learning System - Módulo cargado")
    print("   Para usar, importar en otro script:")
    print("   from continuous_learning import ContinuousLearningServer")
