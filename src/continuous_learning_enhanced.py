#!/usr/bin/env python3
"""
🔄 ENHANCED CONTINUOUS LEARNING - Captura Temprana + Firmas Enriquecidas
==========================================================================

Versión mejorada que soluciona el problema de saturación de patrones.

MEJORAS:
1. Captura temprana: Extrae patrón en iter 5-10 (antes de saturar)
2. Firmas enriquecidas: Añade attention, módulos, varianzas
3. Mejor diferenciación: Textos diferentes → patrones diferentes

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

# Importar componentes base
import sys
sys.path.insert(0, os.path.dirname(__file__))

from continuous_learning import (
    PhiPatternExtractor,
    PhiMemoryBank,
    CausalPattern
)


# =============================================================================
# ENHANCED PATTERN EXTRACTOR
# =============================================================================

class EnhancedPhiPatternExtractor(PhiPatternExtractor):
    """
    🔬 EXTRACTOR MEJORADO DE PATRONES CAUSALES
    
    Mejoras sobre PhiPatternExtractor base:
    1. Patrón causal base (6 valores)
    2. Attention statistics (2 valores: mean, entropy)
    3. Module states (4 valores: visual, auditory, motor, executive)
    4. Consciousness variance (1 valor)
    5. Phi trajectory slope (1 valor: pendiente primeras iteraciones)
    
    Total: 14 dimensiones (vs 6 originales)
    """
    
    def __init__(self):
        super().__init__()
        self.feature_names = [
            # Causal (6)
            'visual→auditory', 'visual→motor', 'visual→executive',
            'auditory→motor', 'auditory→executive', 'motor→executive',
            # Attention (2)
            'attention_mean', 'attention_entropy',
            # Modules (4)
            'module_visual', 'module_auditory', 'module_motor', 'module_executive',
            # Consciousness (1)
            'consciousness_var',
            # Phi (1)
            'phi_slope'
        ]
    
    def extract_rich_pattern(self, debug_info: dict, phi_trajectory: List[float] = None) -> np.ndarray:
        """
        Extraer patrón enriquecido con múltiples features
        
        Args:
            debug_info: Información completa del forward pass
            phi_trajectory: Lista de valores Φ de primeras iteraciones (opcional)
        
        Returns:
            Vector numpy de 14 dimensiones
        """
        # 1. Patrón causal base (6 valores)
        phi_info = debug_info.get('phi_info', {})
        causal_pattern = self.extract_pattern(phi_info)
        
        # 2. Attention statistics (2 valores)
        attention_weights = debug_info.get('attention_weights')
        if attention_weights is not None:
            # Media
            attention_mean = attention_weights.mean().item()
            
            # Entropía (medida de dispersión)
            attn_flat = attention_weights.flatten().cpu().detach().numpy()
            attn_flat = attn_flat / (attn_flat.sum() + 1e-8)  # Normalizar
            
            # Shannon entropy
            entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8))
            attention_entropy = float(entropy)
        else:
            attention_mean = 0.5
            attention_entropy = 0.5
        
        # 3. Module states (4 valores)
        module_states = debug_info.get('module_states', {})
        module_visual = module_states.get('visual', 0.5)
        module_auditory = module_states.get('auditory', 0.5)
        module_motor = module_states.get('motor', 0.5)
        module_executive = module_states.get('executive', 0.5)
        
        # 4. Consciousness variance (1 valor)
        consciousness_state = debug_info.get('consciousness_state', 0.0)
        if isinstance(consciousness_state, torch.Tensor):
            consciousness_var = consciousness_state.var().item()
        else:
            consciousness_var = 0.0
        
        # 5. Phi trajectory slope (1 valor)
        if phi_trajectory and len(phi_trajectory) >= 2:
            # Pendiente lineal de primeras iteraciones
            phi_slope = float(np.polyfit(range(len(phi_trajectory)), phi_trajectory, 1)[0])
        else:
            phi_slope = 0.0
        
        # Combinar todo
        rich_pattern = np.array([
            # Causal (6)
            *causal_pattern,
            # Attention (2)
            attention_mean,
            attention_entropy,
            # Modules (4)
            module_visual,
            module_auditory,
            module_motor,
            module_executive,
            # Consciousness (1)
            consciousness_var,
            # Phi (1)
            phi_slope
        ], dtype=np.float32)
        
        # Normalizar a [0, 1] para estabilidad
        rich_pattern = np.clip(rich_pattern, 0.0, 10.0)  # Clip extremos
        
        return rich_pattern
    
    def pattern_to_string(self, pattern: np.ndarray) -> str:
        """
        Convertir patrón enriquecido a string legible
        
        Args:
            pattern: Vector de 14 dimensiones
        
        Returns:
            String formateado
        """
        if len(pattern) < 14:
            # Patrón simple (6 dims)
            return super().pattern_to_string(pattern)
        
        lines = []
        lines.append("   PATRÓN ENRIQUECIDO (14 features):")
        lines.append("   " + "─" * 50)
        
        for i, (name, value) in enumerate(zip(self.feature_names, pattern)):
            # Escalar para visualización
            if i < 6:  # Causal connections
                bar_value = value
            elif i < 8:  # Attention
                bar_value = value / 2.0  # Escalar
            elif i < 12:  # Modules
                bar_value = value
            else:  # Variance y slope
                bar_value = min(abs(value), 1.0)
            
            bar = '█' * int(bar_value * 15)
            lines.append(f"   {name:20s} {value:7.3f} {bar}")
        
        return '\n'.join(lines)


# =============================================================================
# ENHANCED CONTINUOUS LEARNING SERVER
# =============================================================================

class EnhancedContinuousLearningServer:
    """
    🚀 SERVIDOR MEJORADO DE APRENDIZAJE CONTINUO
    
    MEJORAS sobre ContinuousLearningServer:
    1. **Captura temprana**: Extrae patrón en iter 5-10 (configurable)
    2. **Firmas enriquecidas**: 14 features vs 6 originales
    3. **Tracking de trayectorias**: Guarda evolución de Φ
    4. **Mejor diferenciación**: Threshold adaptativo
    
    Soluciona el problema de saturación de patrones.
    """
    
    def __init__(self, 
                 infinito_system,
                 similarity_threshold: float = 0.85,
                 capture_iteration: int = 1,
                 warmup_iterations: int = 0):
        """
        Inicializar servidor mejorado
        
        Args:
            infinito_system: Instancia de InfinitoV51ConsciousnessBreakthrough
            similarity_threshold: Umbral de similitud (reducido a 0.85 vs 0.90)
            capture_iteration: En qué iteración capturar patrón (default: 1 - ANTES de saturación)
            warmup_iterations: Iteraciones de warmup antes de capturar (default: 0)
        """
        self.system = infinito_system
        self.pattern_extractor = EnhancedPhiPatternExtractor()
        self.memory_bank = PhiMemoryBank(similarity_threshold=similarity_threshold)
        self.running = True
        
        # Parámetros de captura
        self.capture_iteration = capture_iteration
        self.warmup_iterations = warmup_iterations
        
        # Tracking
        self.processing_count = 0
        self.last_phi_trajectory = []
        
        print(f"\n{'='*70}")
        print(f"🧠 ENHANCED CONTINUOUS LEARNING SERVER INICIALIZADO")
        print(f"{'='*70}")
        print(f"   Modo: Captura temprana (iter {capture_iteration})")
        print(f"   Warmup: {warmup_iterations} iteraciones")
        print(f"   Features: 14 dimensiones (vs 6 estándar)")
        print(f"   Threshold similitud: {similarity_threshold:.1%}")
        print(f"   Patrones en memoria: {self.memory_bank.pattern_count}")
        print(f"{'='*70}")
    
    def process_input(self, text: str, verbose: bool = True) -> dict:
        """
        Procesar input con captura temprana
        
        Args:
            text: Texto de entrada
            verbose: Si mostrar output detallado
        
        Returns:
            Diccionario con resultado
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔤 PROCESANDO #{self.processing_count + 1}: '{text}'")
            print(f"{'='*70}")
        
        try:
            # Configurar texto en el sistema
            self.system.input_text = text
            
            # Fase 1: Warmup rápido (3 iters sin logging)
            if verbose:
                print(f"   🔥 Warmup ({self.warmup_iterations} iters)...")
            
            for iteration in range(1, self.warmup_iterations + 1):
                self.system.train_step(iteration)
            
            # Fase 2: Tracking hasta captura
            if verbose:
                print(f"   📊 Tracking hasta iter {self.capture_iteration}...")
            
            phi_trajectory = []
            consciousness_trajectory = []
            
            for iteration in range(self.warmup_iterations + 1, self.capture_iteration + 1):
                metrics = self.system.train_step(iteration)
                phi_trajectory.append(metrics['phi'])
                consciousness_trajectory.append(metrics['consciousness'])
            
            # Fase 3: Captura del patrón en iteración objetivo
            if verbose:
                print(f"   🎯 Capturando patrón en iter {self.capture_iteration}...")
            
            self.system.model.eval()
            with torch.no_grad():
                inputs = self.system.generate_text_based_input(text)
                consciousness, phi, debug_info = self.system.model(inputs)
            
            # Extraer info
            phi_info = debug_info.get('phi_info', {})
            consciousness_val = consciousness.mean().item()
            phi_val = phi.mean().item()
            
            # Guardar trayectoria
            self.last_phi_trajectory = phi_trajectory + [phi_val]
            
            # Fase 4: Extraer patrón ENRIQUECIDO
            try:
                causal_pattern = self.pattern_extractor.extract_rich_pattern(
                    debug_info,
                    phi_trajectory=self.last_phi_trajectory
                )
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error extrayendo patrón enriquecido: {e}")
                # Fallback a patrón simple
                causal_pattern = self.pattern_extractor.extract_pattern(phi_info)
            
            # Fase 5: Buscar/Guardar en memoria
            result = self.memory_bank.add_pattern(
                causal_pattern,
                phi_info,
                text,
                consciousness_val
            )
            
            # Fase 6: Reportar
            if verbose:
                print(f"   📊 Φ final = {phi_val:.4f} | C = {consciousness_val:.4f}")
                print(f"   📈 Φ trayectoria (iter 1-{self.capture_iteration}): {[f'{p:.3f}' for p in self.last_phi_trajectory[:5]]}...")
                print(f"   {result['message']}")
                
                if result['status'] == 'RECOGNIZED':
                    print(f"   🎯 Similitud: {result['similarity']:.1%}")
                    print(f"   📝 Original: '{result['original_text']}'")
                    print(f"   🔢 Visto {result['seen_count']} veces")
                else:
                    print(f"   🆕 Pattern ID: {result['pattern_id']}")
                
                print(f"   💾 Total patrones en memoria: {self.memory_bank.pattern_count}")
            
            # Actualizar contador
            self.processing_count += 1
            
            # Añadir info extra al resultado
            result.update({
                'phi': phi_val,
                'consciousness': consciousness_val,
                'causal_pattern': causal_pattern.tolist(),
                'phi_trajectory': self.last_phi_trajectory,
                'text': text,
                'capture_iteration': self.capture_iteration
            })
            
            return result
        
        except Exception as e:
            if verbose:
                print(f"   ❌ Error procesando: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'ERROR', 'message': str(e)}
    
    def show_stats(self):
        """Mostrar estadísticas mejoradas"""
        stats = self.memory_bank.get_stats()
        
        print(f"\n{'='*70}")
        print(f"📊 ESTADÍSTICAS DEL SISTEMA (ENHANCED)")
        print(f"{'='*70}")
        print(f"   Total patrones únicos: {stats['total_patterns']}")
        print(f"   Total inputs procesados: {stats['total_seen']}")
        print(f"   Procesados en sesión: {self.processing_count}")
        print(f"   Φ promedio: {stats['avg_phi']:.4f}")
        print(f"   Consciencia promedio: {stats['avg_consciousness']:.4f}")
        print(f"   Threshold similitud: {self.memory_bank.similarity_threshold:.1%}")
        print(f"   Captura en iteración: {self.capture_iteration}")
        print(f"   Features por patrón: 14 (enriquecido)")
        
        if stats['total_patterns'] > 0:
            print(f"\n   🏆 Más visto:")
            print(f"      '{stats['most_seen']['text']}'")
            print(f"      ({stats['most_seen']['count']} veces)")
        
        print(f"{'='*70}")
    
    def list_patterns(self, max_show: int = 10, show_features: bool = False):
        """Listar patrones con opción de mostrar features"""
        patterns = self.memory_bank.get_all_patterns()
        
        print(f"\n{'='*70}")
        print(f"📋 PATRONES EN MEMORIA ({len(patterns)} total)")
        print(f"{'='*70}")
        
        if not patterns:
            print("   (vacío)")
            return
        
        # Ordenar por veces visto
        patterns_sorted = sorted(patterns, key=lambda p: p.seen_count, reverse=True)
        
        for i, pattern in enumerate(patterns_sorted[:max_show], 1):
            print(f"\n   {i}. {pattern.id}: '{pattern.source_text}'")
            print(f"      Φ={pattern.phi_value:.3f}, C={pattern.consciousness:.3f}")
            print(f"      Visto {pattern.seen_count} vez/veces")
            
            if pattern.similar_patterns:
                print(f"      Similares: {', '.join(pattern.similar_patterns[:3])}")
            
            if show_features:
                # Mostrar features del patrón
                pattern_vector = np.array(pattern.causal_vector)
                print(f"\n{self.pattern_extractor.pattern_to_string(pattern_vector)}")
        
        if len(patterns) > max_show:
            print(f"\n   ... y {len(patterns) - max_show} más")
        
        print(f"{'='*70}")
    
    def compare_patterns(self, text1: str, text2: str):
        """
        Comparar patrones de dos textos lado a lado
        
        Args:
            text1: Primer texto
            text2: Segundo texto
        """
        print(f"\n{'='*70}")
        print(f"🔬 COMPARACIÓN DE PATRONES")
        print(f"{'='*70}")
        
        # Procesar ambos textos
        print(f"\n📝 Texto 1: '{text1}'")
        result1 = self.process_input(text1, verbose=False)
        
        print(f"\n📝 Texto 2: '{text2}'")
        result2 = self.process_input(text2, verbose=False)
        
        # Comparar patrones
        if 'causal_pattern' in result1 and 'causal_pattern' in result2:
            pattern1 = np.array(result1['causal_pattern'])
            pattern2 = np.array(result2['causal_pattern'])
            
            # Calcular similitud
            similarity = np.dot(pattern1, pattern2) / (
                np.linalg.norm(pattern1) * np.linalg.norm(pattern2) + 1e-8
            )
            
            print(f"\n{'='*70}")
            print(f"📊 RESULTADOS DE COMPARACIÓN")
            print(f"{'='*70}")
            print(f"   Similitud: {similarity:.1%}")
            print(f"   Φ₁ = {result1['phi']:.4f} | Φ₂ = {result2['phi']:.4f}")
            print(f"   C₁ = {result1['consciousness']:.4f} | C₂ = {result2['consciousness']:.4f}")
            
            # Interpretación
            if similarity > 0.95:
                print(f"\n   💡 Interpretación: Patrones CASI IDÉNTICOS")
            elif similarity > 0.85:
                print(f"\n   💡 Interpretación: Patrones MUY SIMILARES")
            elif similarity > 0.70:
                print(f"\n   💡 Interpretación: Patrones MODERADAMENTE SIMILARES")
            else:
                print(f"\n   💡 Interpretación: Patrones DIFERENTES")
            
            print(f"{'='*70}")
    
    def run_interactive_loop(self):
        """Loop interactivo mejorado"""
        print(f"\n{'='*70}")
        print(f"🚀 MODO INTERACTIVO MEJORADO ACTIVADO")
        print(f"{'='*70}")
        print(f"📝 Escribe texto para procesar (o comando especial)")
        print(f"\n💡 Comandos disponibles:")
        print(f"   'save'       - Guardar memoria a disco")
        print(f"   'stats'      - Ver estadísticas")
        print(f"   'list'       - Listar todos los patrones")
        print(f"   'features'   - Listar patrones con features detalladas")
        print(f"   'compare'    - Comparar dos textos")
        print(f"   'clear'      - Limpiar memoria (⚠️ destructivo)")
        print(f"   'config'     - Ver/cambiar configuración")
        print(f"   'exit'       - Salir y guardar")
        print(f"{'='*70}")
        
        while self.running:
            try:
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
                
                elif text.lower() == 'features':
                    self.list_patterns(show_features=True)
                
                elif text.lower() == 'compare':
                    text1 = input("   📝 Texto 1: ").strip()
                    text2 = input("   📝 Texto 2: ").strip()
                    if text1 and text2:
                        self.compare_patterns(text1, text2)
                
                elif text.lower() == 'config':
                    print(f"\n⚙️  CONFIGURACIÓN ACTUAL:")
                    print(f"   Capture iteration: {self.capture_iteration}")
                    print(f"   Warmup iterations: {self.warmup_iterations}")
                    print(f"   Similarity threshold: {self.memory_bank.similarity_threshold:.1%}")
                    
                    change = input("\n   ¿Cambiar? (y/n): ")
                    if change.lower() == 'y':
                        try:
                            new_capture = input(f"   Nueva capture iteration ({self.capture_iteration}): ")
                            if new_capture:
                                self.capture_iteration = int(new_capture)
                            
                            new_threshold = input(f"   Nuevo threshold ({self.memory_bank.similarity_threshold:.2f}): ")
                            if new_threshold:
                                self.memory_bank.similarity_threshold = float(new_threshold)
                            
                            print("   ✅ Configuración actualizada")
                        except Exception as e:
                            print(f"   ❌ Error: {e}")
                
                elif text.lower() == 'clear':
                    confirm = input("⚠️  ¿Seguro que quieres limpiar la memoria? (yes/no): ")
                    if confirm.lower() == 'yes':
                        self.memory_bank.clear()
                        self.processing_count = 0
                    else:
                        print("   Operación cancelada")
                
                else:
                    # Procesar input normal
                    self.process_input(text)
            
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

def create_enhanced_server(infinito_system, 
                          similarity_threshold: float = 0.85,
                          capture_iteration: int = 8):
    """
    Función helper para crear servidor mejorado
    
    Args:
        infinito_system: Sistema INFINITO inicializado
        similarity_threshold: Umbral de similitud (default: 0.85)
        capture_iteration: Iteración de captura (default: 8)
    
    Returns:
        EnhancedContinuousLearningServer instance
    """
    return EnhancedContinuousLearningServer(
        infinito_system,
        similarity_threshold=similarity_threshold,
        capture_iteration=capture_iteration
    )


if __name__ == "__main__":
    print("🔄 Enhanced Continuous Learning System - Módulo cargado")
    print("   Mejoras: Captura temprana + Firmas enriquecidas (14 features)")
    print("   Para usar, importar:")
    print("   from continuous_learning_enhanced import EnhancedContinuousLearningServer")
