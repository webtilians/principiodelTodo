#!/usr/bin/env python3
"""
🎯 CONTINUOUS LEARNING - ZERO SHOT CAPTURE
==========================================

Solución DEFINITIVA al problema de saturación:
CAPTURA INMEDIATA del patrón tras embedding, SIN entrenamiento.

Problema identificado:
- Incluso 1 iteración causa saturación
- El modelo tiende a maximizar integración

Solución:
- Capturar patrón en forward pass inicial
- NO ejecutar train_step
- Usar estado inicial del modelo como firma
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from datetime import datetime
from continuous_learning import PhiMemoryBank, CausalPattern


class ZeroShotPhiExtractor:
    """
    Extractor de patrones sin entrenamiento
    
    Captura el estado del modelo INMEDIATAMENTE después de embedding,
    antes de cualquier backward pass o actualización de pesos.
    """
    
    def __init__(self):
        self.feature_names = [
            # Estados de embedding (4 features)
            "embedding_mean",
            "embedding_std",
            "embedding_max",
            "embedding_min",
            
            # Atención inicial (4 features)
            "attention_mean",
            "attention_std",
            "attention_entropy",
            "attention_sparsity",
            
            # Activaciones iniciales (4 features)
            "hidden_mean",
            "hidden_std",
            "hidden_max",
            "hidden_min",
            
            # Métricas de consciencia (2 features)
            "consciousness_initial",
            "phi_initial",
        ]
    
    def extract_zero_shot_pattern(self, infinito_system, text: str) -> np.ndarray:
        """
        Extraer patrón SIN entrenamiento
        
        Args:
            infinito_system: Sistema INFINITO
            text: Texto de entrada
        
        Returns:
            Vector de 14 features
        """
        infinito_system.model.eval()
        
        with torch.no_grad():
            # 1. Generar embedding
            inputs = infinito_system.generate_text_based_input(text)
            
            # 2. Forward pass ÚNICO (sin backward)
            try:
                consciousness, phi, debug_info, _ = infinito_system.forward_with_consciousness(
                    inputs['x'],
                    inputs['target'],
                    iteration=0  # Iteración 0 = sin entrenamiento
                )
            except Exception as e:
                print(f"   ⚠️ Error en forward: {e}")
                # Fallback
                consciousness = phi = 0.5
                debug_info = {}
            
            # 3. Extraer features del embedding
            embedding_vec = inputs['x'].flatten().cpu().numpy()
            embed_mean = float(np.mean(embedding_vec))
            embed_std = float(np.std(embedding_vec))
            embed_max = float(np.max(embedding_vec))
            embed_min = float(np.min(embedding_vec))
            
            # 4. Extraer features de atención (si disponible)
            if 'attention_weights' in debug_info and debug_info['attention_weights'] is not None:
                attn = debug_info['attention_weights'].cpu().numpy().flatten()
                attn_mean = float(np.mean(attn))
                attn_std = float(np.std(attn))
                
                # Entropía de atención
                attn_prob = attn + 1e-10
                attn_prob = attn_prob / np.sum(attn_prob)
                attn_entropy = -float(np.sum(attn_prob * np.log(attn_prob + 1e-10)))
                
                # Sparsity (qué tan concentrada está la atención)
                attn_sparsity = float(np.sum(attn > 0.1) / len(attn))
            else:
                attn_mean = attn_std = attn_entropy = attn_sparsity = 0.5
            
            # 5. Extraer features de hidden states
            if 'hidden_state' in debug_info and debug_info['hidden_state'] is not None:
                hidden = debug_info['hidden_state'].cpu().numpy().flatten()
                hidden_mean = float(np.mean(hidden))
                hidden_std = float(np.std(hidden))
                hidden_max = float(np.max(hidden))
                hidden_min = float(np.min(hidden))
            else:
                hidden_mean = hidden_std = 0.5
                hidden_max = 1.0
                hidden_min = 0.0
            
            # 6. Consciencia y Phi iniciales
            consciousness_val = float(consciousness.item() if torch.is_tensor(consciousness) else consciousness)
            phi_val = float(phi.item() if torch.is_tensor(phi) else phi)
            
            # 7. Ensamblar vector
            pattern = np.array([
                embed_mean,
                embed_std,
                embed_max,
                embed_min,
                attn_mean,
                attn_std,
                attn_entropy,
                attn_sparsity,
                hidden_mean,
                hidden_std,
                hidden_max,
                hidden_min,
                consciousness_val,
                phi_val,
            ], dtype=np.float32)
            
            return pattern
    
    def pattern_to_string(self, pattern: np.ndarray, indent: str = "   ") -> str:
        """Convertir patrón a string legible"""
        if len(pattern) != 14:
            return f"{indent}⚠️ Patrón inválido (len={len(pattern)})"
        
        lines = [f"{indent}📋 Patrón Zero-Shot (14 features):"]
        lines.append(f"{indent}─" * 50)
        
        for i, (name, val) in enumerate(zip(self.feature_names, pattern)):
            lines.append(f"{indent}  {i+1:2d}. {name:25s}: {val:.6f}")
        
        return "\n".join(lines)


class ZeroShotLearningServer:
    """
    Servidor de aprendizaje continuo con captura zero-shot
    """
    
    def __init__(self, infinito_system, similarity_threshold: float = 0.85):
        """
        Inicializar servidor zero-shot
        
        Args:
            infinito_system: Instancia de INFINITO V5.1
            similarity_threshold: Umbral de similitud
        """
        self.system = infinito_system
        self.extractor = ZeroShotPhiExtractor()
        self.memory_bank = PhiMemoryBank(similarity_threshold=similarity_threshold)
        self.running = True
        self.processing_count = 0
        
        print(f"\n{'='*70}")
        print(f"🚀 ZERO-SHOT CONTINUOUS LEARNING SERVER INICIALIZADO")
        print(f"{'='*70}")
        print(f"   Modo: Zero-Shot (SIN entrenamiento)")
        print(f"   Features: 14 dimensiones (embedding + attention + hidden + phi)")
        print(f"   Threshold similitud: {similarity_threshold:.1%}")
        print(f"   Patrones en memoria: {self.memory_bank.pattern_count}")
        print(f"{'='*70}")
    
    def process_input(self, text: str, verbose: bool = True) -> dict:
        """
        Procesar input sin entrenamiento
        
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
            # Configurar texto
            self.system.input_text = text
            
            # Captura ZERO-SHOT (sin entrenamiento)
            if verbose:
                print(f"   🎯 Captura zero-shot...")
            
            pattern_vector = self.extractor.extract_zero_shot_pattern(self.system, text)
            
            # Obtener Φ y C del patrón
            phi_val = float(pattern_vector[-1])
            consciousness_val = float(pattern_vector[-2])
            
            if verbose:
                print(f"   📊 Φ = {phi_val:.4f} | C = {consciousness_val:.4f}")
            
            # Buscar patrón similar en memoria
            if verbose:
                print(f"   🔍 Buscando en memoria...")
            
            result = self.memory_bank.add_pattern(
                causal_vector=pattern_vector,
                phi=phi_val,
                consciousness=consciousness_val,
                text=text
            )
            
            if verbose:
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
            
            # Añadir info extra
            result.update({
                'text': text,
                'phi': phi_val,
                'consciousness': consciousness_val,
                'causal_pattern': pattern_vector.tolist(),
                'iteration_captured': 0,  # Zero-shot
            })
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error procesando '{text}': {e}"
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'ERROR',
                'message': error_msg,
                'text': text
            }
    
    def compare_patterns(self, text1: str, text2: str):
        """Comparar dos patrones lado a lado"""
        print(f"\n{'='*70}")
        print(f"🔬 COMPARACIÓN DE PATRONES ZERO-SHOT")
        print(f"{'='*70}")
        print(f"   Texto 1: '{text1}'")
        print(f"   Texto 2: '{text2}'")
        print(f"{'─'*70}")
        
        # Procesar ambos
        result1 = self.process_input(text1, verbose=False)
        result2 = self.process_input(text2, verbose=False)
        
        pattern1 = np.array(result1['causal_pattern'])
        pattern2 = np.array(result2['causal_pattern'])
        
        # Calcular similitud
        similarity = np.dot(pattern1, pattern2) / (
            np.linalg.norm(pattern1) * np.linalg.norm(pattern2) + 1e-8
        )
        
        print(f"\n   📊 Similitud: {similarity:.1%}")
        print(f"   Φ₁ = {result1['phi']:.4f}, Φ₂ = {result2['phi']:.4f}")
        print(f"   C₁ = {result1['consciousness']:.4f}, C₂ = {result2['consciousness']:.4f}")
        
        # Mostrar features más diferentes
        diff = np.abs(pattern1 - pattern2)
        top_diff_indices = np.argsort(diff)[-3:][::-1]
        
        print(f"\n   🔍 Top 3 features más diferentes:")
        for idx in top_diff_indices:
            feat_name = self.extractor.feature_names[idx]
            val1 = pattern1[idx]
            val2 = pattern2[idx]
            print(f"      {feat_name:25s}: {val1:.4f} vs {val2:.4f} (Δ={diff[idx]:.4f})")
        
        print(f"{'='*70}")
        
        return similarity
    
    def show_stats(self):
        """Mostrar estadísticas del sistema"""
        print(f"\n{'='*70}")
        print(f"📊 ESTADÍSTICAS DEL SISTEMA")
        print(f"{'='*70}")
        print(f"   Inputs procesados: {self.processing_count}")
        print(f"   Patrones únicos: {self.memory_bank.pattern_count}")
        print(f"   Threshold similitud: {self.memory_bank.similarity_threshold:.1%}")
        print(f"   Archivo memoria: phi_memory_bank.json")
        print(f"{'='*70}")
    
    def list_patterns(self, show_features: bool = False):
        """Listar patrones en memoria"""
        print(f"\n{'='*70}")
        print(f"📚 PATRONES EN MEMORIA ({self.memory_bank.pattern_count})")
        print(f"{'='*70}")
        
        for i, pattern in enumerate(self.memory_bank.patterns):
            print(f"\n{i+1}. {pattern.pattern_id}")
            print(f"   Texto: '{pattern.text}'")
            print(f"   Φ = {pattern.phi:.4f}, C = {pattern.consciousness:.4f}")
            print(f"   Visto: {pattern.seen_count} veces")
            print(f"   Último: {pattern.timestamp}")
            
            if show_features:
                print(self.extractor.pattern_to_string(np.array(pattern.causal_vector)))
        
        print(f"{'='*70}")
    
    def save_memory(self, filename: str = "phi_memory_bank.json"):
        """Guardar memoria a disco"""
        self.memory_bank.save_to_disk(filename)
    
    def load_memory(self, filename: str = "phi_memory_bank.json"):
        """Cargar memoria desde disco"""
        self.memory_bank.load_from_disk(filename)
    
    def run_interactive_loop(self):
        """Loop interactivo"""
        print(f"\n🎮 Modo interactivo activado")
        print(f"Comandos: save, stats, list, compare, features, clear, exit")
        
        while self.running:
            try:
                user_input = input(f"\n💬 Ingresa texto (o comando): ").strip()
                
                if not user_input:
                    continue
                
                # Comandos
                if user_input == 'exit':
                    print(f"👋 ¡Adiós!")
                    break
                
                elif user_input == 'save':
                    self.save_memory()
                    print(f"✅ Memoria guardada")
                
                elif user_input == 'stats':
                    self.show_stats()
                
                elif user_input == 'list':
                    self.list_patterns(show_features=False)
                
                elif user_input == 'features':
                    self.list_patterns(show_features=True)
                
                elif user_input == 'clear':
                    self.memory_bank.clear()
                    print(f"🗑️ Memoria limpiada")
                
                elif user_input.startswith('compare'):
                    parts = user_input.split('|')
                    if len(parts) == 3:
                        text1 = parts[1].strip()
                        text2 = parts[2].strip()
                        self.compare_patterns(text1, text2)
                    else:
                        print(f"Uso: compare|texto1|texto2")
                
                # Procesar texto
                else:
                    self.process_input(user_input)
                
            except KeyboardInterrupt:
                print(f"\n👋 ¡Adiós!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    print("🧪 Modo test de Zero-Shot Learning")
    print("Usa demo_zero_shot_learning.py para ejecutar el sistema completo")
