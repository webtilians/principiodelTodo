#!/usr/bin/env python3
"""
🎯 CONTINUOUS LEARNING - LOGIT-BASED PATTERN EXTRACTION
========================================================

SOLUCIÓN DEFINITIVA al problema de saturación del sigmoid.

PROBLEMA RAÍZ:
- sigmoid(x) → 0.999 cuando x > 3 (saturación)
- Las redes entrenan para maximizar integración → x > 3
- Resultado: TODOS los patrones causal_matrix ≈ [0.999, 0.999, ...]

SOLUCIÓN:
- NO capturar sigmoid(x) (saturado)
- SÍ capturar x (logits RAW) que tienen variabilidad real
- Los logits pueden ser negativos, positivos, grandes, pequeños
- Mantienen discriminación entre textos diferentes

MEJORAS:
✅ Captura logits pre-sigmoid (variabilidad real)
✅ Normalización robusta (z-score + tanh)
✅ 14 features: logits (6) + attention (2) + modules (4) + meta (2)
✅ Sin necesidad de captura temprana
✅ Funciona en eval() mode sin entrenar
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from datetime import datetime
from continuous_learning import PhiMemoryBank, CausalPattern


class LogitBasedPatternExtractor:
    """
    Extractor de patrones basado en LOGITS RAW
    
    Usa los valores pre-sigmoid que mantienen variabilidad
    entre diferentes textos.
    """
    
    def __init__(self):
        self.feature_names = [
            # Logits causales RAW (6 features) - NO saturados
            "logit_visual_to_auditory",
            "logit_visual_to_motor", 
            "logit_visual_to_executive",
            "logit_auditory_to_motor",
            "logit_auditory_to_executive",
            "logit_motor_to_executive",
            
            # Atención (2 features)
            "attention_mean",
            "attention_entropy",
            
            # Estados modulares (4 features)
            "visual_activation",
            "auditory_activation",
            "motor_activation",
            "executive_activation",
            
            # Meta-features (2 features)
            "consciousness",
            "phi",
        ]
    
    def extract_logit_pattern(self, phi_info: dict, debug_info: dict = None) -> np.ndarray:
        """
        Extraer patrón basado en logits RAW
        
        Args:
            phi_info: Info de Φ (debe contener 'causal_logits')
            debug_info: Info debug del modelo (opcional, para features adicionales)
        
        Returns:
            Vector de 14 features con variabilidad real
        """
        # 1. LOGITS RAW (6 features) - LA CLAVE DE LA SOLUCIÓN
        causal_logits = phi_info.get('causal_logits')
        
        if causal_logits is None:
            raise ValueError("phi_info no contiene 'causal_logits' - usar modelo actualizado")
        
        # Convertir a numpy
        if isinstance(causal_logits, torch.Tensor):
            causal_logits = causal_logits.cpu().detach().numpy()
        
        # Extraer las 6 conexiones (igual que con sigmoid, pero valores RAW)
        logits = np.array([
            causal_logits[0, 1],  # visual → auditory
            causal_logits[0, 2],  # visual → motor
            causal_logits[0, 3],  # visual → executive
            causal_logits[1, 2],  # auditory → motor
            causal_logits[1, 3],  # auditory → executive
            causal_logits[2, 3],  # motor → executive
        ], dtype=np.float32)
        
        # ✅ SOLUCIÓN CORRECTA: Usar logits RAW sin z-score individual
        # La normalización z-score por patrón elimina las diferencias absolutas
        # Solo aplicamos clipping suave para estabilidad numérica
        logits_normalized = np.tanh(logits / 3.0)  # Escalar a [-1, 1] suavemente
        
        # 2. Features de atención (2 features)
        if debug_info and 'attention_weights' in debug_info:
            attn = debug_info['attention_weights']
            if isinstance(attn, torch.Tensor):
                attn = attn.cpu().detach().numpy()
            
            attn_flat = attn.flatten()
            attn_mean = float(np.mean(attn_flat))
            
            # Entropía de atención
            attn_prob = attn_flat + 1e-10
            attn_prob = attn_prob / np.sum(attn_prob)
            attn_entropy = -float(np.sum(attn_prob * np.log(attn_prob + 1e-10)))
            attn_entropy = attn_entropy / 10.0  # Normalizar a ~[0, 1]
        else:
            attn_mean = 0.5
            attn_entropy = 0.5
        
        # 3. Estados modulares (4 features)
        if debug_info and 'module_states' in debug_info:
            mod_states = debug_info['module_states']
            
            # Extraer valores de forma segura (pueden ser tensors o floats)
            def safe_extract(val):
                if isinstance(val, torch.Tensor):
                    return float(val.mean().item())
                elif isinstance(val, (int, float)):
                    return float(val)
                else:
                    return 0.5
            
            visual_act = safe_extract(mod_states.get('visual', 0.5))
            auditory_act = safe_extract(mod_states.get('auditory', 0.5))
            motor_act = safe_extract(mod_states.get('motor', 0.5))
            executive_act = safe_extract(mod_states.get('executive', 0.5))
        else:
            # Fallback: usar info indirecta de los logits
            visual_act = float(np.tanh(np.mean([logits[0], logits[1], logits[2]])))
            auditory_act = float(np.tanh(np.mean([logits[0], logits[3], logits[4]])))
            motor_act = float(np.tanh(np.mean([logits[1], logits[3], logits[5]])))
            executive_act = float(np.tanh(np.mean([logits[2], logits[4], logits[5]])))
        
        # 4. Meta-features (2 features)
        consciousness = float(phi_info.get('consciousness_scaling', 0.5))
        phi = float(phi_info.get('phi_total', 0.5))
        phi_normalized = np.tanh(phi / 5.0)  # Normalizar Φ típicamente 0-10 → [-1, 1]
        
        # 5. Ensamblar vector de 14 features
        pattern = np.concatenate([
            logits_normalized,  # 6 logits normalizados
            [attn_mean, attn_entropy],  # 2 atención
            [visual_act, auditory_act, motor_act, executive_act],  # 4 módulos
            [consciousness, phi_normalized],  # 2 meta
        ]).astype(np.float32)
        
        return pattern
    
    def pattern_to_string(self, pattern: np.ndarray, indent: str = "   ") -> str:
        """Convertir patrón a string legible"""
        if len(pattern) != 14:
            return f"{indent}⚠️ Patrón inválido (len={len(pattern)})"
        
        lines = [f"{indent}📋 Patrón basado en LOGITS (14 features):"]
        lines.append(f"{indent}─" * 50)
        
        # Logits
        lines.append(f"{indent}🔗 Conexiones Causales (logits normalizados):")
        for i in range(6):
            name = self.feature_names[i]
            val = pattern[i]
            lines.append(f"{indent}  {i+1}. {name:30s}: {val:+.4f}")
        
        # Atención
        lines.append(f"{indent}👁️  Atención:")
        for i in range(6, 8):
            name = self.feature_names[i]
            val = pattern[i]
            lines.append(f"{indent}  {i+1}. {name:30s}: {val:.4f}")
        
        # Módulos
        lines.append(f"{indent}🧠 Estados Modulares:")
        for i in range(8, 12):
            name = self.feature_names[i]
            val = pattern[i]
            lines.append(f"{indent}  {i+1}. {name:30s}: {val:+.4f}")
        
        # Meta
        lines.append(f"{indent}📊 Meta-features:")
        for i in range(12, 14):
            name = self.feature_names[i]
            val = pattern[i]
            lines.append(f"{indent}  {i+1}. {name:30s}: {val:.4f}")
        
        return "\n".join(lines)


class LogitBasedLearningServer:
    """
    Servidor de aprendizaje continuo basado en LOGITS
    
    Solución definitiva al problema de saturación.
    Funciona en modo eval sin necesidad de captura temprana.
    """
    
    def __init__(self, infinito_system, similarity_threshold: float = 0.85):
        """
        Inicializar servidor
        
        Args:
            infinito_system: Instancia de INFINITO V5.1 (con causal_logits)
            similarity_threshold: Umbral de similitud
        """
        self.system = infinito_system
        self.extractor = LogitBasedPatternExtractor()
        self.memory_bank = PhiMemoryBank(similarity_threshold=similarity_threshold)
        self.running = True
        self.processing_count = 0
        
        # Poner en modo eval
        if hasattr(self.system, 'model'):
            self.system.model.eval()
        
        print(f"\n{'='*70}")
        print(f"🎯 LOGIT-BASED CONTINUOUS LEARNING SERVER INICIALIZADO")
        print(f"{'='*70}")
        print(f"   Método: Logits RAW pre-sigmoid (NO saturados)")
        print(f"   Features: 14 dimensiones (logits + attention + modules + meta)")
        print(f"   Threshold similitud: {similarity_threshold:.1%}")
        print(f"   Patrones en memoria: {self.memory_bank.pattern_count}")
        print(f"{'='*70}")
    
    def process_input(self, text: str, verbose: bool = True) -> dict:
        """
        Procesar input SIN entrenamiento
        
        Args:
            text: Texto de entrada
            verbose: Mostrar output detallado
        
        Returns:
            Diccionario con resultado
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔤 PROCESANDO #{self.processing_count + 1}: '{text}'")
            print(f"{'='*70}")
        
        try:
            # 1. Generar input
            inputs = self.system.generate_text_based_input(
                text=text,
                batch_size=self.system.batch_size,
                seq_len=64
            )
            
            # 2. Forward pass (sin gradientes)
            with torch.no_grad():
                self.system.model.eval()
                consciousness, phi, debug_info = self.system.model(inputs)
            
            # 3. Extraer phi_info
            phi_info = debug_info.get('phi_info', {})
            consciousness_val = consciousness.mean().item()
            phi_val = phi.mean().item()
            
            # 4. Verificar que tenemos logits
            if 'causal_logits' not in phi_info:
                raise ValueError("El modelo no está generando 'causal_logits'. Asegúrate de usar la versión actualizada de infinito_gpt_text_fixed.py")
            
            # 5. Extraer patrón basado en LOGITS
            if verbose:
                print(f"   🎯 Extrayendo patrón de LOGITS RAW...")
            
            try:
                causal_pattern = self.extractor.extract_logit_pattern(phi_info, debug_info)
            except Exception as e:
                print(f"   ⚠️  Error extrayendo patrón: {e}")
                return {'status': 'ERROR', 'message': str(e)}
            
            # 6. Buscar/Guardar en memoria
            result = self.memory_bank.add_pattern(
                causal_pattern,
                phi_info,
                text,
                consciousness_val
            )
            
            # 7. Reportar
            if verbose:
                print(f"   📊 Φ = {phi_val:.4f} | C = {consciousness_val:.4f}")
                print(f"   {result['message']}")
                
                if result['status'] == 'RECOGNIZED':
                    print(f"   🎯 Similitud: {result['similarity']:.1%}")
                    print(f"   📝 Original: '{result['original_text']}'")
                    print(f"   🔢 Visto {result['seen_count']} veces")
                else:
                    print(f"   🆕 Pattern ID: {result['pattern_id']}")
                
                print(f"   💾 Total patrones: {self.memory_bank.pattern_count}")
            
            # 8. Actualizar contador
            self.processing_count += 1
            
            # 9. Añadir info extra
            result.update({
                'phi': phi_val,
                'consciousness': consciousness_val,
                'causal_pattern': causal_pattern.tolist(),
                'text': text
            })
            
            return result
        
        except Exception as e:
            error_msg = f"❌ Error: {e}"
            if verbose:
                print(f"\n{error_msg}")
                import traceback
                traceback.print_exc()
            
            return {'status': 'ERROR', 'message': str(e)}
    
    def compare_patterns(self, text1: str, text2: str):
        """Comparar dos patrones lado a lado"""
        print(f"\n{'='*70}")
        print(f"🔬 COMPARACIÓN DE PATRONES (basados en logits)")
        print(f"{'='*70}")
        print(f"   Texto 1: '{text1}'")
        print(f"   Texto 2: '{text2}'")
        print(f"{'─'*70}")
        
        # Procesar ambos
        result1 = self.process_input(text1, verbose=False)
        result2 = self.process_input(text2, verbose=False)
        
        if result1['status'] == 'ERROR' or result2['status'] == 'ERROR':
            print(f"   ❌ Error en procesamiento")
            return 0.0
        
        pattern1 = np.array(result1['causal_pattern'])
        pattern2 = np.array(result2['causal_pattern'])
        
        # Calcular similitud
        similarity = np.dot(pattern1, pattern2) / (
            np.linalg.norm(pattern1) * np.linalg.norm(pattern2) + 1e-8
        )
        
        print(f"\n   📊 Similitud: {similarity:.1%}")
        print(f"   Φ₁ = {result1['phi']:.4f}, Φ₂ = {result2['phi']:.4f}")
        print(f"   C₁ = {result1['consciousness']:.4f}, C₂ = {result2['consciousness']:.4f}")
        
        # Mostrar diferencias en logits (primeras 6 features)
        print(f"\n   🔍 Diferencias en LOGITS causales:")
        logit_names = self.extractor.feature_names[:6]
        for i in range(6):
            diff = abs(pattern1[i] - pattern2[i])
            print(f"      {logit_names[i]:30s}: {pattern1[i]:+.3f} vs {pattern2[i]:+.3f} (Δ={diff:.3f})")
        
        print(f"{'='*70}")
        
        return similarity
    
    def show_stats(self):
        """Estadísticas del sistema"""
        stats = self.memory_bank.get_stats()
        
        print(f"\n{'='*70}")
        print(f"📊 ESTADÍSTICAS")
        print(f"{'='*70}")
        print(f"   Inputs procesados: {self.processing_count}")
        print(f"   Patrones únicos: {stats['total_patterns']}")
        print(f"   Total vistos: {stats['total_seen']}")
        print(f"   Φ promedio: {stats['avg_phi']:.4f}")
        print(f"   C promedio: {stats['avg_consciousness']:.4f}")
        print(f"   Threshold: {self.memory_bank.similarity_threshold:.1%}")
        print(f"{'='*70}")
    
    def list_patterns(self, show_features: bool = False):
        """Listar patrones"""
        print(f"\n{'='*70}")
        print(f"📚 PATRONES EN MEMORIA ({self.memory_bank.pattern_count})")
        print(f"{'='*70}")
        
        for i, pattern in enumerate(self.memory_bank.patterns.values()):
            print(f"\n{i+1}. {pattern.id}")
            print(f"   Texto: '{pattern.source_text}'")
            print(f"   Φ = {pattern.phi_value:.4f}, C = {pattern.consciousness:.4f}")
            print(f"   Visto: {pattern.seen_count} veces")
            
            if show_features:
                pattern_vec = np.array(pattern.causal_vector)
                print(self.extractor.pattern_to_string(pattern_vec))
        
        print(f"{'='*70}")
    
    def save_memory(self, filename: str = "phi_memory_bank.json"):
        """Guardar memoria"""
        self.memory_bank.save_to_disk(filename)
    
    def load_memory(self, filename: str = "phi_memory_bank.json"):
        """Cargar memoria"""
        self.memory_bank.load_from_disk(filename)
    
    def run_interactive_loop(self):
        """Loop interactivo"""
        print(f"\n🎮 Modo interactivo activado")
        print(f"Comandos: save, stats, list, compare, features, clear, exit")
        
        while self.running:
            try:
                user_input = input(f"\n💬 Texto (o comando): ").strip()
                
                if not user_input:
                    continue
                
                if user_input == 'exit':
                    print(f"👋 ¡Adiós!")
                    break
                elif user_input == 'save':
                    self.save_memory()
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
                        self.compare_patterns(parts[1].strip(), parts[2].strip())
                    else:
                        print(f"Uso: compare|texto1|texto2")
                else:
                    self.process_input(user_input)
                
            except KeyboardInterrupt:
                print(f"\n👋 ¡Adiós!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🎯 Continuous Learning basado en Logits")
    print("Usa demo_logit_learning.py para ejecutar el sistema")
