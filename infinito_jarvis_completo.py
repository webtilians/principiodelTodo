#!/usr/bin/env python3
"""
🔮 INFINITO JARVIS COMPLETO - USA TODA LA ARQUITECTURA IIT
===========================================================

Esta versión REALMENTE usa toda tu arquitectura:
- IITGuidedMemory (memoria con priorización por PHI)
- ImprovedIITMetrics (4 componentes de integración)
- LearnablePhiWeights (pesos aprendibles)
- Dynamic Gate para importancia
- Memory Gate para inyección de memoria

¡Tu arquitectura completa trabajando!
"""

import torch
import torch.nn as nn
import json
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from infinito_v5_2_refactored import InfinitoV52Refactored

# --- CONFIGURACIÓN OPENAI ---
USE_OPENAI = True
API_KEY = os.environ.get("OPENAI_API_KEY", "")  # Configura tu API key como variable de entorno

openai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=API_KEY)
        print("✅ OpenAI conectado")
    except Exception as e:
        print(f"⚠️ OpenAI no disponible: {e}")
        USE_OPENAI = False


# =============================================================================
# MODELO COMPLETO: USA TODA TU ARQUITECTURA IIT
# =============================================================================

class InfinitoJarvisCompleto(InfinitoV52Refactored):
    """
    🧠 VERSIÓN COMPLETA que usa TODA tu arquitectura IIT:
    
    - IITGuidedMemory: Memoria externa con priorización por PHI
    - ImprovedIITMetrics: 4 componentes de integración de información
    - LearnablePhiWeights: Pesos aprendibles para combinar métricas
    - StochasticExploration: Exploración durante entrenamiento
    - Memory Gate: Cuánta memoria inyectar
    - Importance Gate: Qué guardar en JSON (nuevo)
    """
    
    def __init__(self, *args, **kwargs):
        # Forzar uso de todas las features IIT
        kwargs['use_improved_memory'] = True
        kwargs['use_improved_iit'] = True
        kwargs['use_learnable_phi'] = True
        kwargs['use_stochastic_exploration'] = True
        kwargs['use_dynamic_gate'] = False  # Usamos el memory_gate original
        
        super().__init__(*args, **kwargs)
        
        # Añadir Importance Gate (decide qué guardar en JSON)
        # Este es ADICIONAL al memory_gate que decide cuánta memoria inyectar
        self.importance_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, 1)
        )
        self._init_importance_gate()
        
        print("\n   🧠 ARQUITECTURA COMPLETA IIT ACTIVADA:")
        print("   ├── IITGuidedMemory (priorización por PHI)")
        print("   ├── ImprovedIITMetrics (4 componentes)")
        print("   ├── LearnablePhiWeights (pesos aprendibles)")
        print("   ├── Memory Gate (inyección de memoria)")
        print("   └── Importance Gate (decisión de guardar)")
    
    def _init_importance_gate(self):
        """Inicializa el gate de importancia."""
        for layer in self.importance_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Empezar ligeramente cerrado
        nn.init.constant_(self.importance_gate[-1].bias, -2.0)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_metrics: bool = False,
        write_to_memory: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward COMPLETO que usa TODA la arquitectura IIT.
        
        Flujo:
        1. Embeddings + Transformer
        2. IIT Metrics (4 componentes PHI)
        3. IIT Memory Read (priorizado por PHI)
        4. Memory Gate (cuánta memoria inyectar)
        5. IIT Memory Write (si write_to_memory=True)
        6. Importance Gate (qué tan importante es para guardar)
        7. Output
        """
        batch_size, seq_len = input_ids.shape
        
        # =====================================================================
        # 1. EMBEDDINGS
        # =====================================================================
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding[:, :seq_len, :]
        hidden = self.embedding_dropout(hidden)
        
        # =====================================================================
        # 2. TRANSFORMER LAYERS (con atención multi-cabeza)
        # =====================================================================
        all_attentions = []
        for i, (attn, ff, ln1, ln2) in enumerate(zip(
            self.attention_layers,
            self.ff_layers,
            self.layer_norms_1,
            self.layer_norms_2
        )):
            # Self-Attention
            attn_out, attn_weights = attn(hidden)
            all_attentions.append(attn_weights)
            hidden = ln1(hidden + attn_out)
            
            # Feed-Forward
            ff_out = ff(hidden)
            hidden = ln2(hidden + ff_out)
            
            # Stochastic Exploration (solo en training)
            if self.training and self.stochastic_exploration is not None:
                hidden = self.stochastic_exploration(hidden)
        
        # =====================================================================
        # 3. IIT METRICS (4 componentes de integración de información)
        # =====================================================================
        # Calcular métricas IIT sobre el estado oculto
        query = hidden.mean(dim=1)  # [batch, hidden_dim]
        
        # Apilar las atenciones en un tensor si hay varias
        stacked_attention = None
        if all_attentions and len(all_attentions) > 0:
            # Filtrar None y apilar
            valid_attns = [a for a in all_attentions if a is not None]
            if valid_attns:
                try:
                    stacked_attention = torch.stack(valid_attns, dim=0).mean(dim=0)
                except:
                    stacked_attention = valid_attns[-1]  # Usar la última
        
        # Usar el IIT metrics mejorado (4 componentes)
        try:
            iit_metrics_dict = self.iit_metrics(hidden, stacked_attention)
        except Exception as e:
            # Fallback sin atención
            iit_metrics_dict = self.iit_metrics(hidden, None)
        
        # Combinar con pesos aprendibles si está disponible
        if self.learnable_phi_weights is not None:
            # Los pesos aprendibles solo retornan los pesos, no combinan
            phi_weights = self.learnable_phi_weights()  # Sin argumentos
            
            # Combinar manualmente las métricas con los pesos
            temporal = iit_metrics_dict.get('temporal_coherence', torch.tensor(0.0))
            integration = iit_metrics_dict.get('integration_strength', torch.tensor(0.0))
            complexity = iit_metrics_dict.get('complexity', torch.tensor(0.0))
            attention = iit_metrics_dict.get('attention_diversity', torch.tensor(0.0))
            
            # Asegurar que son tensores
            if not torch.is_tensor(temporal):
                temporal = torch.tensor(temporal)
            if not torch.is_tensor(integration):
                integration = torch.tensor(integration)
            if not torch.is_tensor(complexity):
                complexity = torch.tensor(complexity)
            if not torch.is_tensor(attention):
                attention = torch.tensor(attention)
            
            # Combinación ponderada
            combined_phi = (
                phi_weights['temporal'] * temporal.mean() +
                phi_weights['integration'] * integration.mean() +
                phi_weights['complexity'] * complexity.mean() +
                phi_weights['attention'] * attention.mean()
            )
            integration_level = combined_phi
        else:
            integration_level = iit_metrics_dict.get('phi', torch.tensor(0.5))
        
        # =====================================================================
        # 4. IIT MEMORY READ (Priorizado por PHI)
        # =====================================================================
        read_content = None
        memory_stats = {}
        
        if hasattr(self, 'memory') and self.memory is not None:
            try:
                # Leer de la memoria con priorización por PHI
                read_result = self.memory.read(query, top_k=5)
                
                # El resultado puede ser tuple o tensor
                if isinstance(read_result, tuple):
                    read_content, read_weights = read_result
                else:
                    read_content = read_result
                
                memory_stats = self.memory.get_statistics()
            except Exception as e:
                # Si hay error, continuar sin memoria
                read_content = None
        
        # =====================================================================
        # 5. MEMORY GATE (Cuánta memoria inyectar)
        # =====================================================================
        memory_gate_value = torch.sigmoid(self.memory_gate)
        
        if read_content is not None:
            try:
                # Asegurar que las dimensiones sean compatibles
                if read_content.dim() == 2:
                    # [batch, hidden] -> [batch, 1, hidden]
                    read_content = read_content.unsqueeze(1)
                
                # Expandir para que coincida con hidden [batch, seq, hidden]
                if read_content.shape[1] != hidden.shape[1]:
                    read_content = read_content.expand(-1, hidden.shape[1], -1)
                
                # Aplicar gate
                gated_memory = memory_gate_value * read_content
                hidden = hidden + gated_memory
                hidden = self.memory_norm(hidden)
            except Exception as e:
                # Si hay error de dimensiones, continuar sin memoria
                pass
        
        # =====================================================================
        # 6. IIT MEMORY WRITE (Guardar en memoria IIT)
        # =====================================================================
        if write_to_memory and hasattr(self, 'memory') and self.memory is not None:
            try:
                # Escribir en memoria con el puntaje PHI
                phi_score = integration_level.mean().item() if torch.is_tensor(integration_level) else integration_level
                self.memory.write(query, priority=phi_score)
            except Exception as e:
                pass
        
        # =====================================================================
        # 7. IMPORTANCE GATE (Qué tan importante para guardar en JSON)
        # =====================================================================
        importance_logit = self.importance_gate(query)
        importance_value = torch.sigmoid(importance_logit)
        
        # =====================================================================
        # 8. OUTPUT
        # =====================================================================
        logits = self.output_projection(hidden)
        
        if return_metrics:
            metrics = {
                # Gate de importancia (para decidir guardar en JSON)
                'importance': importance_value.mean().item(),
                
                # Gate de memoria (cuánta memoria se inyectó)
                'memory_gate': memory_gate_value.item() if torch.is_tensor(memory_gate_value) else memory_gate_value,
                
                # Métricas IIT
                'phi': integration_level.mean().item() if torch.is_tensor(integration_level) else integration_level,
                'coherence': iit_metrics_dict.get('coherence', torch.tensor(0.0)).mean().item(),
                'complexity': iit_metrics_dict.get('complexity', torch.tensor(0.0)).mean().item(),
                'integration': iit_metrics_dict.get('integration_strength', torch.tensor(0.0)).mean().item(),
                
                # Estadísticas de memoria
                'memory_utilization': memory_stats.get('utilization', 0.0),
                'memory_size': memory_stats.get('size', 0),
            }
            return logits, metrics
        
        return logits, None


def text_to_ids(text, seq_len=32):
    """Convierte texto a IDs ASCII."""
    ids = [ord(c) % 256 for c in text]
    if len(ids) < seq_len:
        ids = ids + [0] * (seq_len - len(ids))
    else:
        ids = ids[:seq_len]
    return torch.tensor([ids])


# =============================================================================
# SISTEMA JARVIS COMPLETO
# =============================================================================

class JarvisCompletoSystem:
    """Sistema Jarvis que usa TODA la arquitectura IIT."""
    
    def __init__(self, model_path=None, db_file="memoria_infinito_completo.json"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.db_file = db_file
        
        print(f"\n{'='*65}")
        print(f"🔮 INFINITO JARVIS COMPLETO")
        print(f"{'='*65}")
        print(f"Device: {self.device}")
        print(f"LLM: {'OpenAI GPT' if USE_OPENAI else 'Simulación'}")
        
        # Crear modelo completo
        self._create_model(model_path)
        
        # Cargar memorias
        self._load_memories()
        
        print(f"{'='*65}\n")
    
    def _create_model(self, model_path):
        """Crea el modelo completo con toda la arquitectura IIT."""
        print(f"\n🧠 Inicializando arquitectura IIT completa...")
        
        self.model = InfinitoJarvisCompleto(
            vocab_size=256,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        ).to(self.device)
        
        # Intentar cargar pesos entrenados
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Cargar solo pesos compatibles
                model_dict = self.model.state_dict()
                compatible = {k: v for k, v in state_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(compatible)
                self.model.load_state_dict(model_dict, strict=False)
                
                print(f"   ✅ Cargados {len(compatible)}/{len(model_dict)} parámetros")
            except Exception as e:
                print(f"   ⚠️ No se cargaron pesos: {e}")
        
        self.model.eval()
        print(f"   ✅ Modelo listo")
    
    def _load_memories(self):
        """Carga memorias del JSON."""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)
            print(f"📚 {len(self.memories)} recuerdos cargados")
        else:
            self.memories = []
            print("✨ Nueva memoria iniciada")
    
    def _es_pregunta(self, texto):
        """Detecta si es una pregunta."""
        t = texto.lower().strip()
        if '?' in t or '¿' in t:
            return True
        starts = ['que ', 'qué ', 'como ', 'cómo ', 'cual ', 'cuál ', 
                  'quien ', 'quién ', 'donde ', 'dónde ', 'cuando ', 
                  'cuándo ', 'sabes ', 'recuerdas ', 'dime ']
        for s in starts:
            if t.startswith(s):
                return True
        return False
    
    def _categorize(self, text):
        """Categoriza la información."""
        t = text.lower()
        if any(x in t for x in ['me llamo', 'mi nombre', 'soy ']):
            return "👤 identidad"
        elif any(x in t for x in ['contraseña', 'clave', 'password', 'pin']):
            return "🔐 credencial"
        elif any(x in t for x in ['teléfono', 'email', 'correo', 'dirección']):
            return "📞 contacto"
        elif any(x in t for x in ['mi primo', 'mi hermano', 'mi madre', 'mi padre', 'mi amigo', 'mi esposa', 'mi hijo']):
            return "👨‍👩‍👧 familia"
        elif any(x in t for x in ['recuerda', 'no olvides', 'mañana', 'cita']):
            return "📌 recordatorio"
        elif any(x in t for x in ['me gusta', 'prefiero', 'favorito']):
            return "❤️ preferencia"
        return "📝 general"
    
    def _save_memory(self, text, importance, phi, category):
        """Guarda en memoria con métricas IIT."""
        entry = {
            "id": len(self.memories) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "content": text,
            "importance": round(importance * 100, 1),
            "phi": round(phi, 4),
            "category": category
        }
        self.memories.append(entry)
        
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, indent=2, ensure_ascii=False)
    
    def _construct_prompt(self):
        """Construye prompt con memoria."""
        memory_block = "NO TIENES RECUERDOS PREVIOS."
        
        if self.memories:
            # Ordenar por PHI (los más "integrados" primero)
            sorted_memories = sorted(self.memories, key=lambda x: x.get('phi', 0), reverse=True)
            
            memory_block = "📚 MEMORIA A LARGO PLAZO (ordenada por integración PHI):\n"
            for mem in sorted_memories[-15:]:
                phi = mem.get('phi', 0)
                imp = mem.get('importance', 0)
                cat = mem.get('category', '📝')
                memory_block += f"  • {mem['content']} [{cat}] (PHI:{phi:.2f}, Imp:{imp}%)\n"
        
        return f"""Eres Infinito, un asistente con MEMORIA PERSISTENTE basada en teoría IIT.

{memory_block}

INSTRUCCIONES:
1. USA la memoria para personalizar respuestas
2. Si preguntan algo en tu memoria, RESPONDE DIRECTAMENTE
3. Los recuerdos con PHI alto son más "integrados" en tu consciencia
4. Sé breve y útil. Responde en español."""
    
    def analyze(self, text):
        """Analiza texto con TODA la arquitectura IIT."""
        inp = text_to_ids(text).to(self.device)
        
        with torch.no_grad():
            _, metrics = self.model(inp, return_metrics=True, write_to_memory=False)
        
        return metrics
    
    def chat(self, user_text):
        """Procesa mensaje usando arquitectura completa."""
        
        # === ANÁLISIS CON ARQUITECTURA IIT COMPLETA ===
        metrics = self.analyze(user_text)
        
        importance = metrics['importance']
        phi = metrics['phi']
        memory_gate = metrics['memory_gate']
        coherence = metrics['coherence']
        is_question = self._es_pregunta(user_text)
        category = self._categorize(user_text)
        
        # === DECISIÓN INTELIGENTE DE GUARDAR ===
        # Combinar múltiples señales (no solo el gate de importancia)
        
        # 1. El gate de importancia (señal aprendida, pero no entrenada aún)
        importance_score = importance
        
        # 2. Bonus por categoría detectada (heurística semántica)
        category_bonus = 0.0
        if '👤' in category:  # identidad
            category_bonus = 0.5
        elif '🔐' in category:  # credencial
            category_bonus = 0.6
        elif '📞' in category:  # contacto
            category_bonus = 0.4
        elif '👨‍👩‍👧' in category:  # familia
            category_bonus = 0.4
        elif '📌' in category:  # recordatorio
            category_bonus = 0.5
        elif '❤️' in category:  # preferencia
            category_bonus = 0.3
        
        # 3. Bonus por PHI alto (información bien integrada)
        phi_bonus = phi * 0.3 if phi > 0.3 else 0.0
        
        # 4. Score combinado
        combined_score = importance_score + category_bonus + phi_bonus
        
        # Umbral adaptativo: guardar si score combinado > 0.3 o tiene categoría importante
        should_save = (combined_score > 0.3 or category_bonus > 0.3) and (not is_question)
        
        # === FEEDBACK VISUAL ===
        print(f"\n   ┌{'─'*55}┐")
        print(f"   │ 🧠 ANÁLISIS IIT COMPLETO                             │")
        print(f"   ├{'─'*55}┤")
        
        # Barra de importancia
        bar_imp = "█" * int(importance * 20) + "░" * (20 - int(importance * 20))
        print(f"   │ Importance: [{bar_imp}] {importance*100:>5.1f}% ", end="")
        print("🟢 │" if should_save else "🟡 │" if is_question else "🔴 │")
        
        # Score combinado
        bar_comb = "█" * int(min(combined_score, 1.0) * 20) + "░" * (20 - int(min(combined_score, 1.0) * 20))
        print(f"   │ Combined:   [{bar_comb}] {combined_score*100:>5.1f}%   │")
        
        # Barra de PHI
        bar_phi = "█" * int(phi * 20) + "░" * (20 - int(phi * 20))
        print(f"   │ PHI:        [{bar_phi}] {phi:>5.3f}   │")
        
        # Memory Gate
        bar_mem = "█" * int(memory_gate * 20) + "░" * (20 - int(memory_gate * 20))
        print(f"   │ Mem Gate:   [{bar_mem}] {memory_gate*100:>5.1f}%   │")
        
        # Coherence
        bar_coh = "█" * int(coherence * 20) + "░" * (20 - int(coherence * 20))
        print(f"   │ Coherence:  [{bar_coh}] {coherence:>5.3f}   │")
        
        print(f"   ├{'─'*55}┤")
        
        if should_save:
            self._save_memory(user_text, importance, phi, category)
            print(f"   │ 💾 GUARDADO: {category:<39} │")
        elif is_question:
            print(f"   │ ❓ Pregunta detectada (consultando memoria)         │")
        else:
            print(f"   │ 🔇 No relevante para memoria permanente             │")
        
        print(f"   └{'─'*55}┘")
        
        # === RESPUESTA CON LLM ===
        if USE_OPENAI and openai_client:
            try:
                prompt = self._construct_prompt()
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_text}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"❌ Error OpenAI: {e}"
        else:
            return self._simulate_response(user_text)
    
    def _simulate_response(self, text):
        """Respuesta simulada."""
        # Buscar nombre
        name = None
        for mem in self.memories:
            if '👤' in mem.get('category', ''):
                content = mem['content'].lower()
                for pattern in ['me llamo ', 'mi nombre es ', 'soy ']:
                    if pattern in content:
                        idx = content.find(pattern) + len(pattern)
                        name = mem['content'][idx:].split()[0].strip('.,!?')
                        break
        
        t = text.lower()
        if any(x in t for x in ['hola', 'buenos']):
            return f"¡Hola{' ' + name if name else ''}! ¿En qué puedo ayudarte?"
        elif 'cómo me llamo' in t or 'como me llamo' in t:
            return f"Te llamas {name}." if name else "No me has dicho tu nombre."
        elif 'qué sabes' in t or 'que sabes' in t:
            return f"Tengo {len(self.memories)} recuerdos. PHI promedio: {sum(m.get('phi',0) for m in self.memories)/max(len(self.memories),1):.3f}"
        return "Entendido. (Conecta OpenAI para respuestas reales)"
    
    def show_memory(self):
        """Muestra memoria con métricas IIT."""
        print(f"\n{'='*65}")
        print(f"🧠 MEMORIA IIT ({len(self.memories)} recuerdos)")
        print(f"{'='*65}")
        
        if not self.memories:
            print("   (Vacía)")
        else:
            # Ordenar por PHI
            for mem in sorted(self.memories, key=lambda x: x.get('phi', 0), reverse=True):
                cat = mem.get('category', '📝')
                phi = mem.get('phi', 0)
                imp = mem.get('importance', 0)
                print(f"\n   {cat} #{mem['id']} [{mem['timestamp']}]")
                print(f"      \"{mem['content']}\"")
                print(f"      PHI: {phi:.4f} | Importancia: {imp}%")
        
        print(f"\n{'='*65}")
    
    def show_iit_status(self):
        """Muestra estado de la arquitectura IIT."""
        print(f"\n{'='*65}")
        print(f"🔬 ESTADO ARQUITECTURA IIT")
        print(f"{'='*65}")
        
        # Estadísticas de memoria IIT
        if hasattr(self.model, 'memory') and self.model.memory is not None:
            stats = self.model.memory.get_statistics()
            print(f"\n   📦 IIT Memory:")
            print(f"      Tamaño: {stats.get('size', 0)} / {stats.get('capacity', 'N/A')}")
            print(f"      Utilización: {stats.get('utilization', 0)*100:.1f}%")
        
        # PHI weights si existen
        if hasattr(self.model, 'learnable_phi_weights') and self.model.learnable_phi_weights is not None:
            weights = self.model.learnable_phi_weights.get_weights_dict()
            print(f"\n   ⚖️ PHI Weights (aprendibles):")
            for k, v in weights.items():
                print(f"      {k}: {v:.4f}")
        
        # Memory Gate
        mg = torch.sigmoid(self.model.memory_gate).item()
        print(f"\n   🚪 Memory Gate: {mg*100:.2f}%")
        
        print(f"\n{'='*65}")


def main():
    """Función principal."""
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dynamic_chat_detector_v2.pt")
    
    jarvis = JarvisCompletoSystem(MODEL_PATH if os.path.exists(MODEL_PATH) else None)
    
    print("💬 JARVIS COMPLETO con arquitectura IIT")
    print("   Comandos: 'ver memoria', 'ver iit', 'borrar', 'salir'")
    print("─" * 65)
    
    while True:
        try:
            user_input = input("\n👤 Tú > ").strip()
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            
            if cmd in ['salir', 'exit']:
                break
            elif cmd == 'ver memoria':
                jarvis.show_memory()
                continue
            elif cmd == 'ver iit':
                jarvis.show_iit_status()
                continue
            elif cmd == 'borrar':
                if input("   ¿Seguro? (s/n): ").lower() == 's':
                    jarvis.memories = []
                    with open(jarvis.db_file, 'w') as f:
                        json.dump([], f)
                    print("   🗑️ Borrado")
                continue
            
            print("   🤔 Procesando con IIT...", end="\r")
            response = jarvis.chat(user_input)
            print(f"\n🤖 Infinito > {response}")
            
        except KeyboardInterrupt:
            break
    
    print(f"\n👋 ¡Hasta luego! ({len(jarvis.memories)} recuerdos guardados)")


if __name__ == "__main__":
    main()
