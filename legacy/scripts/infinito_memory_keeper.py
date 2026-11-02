#!/usr/bin/env python3
"""
🧠 INFINITO MEMORY KEEPER
=========================

Sistema de chat con memoria selectiva basado en IIT.

El modelo Infinito analiza cada frase y decide:
- Gate > 50% → GUARDAR en memoria permanente (información importante)
- Gate < 50% → IGNORAR (ruido trivial)

Esta es la demostración final de la tecnología de gate dinámico.
"""

import torch
import torch.nn as nn
import json
import os
import sys
from datetime import datetime

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infinito_v5_2_refactored import InfinitoV52Refactored


# --- 1. ARQUITECTURA DEL MODELO (igual que en entrenamiento) ---

class InfinitoDynamicChat(InfinitoV52Refactored):
    """Modelo con gate dinámico para detectar información importante.
    
    v3: Gate con arquitectura mejorada (64→64→32→1) entrenado con 10k+ ejemplos.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['use_dynamic_gate'] = False
        super().__init__(*args, **kwargs)
        
        if hasattr(self, 'memory_gate'):
            del self.memory_gate
        
        # Gate v3: arquitectura más profunda (64→64→32→1)
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),      # 64→64
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), # 64→32
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)                # 32→1
        )
    
    def forward(self, input_ids, return_metrics=False):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding[:, :seq_len, :]
        hidden = self.embedding_dropout(hidden)
        
        # Transformer
        for attn, ff, ln1, ln2 in zip(
            self.attention_layers, 
            self.ff_layers, 
            self.layer_norms_1, 
            self.layer_norms_2
        ):
            attn_out, _ = attn(hidden)
            hidden = ln1(hidden + attn_out)
            ff_out = ff(hidden)
            hidden = ln2(hidden + ff_out)
        
        # Gate dinámico
        sentence_context = hidden.mean(dim=1)
        gate_logit = self.gate_network(sentence_context)
        gate_open_pct = torch.sigmoid(gate_logit)
        
        logits = self.output_projection(hidden)
        
        if return_metrics:
            return logits, {'gate_value': gate_open_pct.mean().item()}
        return logits, None


# --- 2. TOKENIZER ---

def text_to_ids(text, seq_len=32):
    """Convierte texto a IDs (ASCII)."""
    ids = [ord(c) % 256 for c in text]
    if len(ids) < seq_len:
        ids = ids + [0] * (seq_len - len(ids))
    else:
        ids = ids[:seq_len]
    return torch.tensor([ids])  # Batch size 1


# --- 3. SISTEMA DE MEMORIA ---

class MemoryKeeperSystem:
    """Sistema de chat con memoria selectiva."""
    
    def __init__(self, model_path, db_file="memoria_permanente.json", threshold=50.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.db_file = db_file
        self.threshold = threshold
        
        print(f"\n{'='*60}")
        print(f"🧠 INFINITO MEMORY KEEPER")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Umbral de importancia: {threshold}%")
        
        # Cargar modelo
        self._load_model(model_path)
        
        # Cargar/crear base de datos
        self._load_database()
        
        print(f"{'='*60}\n")
    
    def _load_model(self, model_path):
        """Carga el modelo entrenado."""
        print(f"\n📦 Cargando modelo desde: {model_path}")
        
        self.model = InfinitoDynamicChat(
            vocab_size=256, 
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            use_improved_memory=True,
            use_improved_iit=True,
        ).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                # Cargar con strict=False para ignorar claves incompatibles
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if 'accuracy' in checkpoint:
                    print(f"   Accuracy del modelo: {checkpoint['accuracy']*100:.1f}%")
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.eval()
            print("   ✅ Modelo cargado correctamente")
            
        except Exception as e:
            print(f"   ❌ Error cargando modelo: {e}")
            sys.exit(1)
    
    def _load_database(self):
        """Carga o crea la base de datos de memoria."""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r', encoding='utf-8') as f:
                self.memory_db = json.load(f)
            print(f"📂 Memoria cargada: {len(self.memory_db)} recuerdos")
        else:
            self.memory_db = []
            print("✨ Nueva memoria creada")
    
    def analyze_importance(self, text):
        """Analiza la importancia de un texto."""
        inp_tensor = text_to_ids(text).to(self.device)
        
        with torch.no_grad():
            _, metrics = self.model(inp_tensor, return_metrics=True)
        
        return metrics['gate_value'] * 100
    
    def process_input(self, user_text):
        """Procesa la entrada del usuario."""
        # Analizar importancia
        importance = self.analyze_importance(user_text)
        is_important = importance > self.threshold
        
        # Feedback visual
        bar_len = int(importance / 5)  # 20 chars para 100%
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"\n   ┌{'─'*40}┐")
        print(f"   │ 🔍 Análisis de Importancia            │")
        print(f"   ├{'─'*40}┤")
        print(f"   │ Gate: [{bar}] {importance:>6.2f}% │")
        
        if is_important:
            print(f"   │ Estado: 🟢 IMPORTANTE → GUARDAR       │")
            print(f"   └{'─'*40}┘")
            self._save_memory(user_text, importance)
            return "💾 He guardado eso en mi memoria permanente."
        else:
            print(f"   │ Estado: 🔴 TRIVIAL → IGNORAR          │")
            print(f"   └{'─'*40}┘")
            return "👂 Escuchado, pero no es relevante para guardar."
    
    def _save_memory(self, text, score):
        """Guarda un recuerdo en la memoria permanente."""
        entry = {
            "id": len(self.memory_db) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content": text,
            "importance_score": round(score, 2)
        }
        self.memory_db.append(entry)
        
        # Guardar en disco
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory_db, f, indent=2, ensure_ascii=False)
    
    def show_all_memories(self):
        """Muestra todos los recuerdos guardados."""
        print(f"\n{'='*60}")
        print(f"🧠 MEMORIA PERMANENTE ({len(self.memory_db)} recuerdos)")
        print(f"{'='*60}")
        
        if not self.memory_db:
            print("   (La memoria está vacía)")
        else:
            for mem in self.memory_db:
                score = mem['importance_score']
                emoji = "🔥" if score > 90 else "⭐" if score > 70 else "📝"
                print(f"\n   {emoji} #{mem['id']} [{mem['timestamp']}]")
                print(f"      \"{mem['content']}\"")
                print(f"      Importancia: {score:.1f}%")
        
        print(f"\n{'='*60}")
    
    def clear_memory(self):
        """Borra toda la memoria."""
        self.memory_db = []
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print("🗑️ Memoria borrada.")
    
    def search_memory(self, query):
        """Busca en la memoria."""
        query_lower = query.lower()
        results = [m for m in self.memory_db if query_lower in m['content'].lower()]
        
        if results:
            print(f"\n🔎 Encontrados {len(results)} recuerdos con '{query}':")
            for mem in results:
                print(f"   #{mem['id']}: \"{mem['content']}\"")
        else:
            print(f"   No encontré nada con '{query}'")


# --- 4. BUCLE DE CHAT ---

def print_help():
    """Muestra la ayuda."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  📚 COMANDOS DISPONIBLES                                     ║
╠══════════════════════════════════════════════════════════════╣
║  ver        - Muestra todos los recuerdos guardados          ║
║  buscar X   - Busca 'X' en la memoria                        ║
║  borrar     - Borra toda la memoria                          ║
║  test       - Ejecuta pruebas de demostración                ║
║  ayuda      - Muestra esta ayuda                             ║
║  salir      - Termina el programa                            ║
╚══════════════════════════════════════════════════════════════╝
""")


def run_demo_test(bot):
    """Ejecuta pruebas de demostración."""
    print(f"\n{'='*60}")
    print("🧪 PRUEBA DE DEMOSTRACIÓN")
    print(f"{'='*60}")
    
    test_phrases = [
        ("Hoy hace buen día", "Debería IGNORAR (ruido)"),
        ("Me llamo Enrique", "Debería GUARDAR (identidad)"),
        ("La contraseña de wifi es 1234", "Debería GUARDAR (información sensible)"),
        ("Jajajaja", "Debería IGNORAR (ruido)"),
        ("Mi número de teléfono es 555-1234", "Debería GUARDAR (dato personal)"),
        ("Qué hora es", "Debería IGNORAR (pregunta trivial)"),
        ("Recuerda que mañana tengo cita", "Debería GUARDAR (recordatorio)"),
    ]
    
    for phrase, expected in test_phrases:
        print(f"\n📝 Frase: \"{phrase}\"")
        print(f"   Esperado: {expected}")
        response = bot.process_input(phrase)
        print(f"   Bot: {response}")
    
    print(f"\n{'='*60}")
    print("✅ Prueba completada. Escribe 'ver' para ver la memoria.")
    print(f"{'='*60}")


def main():
    """Función principal."""
    # Ruta al modelo entrenado
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dynamic_chat_detector_v3.pt")  # v3: 10k+ datos
    
    # Verificar que existe
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No se encontró el modelo en: {MODEL_PATH}")
        print("   Asegúrate de haber ejecutado test_dynamic_chat_v2.py primero.")
        sys.exit(1)
    
    # Inicializar sistema
    bot = MemoryKeeperSystem(MODEL_PATH)
    
    # Mostrar instrucciones
    print("💬 CHAT CON MEMORIA SELECTIVA INFINITO")
    print("─" * 60)
    print("El modelo analiza cada frase y decide si es importante.")
    print("Solo guarda información relevante (identidad, claves, etc.)")
    print("\nEscribe 'ayuda' para ver comandos. 'salir' para terminar.")
    print("─" * 60)
    
    # Bucle principal
    while True:
        try:
            user_input = input("\n👤 Tú > ").strip()
            
            if not user_input:
                continue
            
            # Comandos especiales
            cmd = user_input.lower()
            
            if cmd in ['salir', 'exit', 'quit']:
                break
            elif cmd == 'ver':
                bot.show_all_memories()
            elif cmd == 'ayuda' or cmd == 'help':
                print_help()
            elif cmd == 'borrar':
                confirm = input("   ¿Seguro? (s/n): ")
                if confirm.lower() == 's':
                    bot.clear_memory()
            elif cmd == 'test':
                run_demo_test(bot)
            elif cmd.startswith('buscar '):
                query = user_input[7:]
                bot.search_memory(query)
            else:
                # Procesar como entrada normal
                response = bot.process_input(user_input)
                print(f"\n🤖 Bot > {response}")
                
        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            break
    
    # Despedida
    print(f"\n{'='*60}")
    print(f"👋 ¡Hasta luego!")
    print(f"   Recuerdos guardados: {len(bot.memory_db)}")
    if bot.memory_db:
        print(f"   Archivo: {bot.db_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
