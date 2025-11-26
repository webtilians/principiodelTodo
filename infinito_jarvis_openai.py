#!/usr/bin/env python3
"""
🔮 INFINITO JARVIS - CONECTADO A OPENAI
========================================

Sistema completo que combina:
1. Tu modelo Infinito (Gate Dinámico) para filtrar qué recordar
2. OpenAI GPT para generar respuestas inteligentes
3. Memoria persistente en JSON

¡Ahora con respuestas REALES de GPT!
"""

import torch
import torch.nn as nn
import json
import os
import sys
from datetime import datetime
from openai import OpenAI

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from infinito_v5_2_refactored import InfinitoV52Refactored

# --- CONFIGURACIÓN OPENAI ---
# Configura tu API key como variable de entorno: set OPENAI_API_KEY=tu-api-key
API_KEY = os.environ.get("OPENAI_API_KEY", "")

if API_KEY.startswith("sk-") and len(API_KEY) > 10:
    client = OpenAI(api_key=API_KEY)
    print("✅ Conexión con OpenAI configurada.")
else:
    print("⚠️ API Key no configurada. Usa: set OPENAI_API_KEY=tu-api-key")
    client = None


# --- MODELO INFINITO (Gate Dinámico) ---
class InfinitoDynamicChat(InfinitoV52Refactored):
    """Modelo con gate dinámico para detectar información importante."""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_dynamic_gate'] = False
        super().__init__(*args, **kwargs)
        
        if hasattr(self, 'memory_gate'):
            del self.memory_gate
        
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, 1)
        )
    
    def forward(self, input_ids, return_metrics=False):
        batch_size, seq_len = input_ids.shape
        
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding[:, :seq_len, :]
        hidden = self.embedding_dropout(hidden)
        
        for attn, ff, ln1, ln2 in zip(
            self.attention_layers, self.ff_layers, 
            self.layer_norms_1, self.layer_norms_2
        ):
            attn_out, _ = attn(hidden)
            hidden = ln1(hidden + attn_out)
            ff_out = ff(hidden)
            hidden = ln2(hidden + ff_out)
        
        sentence_context = hidden.mean(dim=1)
        gate_logit = self.gate_network(sentence_context)
        gate_open_pct = torch.sigmoid(gate_logit)
        
        logits = self.output_projection(hidden)
        
        if return_metrics:
            return logits, {'gate_value': gate_open_pct.mean().item()}
        return logits, None


def text_to_ids(text, seq_len=32):
    """Convierte texto a IDs ASCII."""
    ids = [ord(c) % 256 for c in text]
    if len(ids) < seq_len:
        ids = ids + [0] * (seq_len - len(ids))
    else:
        ids = ids[:seq_len]
    return torch.tensor([ids])


# --- SISTEMA JARVIS CON OPENAI ---
class JarvisSystem:
    """Asistente inteligente con memoria selectiva y GPT."""
    
    def __init__(self, keeper_model_path, db_file="memoria_infinito.json"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.db_file = db_file
        
        print(f"\n{'='*60}")
        print(f"🔮 INFINITO JARVIS + OPENAI")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        
        # 1. Cargar modelo Infinito (el "portero" de la memoria)
        self._load_keeper(keeper_model_path)
        
        # 2. Cargar memorias existentes
        self._load_memories()
        
        print(f"{'='*60}\n")
    
    def _load_keeper(self, model_path):
        """Carga el modelo Infinito."""
        print(f"🧠 Cargando Infinito Gate...")
        
        self.keeper = InfinitoDynamicChat(
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
                self.keeper.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.keeper.load_state_dict(checkpoint, strict=False)
            self.keeper.eval()
            print("   ✅ Modelo cargado")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            sys.exit(1)
    
    def _load_memories(self):
        """Carga las memorias del archivo JSON."""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)
            print(f"📚 {len(self.memories)} recuerdos cargados")
        else:
            self.memories = []
            print("✨ Nueva memoria iniciada")
    
    def _es_pregunta(self, texto):
        """Detecta si el usuario está preguntando."""
        t = texto.lower().strip()
        if '?' in t or '¿' in t:
            return True
        starts = ['que ', 'qué ', 'como ', 'cómo ', 'cual ', 'cuál ', 'quien ', 'quién ', 
                  'donde ', 'dónde ', 'cuando ', 'cuándo ', 'cuanto ', 'cuánto ', 
                  'por que ', 'por qué ', 'y como ', 'y quien ', 'sabes ', 'recuerdas ']
        for s in starts:
            if t.startswith(s):
                return True
        return False
    
    def _analyze_importance(self, text):
        """Usa el modelo Infinito para medir importancia."""
        inp = text_to_ids(text).to(self.device)
        with torch.no_grad():
            _, metrics = self.keeper(inp, return_metrics=True)
        return metrics['gate_value'] * 100
    
    def _categorize(self, text):
        """Categoriza el tipo de información."""
        text_lower = text.lower()
        
        if any(x in text_lower for x in ['me llamo', 'mi nombre es', 'soy ', 'llámame']):
            return "👤 identidad"
        elif any(x in text_lower for x in ['contraseña', 'clave', 'password', 'pin', 'secreto']):
            return "🔐 credencial"
        elif any(x in text_lower for x in ['teléfono', 'email', 'correo', 'dirección', 'vivo en']):
            return "📞 contacto"
        elif any(x in text_lower for x in ['recuerda', 'no olvides', 'importante', 'mañana', 'cita']):
            return "📌 recordatorio"
        elif any(x in text_lower for x in ['me gusta', 'prefiero', 'favorito', 'odio']):
            return "❤️ preferencia"
        elif any(x in text_lower for x in ['mi primo', 'mi hermano', 'mi madre', 'mi padre', 'mi amigo']):
            return "👨‍👩‍👧 familia"
        else:
            return "📝 general"
    
    def _save_memory(self, text, score, category):
        """Guarda un recuerdo importante."""
        entry = {
            "id": len(self.memories) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "content": text,
            "score": round(score, 1),
            "category": category
        }
        self.memories.append(entry)
        
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, indent=2, ensure_ascii=False)
    
    def _construct_prompt(self):
        """Crea el prompt del sistema con la memoria."""
        memory_block = "NO TIENES RECUERDOS PREVIOS DEL USUARIO."
        
        if self.memories:
            memory_block = "📚 MEMORIA A LARGO PLAZO (Hechos que conoces sobre el usuario):\n"
            for mem in self.memories[-20:]:  # Últimos 20 recuerdos
                memory_block += f"  • {mem['content']} [{mem.get('category', 'general')}] (Guardado: {mem['timestamp']})\n"
        
        system_prompt = f"""Eres Infinito, un asistente personal avanzado con MEMORIA PERSISTENTE.

{memory_block}

INSTRUCCIONES CRÍTICAS:
1. USA la memoria anterior para personalizar TODAS tus respuestas
2. Si el usuario pregunta algo que está en tu memoria (nombres, claves, datos), RESPONDE DIRECTAMENTE con esa información
3. Si te preguntan "¿cómo me llamo?" y tienes su nombre en memoria, DILO
4. Si te preguntan sobre familiares/amigos y tienes esa info, ÚSALA
5. Sé breve, útil y amable
6. NUNCA digas "no tengo acceso a esa información" si está en tu memoria
7. Responde en español

Eres como Jarvis de Iron Man, pero con memoria real sobre tu usuario."""
        
        return system_prompt
    
    def chat(self, user_text):
        """Procesa un mensaje del usuario."""
        
        # --- FASE 1: INFINITO KEEPER (El Filtro) ---
        importance = self._analyze_importance(user_text)
        is_question = self._es_pregunta(user_text)
        
        # Lógica de decisión
        should_save = (importance > 50.0) and (not is_question)
        
        # Feedback visual
        bar_len = int(importance / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"\n   ┌{'─'*48}┐")
        print(f"   │ 🔍 Gate: [{bar}] {importance:>6.1f}% ", end="")
        
        if should_save:
            category = self._categorize(user_text)
            self._save_memory(user_text, importance, category)
            print(f"🟢 │")
            print(f"   │ 💾 Guardado: {category:<30} │")
        elif is_question and importance > 50.0:
            print(f"🟡 │")
            print(f"   │ ❓ Pregunta detectada (consultando memoria)  │")
        else:
            print(f"🔴 │")
        print(f"   └{'─'*48}┘")
        
        # --- FASE 2: OPENAI GPT (El Orador) ---
        try:
            prompt_system = self._construct_prompt()
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Puedes cambiar a "gpt-4" si prefieres
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error de OpenAI: {e}"
    
    def show_memory(self):
        """Muestra la memoria completa."""
        print(f"\n{'='*60}")
        print(f"🧠 MEMORIA DE INFINITO ({len(self.memories)} recuerdos)")
        print(f"{'='*60}")
        
        if not self.memories:
            print("   (La memoria está vacía)")
        else:
            for mem in self.memories:
                cat = mem.get('category', '📝 general')
                print(f"\n   {cat} #{mem['id']} [{mem['timestamp']}]")
                print(f"      \"{mem['content']}\"")
                print(f"      Importancia: {mem['score']}%")
        
        print(f"\n{'='*60}")


def main():
    """Función principal."""
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dynamic_chat_detector_v2.pt")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No se encontró el modelo en: {MODEL_PATH}")
        print("   Ejecuta primero: python test_dynamic_chat_v2.py")
        sys.exit(1)
    
    # Inicializar Jarvis
    jarvis = JarvisSystem(MODEL_PATH)
    
    # Instrucciones
    print("💬 Chatea conmigo. Ahora con GPT REAL.")
    print("   Escribe 'ver memoria' para ver recuerdos.")
    print("   Escribe 'salir' para terminar.")
    print("─" * 60)
    
    while True:
        try:
            user_input = input("\n👤 Tú > ").strip()
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            
            if cmd in ['salir', 'exit', 'quit']:
                break
            elif cmd == 'ver memoria':
                jarvis.show_memory()
                continue
            elif cmd == 'borrar memoria':
                confirm = input("   ¿Borrar toda la memoria? (s/n): ")
                if confirm.lower() == 's':
                    jarvis.memories = []
                    with open(jarvis.db_file, 'w') as f:
                        json.dump([], f)
                    print("   🗑️ Memoria borrada")
                continue
            
            # Chat con GPT
            print("   🤔 Pensando...", end="\r")
            response = jarvis.chat(user_input)
            print(f"\n🤖 Infinito > {response}")
            
        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            break
    
    print(f"\n{'='*60}")
    print(f"👋 ¡Hasta luego!")
    print(f"   Recuerdos guardados: {len(jarvis.memories)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
