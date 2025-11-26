#!/usr/bin/env python3
"""
🔮 INFINITO JARVIS - Asistente con Memoria Selectiva
=====================================================

Sistema completo que combina:
1. Tu modelo Infinito (Gate Dinámico) para filtrar qué recordar
2. Un LLM (OpenAI/Simulación) para generar respuestas inteligentes
3. Memoria persistente en JSON

El flujo:
  Usuario dice algo → Infinito analiza importancia → 
  Si importante: GUARDA → Construye prompt con memoria → LLM responde

Es como tener un Jarvis que RECUERDA lo importante sobre ti.
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


# --- CONFIGURACIÓN LLM ---
# Cambia USE_OPENAI a True y pon tu API Key para usar GPT real
USE_OPENAI = False
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-...")

# Intentar importar OpenAI si está disponible
openai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI conectado")
    except ImportError:
        print("⚠️ OpenAI no instalado. Ejecuta: pip install openai")
        USE_OPENAI = False
    except Exception as e:
        print(f"⚠️ Error conectando OpenAI: {e}")
        USE_OPENAI = False


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


# --- SISTEMA JARVIS ---

class JarvisSystem:
    """Asistente inteligente con memoria selectiva."""
    
    def __init__(self, keeper_model_path, db_file="memoria_jarvis.json"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.db_file = db_file
        self.conversation_history = []
        
        print(f"\n{'='*60}")
        print(f"🔮 INFINITO JARVIS")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"LLM: {'OpenAI GPT' if USE_OPENAI else 'Simulación'}")
        
        # 1. Cargar modelo Infinito (el "portero" de la memoria)
        self._load_keeper(keeper_model_path)
        
        # 2. Cargar memorias existentes
        self._load_memories()
        
        print(f"{'='*60}\n")
    
    def _load_keeper(self, model_path):
        """Carga el modelo Infinito que decide qué guardar."""
        print(f"\n🧠 Cargando Infinito Gate...")
        
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
            print(f"📂 {len(self.memories)} recuerdos cargados")
        else:
            self.memories = []
            print("✨ Nueva memoria iniciada")
    
    def _analyze_importance(self, text):
        """Usa el modelo Infinito para medir importancia."""
        inp = text_to_ids(text).to(self.device)
        with torch.no_grad():
            _, metrics = self.keeper(inp, return_metrics=True)
        return metrics['gate_value'] * 100
    
    def _save_memory(self, text, score, category="general"):
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
    
    def _categorize(self, text):
        """Categoriza el tipo de información."""
        text_lower = text.lower()
        
        # Si es una pregunta, no es información para guardar
        if text_lower.endswith('?') or text_lower.startswith(('qué', 'que', 'cómo', 'como', 'cuál', 'cual')):
            return "pregunta"  # Las preguntas no deberían guardarse
        
        if any(x in text_lower for x in ['me llamo', 'mi nombre es', 'soy ', 'llámame']):
            return "identidad"
        elif any(x in text_lower for x in ['contraseña', 'clave', 'password', 'pin', 'secreto']):
            return "credencial"
        elif any(x in text_lower for x in ['teléfono', 'email', 'correo', 'dirección', 'vivo en']):
            return "contacto"
        elif any(x in text_lower for x in ['recuerda', 'no olvides', 'importante', 'mañana', 'cita']):
            return "recordatorio"
        elif any(x in text_lower for x in ['me gusta', 'prefiero', 'favorito', 'odio']):
            return "preferencia"
        else:
            return "general"
    
    def _construct_system_prompt(self):
        """Construye el prompt del sistema con la memoria."""
        
        # Agrupar memorias por categoría
        categories = {}
        for mem in self.memories[-10:]:  # Últimos 10 recuerdos
            cat = mem.get('category', 'general')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(mem['content'])
        
        # Construir bloque de memoria
        memory_block = ""
        if self.memories:
            memory_block = "\n📚 INFORMACIÓN QUE CONOCES SOBRE EL USUARIO:\n"
            
            for cat, items in categories.items():
                emoji = {
                    'identidad': '👤',
                    'credencial': '🔐',
                    'contacto': '📞',
                    'recordatorio': '📌',
                    'preferencia': '❤️',
                    'general': '📝'
                }.get(cat, '📝')
                
                memory_block += f"\n{emoji} {cat.upper()}:\n"
                for item in items:
                    memory_block += f"   • {item}\n"
        
        system_prompt = f"""Eres Infinito, un asistente personal inteligente con memoria a largo plazo.

CARACTERÍSTICAS:
- Eres amable, conciso y útil
- RECUERDAS información importante sobre el usuario
- Usas esa información para personalizar tus respuestas
- Si el usuario te dio su nombre, ÚSALO
- Si te compartió datos, refiérete a ellos cuando sea relevante
{memory_block}

INSTRUCCIONES:
1. Responde de forma natural y conversacional
2. Si tienes información relevante en tu memoria, ÚSALA
3. No inventes información que no tienes
4. Sé breve pero completo
"""
        return system_prompt
    
    def _get_llm_response(self, user_text, system_prompt):
        """Obtiene respuesta del LLM."""
        
        if USE_OPENAI and openai_client:
            try:
                # Construir mensajes con historial
                messages = [{"role": "system", "content": system_prompt}]
                
                # Añadir últimos turnos de conversación
                for turn in self.conversation_history[-4:]:
                    messages.append({"role": "user", "content": turn['user']})
                    messages.append({"role": "assistant", "content": turn['assistant']})
                
                messages.append({"role": "user", "content": user_text})
                
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content
                
            except Exception as e:
                return f"Error con OpenAI: {e}"
        else:
            # Modo simulación - mostrar lo que vería el LLM
            return self._simulate_response(user_text, system_prompt)
    
    def _simulate_response(self, user_text, system_prompt):
        """Simula una respuesta inteligente basada en la memoria."""
        
        # Buscar información del usuario en memoria
        user_name = None
        user_data = {'credenciales': [], 'contactos': [], 'recordatorios': [], 'preferencias': []}
        
        for mem in self.memories:
            content = mem['content']
            content_lower = content.lower()
            category = mem.get('category', 'general')
            
            # Extraer nombre
            if category == 'identidad' or 'me llamo' in content_lower or 'mi nombre' in content_lower:
                for pattern in ['me llamo ', 'mi nombre es ', 'soy ']:
                    if pattern in content_lower:
                        idx = content_lower.find(pattern) + len(pattern)
                        name_part = content[idx:].split()[0] if content[idx:].split() else None
                        if name_part:
                            user_name = name_part.strip('.,!?¿¡')
                            break
            
            # Categorizar datos
            if category == 'credencial':
                user_data['credenciales'].append(content)
            elif category == 'contacto':
                user_data['contactos'].append(content)
            elif category == 'recordatorio':
                user_data['recordatorios'].append(content)
            elif category == 'preferencia':
                user_data['preferencias'].append(content)
        
        # Generar respuesta contextual
        text_lower = user_text.lower()
        
        # Saludos
        if any(x in text_lower for x in ['hola', 'buenos', 'buenas', 'hey']):
            if user_name:
                return f"¡Hola {user_name}! 😊 ¿En qué puedo ayudarte hoy?"
            return "¡Hola! Soy Infinito, tu asistente con memoria. ¿Cómo te llamas?"
        
        # Preguntas sobre identidad
        elif any(x in text_lower for x in ['cómo me llamo', 'como me llamo', 'mi nombre', 'quién soy', 'quien soy']):
            if user_name:
                return f"Te llamas {user_name}. Lo recuerdo perfectamente desde que me lo dijiste 😊"
            return "Todavía no me has dicho tu nombre. ¿Cómo te llamas?"
        
        # Preguntas sobre memoria
        elif any(x in text_lower for x in ['qué sabes', 'que sabes', 'qué recuerdas', 'que recuerdas']):
            if self.memories:
                parts = [f"Tengo {len(self.memories)} recuerdos sobre ti"]
                if user_name:
                    parts.append(f"Sé que te llamas {user_name}")
                if user_data['credenciales']:
                    parts.append(f"Tengo {len(user_data['credenciales'])} credenciales guardadas")
                if user_data['recordatorios']:
                    parts.append(f"Tienes {len(user_data['recordatorios'])} recordatorios")
                return ". ".join(parts) + ". Escribe 'ver memoria' para ver todo."
            return "Aún no tengo recuerdos tuyos. ¡Cuéntame algo importante sobre ti!"
        
        # Agradecimientos
        elif any(x in text_lower for x in ['gracias', 'thanks', 'thx']):
            if user_name:
                return f"¡De nada, {user_name}! Siempre a tu servicio 🙌"
            return "¡Con gusto! Para eso estoy 😊"
        
        # Despedidas
        elif any(x in text_lower for x in ['adiós', 'adios', 'chao', 'bye', 'hasta luego']):
            if user_name:
                return f"¡Hasta pronto, {user_name}! Recordaré todo lo que me dijiste 👋"
            return "¡Hasta luego! Fue un gusto chatear contigo 👋"
        
        # Confirmación de que guardó algo
        elif any(x in text_lower for x in ['me llamo', 'mi nombre es', 'soy ']):
            # Extraer el nombre que acaba de decir
            for pattern in ['me llamo ', 'mi nombre es ', 'soy ']:
                if pattern in text_lower:
                    idx = text_lower.find(pattern) + len(pattern)
                    new_name = user_text[idx:].split()[0] if user_text[idx:].split() else "amigo"
                    new_name = new_name.strip('.,!?¿¡')
                    return f"¡Encantado de conocerte, {new_name}! 😊 Ya guardé tu nombre en mi memoria."
            return "¡Encantado! Ya guardé tu nombre."
        
        elif any(x in text_lower for x in ['contraseña', 'clave', 'password', 'secreto']):
            return "🔐 Guardado de forma segura en mi memoria. No lo olvidaré."
        
        elif any(x in text_lower for x in ['recuerda', 'no olvides']):
            return "📌 ¡Anotado! Te lo recordaré cuando sea necesario."
        
        elif any(x in text_lower for x in ['teléfono', 'email', 'correo']):
            return "📞 Información de contacto guardada. La tendré presente."
        
        else:
            # Respuesta genérica pero personalizada
            if user_name:
                return f"Entendido, {user_name}. ¿Hay algo más en lo que pueda ayudarte?"
            return "Entendido. ¿Hay algo más que quieras contarme o preguntarme?"
    
    def chat(self, user_text):
        """Procesa un mensaje del usuario."""
        
        # PASO 1: Analizar importancia con Infinito
        importance = self._analyze_importance(user_text)
        is_important = importance > 50.0
        
        # Feedback visual del gate
        bar_len = int(importance / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"\n   ┌{'─'*44}┐")
        print(f"   │ 🔍 Gate: [{bar}] {importance:>6.1f}% ", end="")
        
        if is_important:
            category = self._categorize(user_text)
            # No guardar preguntas (aunque tengan alta importancia)
            if category != "pregunta":
                self._save_memory(user_text, importance, category)
                print(f"🟢 │")
                print(f"   │ 💾 Guardado como: {category:<24} │")
            else:
                print(f"🟡 │")
                print(f"   │ ❓ Pregunta detectada (no se guarda)      │")
        else:
            print(f"🔴 │")
        print(f"   └{'─'*44}┘")
        
        # PASO 2: Construir prompt con memoria
        system_prompt = self._construct_system_prompt()
        
        # PASO 3: Obtener respuesta del LLM
        response = self._get_llm_response(user_text, system_prompt)
        
        # Guardar en historial de conversación
        self.conversation_history.append({
            'user': user_text,
            'assistant': response,
            'importance': importance
        })
        
        return response
    
    def show_memory(self):
        """Muestra la memoria completa."""
        print(f"\n{'='*60}")
        print(f"🧠 MEMORIA DE INFINITO ({len(self.memories)} recuerdos)")
        print(f"{'='*60}")
        
        if not self.memories:
            print("   (La memoria está vacía)")
        else:
            for mem in self.memories:
                emoji = {
                    'identidad': '👤',
                    'credencial': '🔐',
                    'contacto': '📞',
                    'recordatorio': '📌',
                    'preferencia': '❤️',
                    'general': '📝'
                }.get(mem.get('category', 'general'), '📝')
                
                print(f"\n   {emoji} #{mem['id']} [{mem['timestamp']}]")
                print(f"      \"{mem['content']}\"")
                print(f"      Importancia: {mem['score']}% | Tipo: {mem.get('category', 'general')}")
        
        print(f"\n{'='*60}")
    
    def show_prompt(self):
        """Muestra el prompt actual que se enviaría al LLM."""
        prompt = self._construct_system_prompt()
        print(f"\n{'='*60}")
        print("🤖 SYSTEM PROMPT ACTUAL (lo que ve el LLM):")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}")


def print_help():
    """Muestra la ayuda."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🔮 COMANDOS DE INFINITO JARVIS                              ║
╠══════════════════════════════════════════════════════════════╣
║  ver memoria  - Muestra todos los recuerdos                  ║
║  ver prompt   - Muestra el prompt que ve el LLM              ║
║  borrar       - Borra toda la memoria                        ║
║  ayuda        - Muestra esta ayuda                           ║
║  salir        - Termina el programa                          ║
╠══════════════════════════════════════════════════════════════╣
║  💡 PRUEBA DECIR:                                            ║
║  • "Hola, me llamo [Tu Nombre]"                              ║
║  • "La contraseña del wifi es 1234"                          ║
║  • "Mi email es ejemplo@mail.com"                            ║
║  • "Recuerda que mañana tengo reunión"                       ║
║  • "¿Cómo me llamo?" (después de decir tu nombre)            ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    """Función principal."""
    
    # Ruta al modelo
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dynamic_chat_detector_v2.pt")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No se encontró el modelo en: {MODEL_PATH}")
        print("   Ejecuta primero: python test_dynamic_chat_v2.py")
        sys.exit(1)
    
    # Inicializar Jarvis
    jarvis = JarvisSystem(MODEL_PATH)
    
    # Instrucciones
    print("💬 Chatea conmigo. Recordaré lo importante.")
    print("   Escribe 'ayuda' para ver comandos.")
    print("─" * 60)
    
    while True:
        try:
            user_input = input("\n👤 Tú > ").strip()
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            
            # Comandos especiales
            if cmd in ['salir', 'exit', 'quit']:
                break
            elif cmd == 'ver memoria':
                jarvis.show_memory()
                continue
            elif cmd == 'ver prompt':
                jarvis.show_prompt()
                continue
            elif cmd in ['ayuda', 'help']:
                print_help()
                continue
            elif cmd == 'borrar':
                confirm = input("   ¿Borrar toda la memoria? (s/n): ")
                if confirm.lower() == 's':
                    jarvis.memories = []
                    with open(jarvis.db_file, 'w') as f:
                        json.dump([], f)
                    print("   🗑️ Memoria borrada")
                continue
            
            # Chat normal
            response = jarvis.chat(user_input)
            print(f"\n🤖 Infinito > {response}")
            
        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            break
    
    # Despedida
    print(f"\n{'='*60}")
    print(f"👋 ¡Hasta luego!")
    print(f"   Recuerdos guardados: {len(jarvis.memories)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
