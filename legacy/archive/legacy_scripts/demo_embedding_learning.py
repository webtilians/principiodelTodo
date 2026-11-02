#!/usr/bin/env python3
"""
🎮 DEMO: Sistema de Aprendizaje Continuo (Embeddings)
======================================================

Demo interactivo del sistema de aprendizaje continuo
basado en embeddings TF-IDF.

CARACTERÍSTICAS:
✅ Reconocimiento de patrones en tiempo real
✅ Memoria persistente (JSON)
✅ Diferenciación semántica
✅ No requiere modelo entrenado
✅ Rápido y eficiente

COMANDOS:
- Escribe cualquier texto para procesarlo
- 'save' - Guardar memoria a disco
- 'stats' - Ver estadísticas
- 'list' - Listar todos los patrones
- 'compare|texto1|texto2' - Comparar dos textos
- 'clear' - Limpiar memoria
- 'exit' - Salir
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough
from continuous_learning_embeddings import EmbeddingBasedLearningServer


def main():
    """Función principal del demo"""
    
    print("\n" + "="*70)
    print("🎮 DEMO: Sistema de Aprendizaje Continuo (Embeddings)")
    print("="*70)
    print("📚 Basado en embeddings TF-IDF")
    print("🎯 Reconocimiento de patrones en tiempo real")
    print("="*70)
    
    # Setup
    print("\n🔧 Inicializando sistema...")
    
    class Args:
        def __init__(self):
            self.input_dim = 257
            self.batch_size = 4
            self.hidden_dim = 512
            self.attention_heads = 8
            self.memory_slots = 256
            self.lr = 1e-3
            self.seed = 42
            self.input_text = None
            self.text_mode = True
            self.max_iter = 1000
            self.comparative = False
            self.comparative_iterations = 100
            self.bootstrap_samples = 1000
    
    try:
        infinito = InfinitoV51ConsciousnessBreakthrough(Args())
        server = EmbeddingBasedLearningServer(infinito, similarity_threshold=0.85)
        
        # Intentar cargar memoria existente
        try:
            server.load_memory()
            print(f"\n✅ Memoria cargada desde phi_memory_bank.json")
        except:
            print(f"\nℹ️  Memoria nueva - se creará al guardar")
        
        # Ejecutar loop interactivo
        server.run_interactive_loop()
        
        # Guardar al salir
        print(f"\n💾 Guardando memoria...")
        server.save_memory()
        print(f"✅ Memoria guardada en phi_memory_bank.json")
        
    except KeyboardInterrupt:
        print(f"\n\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
