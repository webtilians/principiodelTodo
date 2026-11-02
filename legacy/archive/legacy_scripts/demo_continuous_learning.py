#!/usr/bin/env python3
"""
🎮 DEMO: Sistema de Aprendizaje Continuo
========================================

Script interactivo para probar el sistema de aprendizaje continuo.

Uso:
    python demo_continuous_learning.py

El sistema:
1. Carga el modelo INFINITO V5.1
2. Inicia el servidor de aprendizaje continuo
3. Acepta inputs de texto sin parar
4. Reconoce patrones que ya conoce
5. Aprende nuevos patrones automáticamente
"""

import sys
import os
import argparse
import torch

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Importar INFINITO
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough

# Importar sistema de aprendizaje continuo
from continuous_learning import ContinuousLearningServer, PhiPatternExtractor, PhiMemoryBank


def create_simple_args():
    """Crear argumentos mínimos para INFINITO"""
    class Args:
        def __init__(self):
            self.input_dim = 257
            self.batch_size = 4
            self.hidden_dim = 512
            self.attention_heads = 8
            self.memory_slots = 256
            self.lr = 1e-3
            self.seed = 42
            self.input_text = None  # Se establece dinámicamente
            self.text_mode = True
            self.max_iter = 1000
            self.comparative = False
            self.comparative_iterations = 100
            self.bootstrap_samples = 1000
    
    return Args()


def run_quick_test():
    """Test rápido con 5 textos predefinidos"""
    
    print("\n" + "="*70)
    print("🧪 TEST RÁPIDO - 5 TEXTOS PREDEFINIDOS")
    print("="*70)
    
    # Inicializar sistema
    print("\n🚀 Inicializando INFINITO V5.1...")
    args = create_simple_args()
    infinito = InfinitoV51ConsciousnessBreakthrough(args)
    
    # Crear servidor continuo
    print("🔄 Creando servidor de aprendizaje continuo...")
    server = ContinuousLearningServer(infinito, similarity_threshold=0.90)
    
    # Textos de prueba
    test_texts = [
        "mi perro es rojo",
        "mi perro es verde",      # Similar al anterior
        "mi perro es azul",       # Similar al anterior
        "yo pienso luego existo", # Totalmente diferente
        "mi gato es rojo",        # Parecido al primero pero con "gato"
    ]
    
    print(f"\n📝 Procesando {len(test_texts)} textos de prueba...")
    
    results = []
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'─'*70}")
        print(f"TEST {i}/{len(test_texts)}")
        result = server.process_input(text)
        results.append(result)
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"📊 RESUMEN DEL TEST")
    print(f"{'='*70}")
    
    new_count = sum(1 for r in results if r.get('status') == 'NEW')
    recognized_count = sum(1 for r in results if r.get('status') == 'RECOGNIZED')
    
    print(f"   Textos procesados: {len(results)}")
    print(f"   Patrones nuevos: {new_count}")
    print(f"   Patrones reconocidos: {recognized_count}")
    
    print(f"\n{'='*70}")
    print(f"📋 DETALLE POR TEXTO")
    print(f"{'='*70}")
    
    for i, (text, result) in enumerate(zip(test_texts, results), 1):
        status_emoji = "🆕" if result.get('status') == 'NEW' else "🎯"
        print(f"\n{i}. {status_emoji} '{text}'")
        print(f"   Status: {result.get('status')}")
        
        if result.get('status') == 'RECOGNIZED':
            print(f"   Similar a: '{result.get('original_text')}'")
            print(f"   Similitud: {result.get('similarity', 0):.1%}")
    
    # Guardar memoria
    print(f"\n💾 Guardando memoria del test...")
    server.memory_bank.save_to_disk('phi_memory_bank_test.json')
    
    print(f"\n✅ Test completado!")


def run_interactive_mode():
    """Modo interactivo - acepta inputs del usuario"""
    
    print("\n" + "="*70)
    print("🎮 MODO INTERACTIVO")
    print("="*70)
    
    # Inicializar sistema
    print("\n🚀 Inicializando INFINITO V5.1...")
    args = create_simple_args()
    infinito = InfinitoV51ConsciousnessBreakthrough(args)
    
    # Crear servidor continuo
    print("🔄 Creando servidor de aprendizaje continuo...")
    server = ContinuousLearningServer(infinito, similarity_threshold=0.90)
    
    # Iniciar loop interactivo
    server.run_interactive_loop()


def run_batch_mode(texts_file: str):
    """
    Modo batch - procesar textos desde archivo
    
    Args:
        texts_file: Ruta al archivo con textos (uno por línea)
    """
    
    print("\n" + "="*70)
    print(f"📄 MODO BATCH - Archivo: {texts_file}")
    print("="*70)
    
    # Leer textos
    if not os.path.exists(texts_file):
        print(f"❌ Archivo no encontrado: {texts_file}")
        return
    
    with open(texts_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"📝 {len(texts)} textos cargados")
    
    # Inicializar sistema
    print("\n🚀 Inicializando INFINITO V5.1...")
    args = create_simple_args()
    infinito = InfinitoV51ConsciousnessBreakthrough(args)
    
    # Crear servidor continuo
    print("🔄 Creando servidor de aprendizaje continuo...")
    server = ContinuousLearningServer(infinito, similarity_threshold=0.90)
    
    # Procesar todos los textos
    results = []
    for i, text in enumerate(texts, 1):
        print(f"\n{'─'*70}")
        print(f"PROCESANDO {i}/{len(texts)}")
        result = server.process_input(text)
        results.append(result)
    
    # Mostrar estadísticas finales
    print(f"\n")
    server.show_stats()
    
    # Guardar memoria
    print(f"\n💾 Guardando memoria...")
    server.memory_bank.save_to_disk()


def main():
    """Función principal"""
    
    parser = argparse.ArgumentParser(
        description="🔄 Sistema de Aprendizaje Continuo - INFINITO V5.1"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'interactive', 'batch'],
        default='test',
        help='Modo de ejecución (default: test)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Archivo de textos para modo batch (uno por línea)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.90,
        help='Umbral de similitud para reconocimiento (0-1, default: 0.90)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("🧠 INFINITO V5.1 - CONTINUOUS LEARNING SYSTEM")
    print("="*70)
    print("   Sistema de aprendizaje sin detención")
    print("   Reconoce patrones causales automáticamente")
    print("="*70)
    
    # Ejecutar modo seleccionado
    if args.mode == 'test':
        run_quick_test()
    
    elif args.mode == 'interactive':
        run_interactive_mode()
    
    elif args.mode == 'batch':
        if not args.file:
            print("❌ Error: Modo batch requiere --file")
            return
        run_batch_mode(args.file)


if __name__ == "__main__":
    main()
