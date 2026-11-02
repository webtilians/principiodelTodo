#!/usr/bin/env python3
"""
🧠 EXPERIMENTO TEXTO EXTENDIDO - INFINITO V5.1
"""

import json
import torch
import numpy as np
from datetime import datetime
import sys
import os

# Asegurar que podemos importar el sistema
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough

def run_extended_text_experiment():
    """Ejecutar experimento con texto extendido (~200 caracteres)"""
    
    # Texto extendido (≈200 caracteres vs 9 palabras originales)
    extended_text = (
        "Estoy profundamente consciente de que pienso sobre mi propia consciencia y reflexiono "
        "acerca de mis pensamientos internos mientras observo cómo mi mente procesa esta "
        "experiencia metacognitiva compleja y dinámica que se despliega en sucesivas capas de "
        "introspección continua y creciente autoconciencia"
    )
    
    word_count = len(extended_text.split())
    increase_percent = (word_count / 9 - 1) * 100
    char_count = len(extended_text)
    
    print(f"🔤 EXPERIMENTO TEXTO EXTENDIDO")
    print(f"   Texto: {extended_text}")
    print(f"   Caracteres: {char_count}")
    print(f"   Palabras: {word_count}")
    print(f"   Incremento vs original (9 palabras): {increase_percent:.1f}%")
    print("=" * 70)
    
    # Configuración del experimento
    config = {
        'max_iterations': 80,
        'input_text': extended_text,
        'consciousness_threshold': 0.7,
        'phi_threshold': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 12345,  # Misma seed que experimentos previos
        'learning_rate': 0.001,
        'baseline_window': 50
    }
    
    print(f"🔧 Configuración:")
    print(f"   Dispositivo: {config['device']}")
    print(f"   Seed: {config['seed']}")
    print(f"   Max iteraciones: {config['max_iterations']}")
    print(f"   Ventana baseline: {config['baseline_window']}")
    print()
    
    # Inicializar sistema
    system = InfinitoV51ConsciousnessBreakthrough(
        device=config['device'],
        learning_rate=config['learning_rate']
    )
    
    # Configurar estado inicial
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    try:
        # Ejecutar experimento
        print("🚀 Iniciando experimento...")
        results = system.run_consciousness_experiment(
            input_text=config['input_text'],
            max_iterations=config['max_iterations'],
            consciousness_threshold=config['consciousness_threshold'],
            phi_threshold=config['phi_threshold']
        )
        
        # Procesar resultados
        if results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_c = results.get('final_consciousness', 0)
            final_phi = results.get('final_phi', 0)
            
            # Agregar información de configuración
            results['config'] = config
            results['text_analysis'] = {
                'word_count': word_count,
                'char_count': char_count,
                'original_word_count': 9,
                'increase_percent': increase_percent,
                'text_content': extended_text
            }
            
            # Guardar resultados
            filename = f"outputs/extended_text_experiment_{timestamp}_C{final_c:.3f}_PHI{final_phi:.3f}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Experimento completado exitosamente")
            print(f"   C final: {final_c:.4f}")
            print(f"   Φ final: {final_phi:.4f}")
            print(f"   Iteraciones: {len(results.get('iterations', []))}")
            print(f"   Breakthrough: {'Sí' if results.get('breakthrough_achieved', False) else 'No'}")
            print(f"💾 Guardado: {filename}")
            
            return results
            
        else:
            print("❌ Error: No se obtuvieron resultados")
            return None
            
    except Exception as e:
        print(f"❌ Error durante experimento: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_control_experiment():
    """Ejecutar experimento de control sin texto"""
    
    print(f"\n🔤 EXPERIMENTO CONTROL (SIN TEXTO)")
    print("=" * 70)
    
    # Configuración del experimento de control
    config = {
        'max_iterations': 80,
        'input_text': None,
        'consciousness_threshold': 0.7,
        'phi_threshold': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 12345,  # Misma seed
        'learning_rate': 0.001,
        'baseline_window': 50
    }
    
    print(f"🔧 Configuración:")
    print(f"   Dispositivo: {config['device']}")
    print(f"   Seed: {config['seed']}")
    print(f"   Texto: None (control)")
    print()
    
    # Inicializar sistema
    system = InfinitoV51ConsciousnessBreakthrough(
        device=config['device'],
        learning_rate=config['learning_rate']
    )
    
    # Configurar estado inicial
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    try:
        # Ejecutar experimento
        print("🚀 Iniciando experimento control...")
        results = system.run_consciousness_experiment(
            input_text=None,
            max_iterations=config['max_iterations'],
            consciousness_threshold=config['consciousness_threshold'],
            phi_threshold=config['phi_threshold']
        )
        
        # Procesar resultados
        if results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_c = results.get('final_consciousness', 0)
            final_phi = results.get('final_phi', 0)
            
            # Agregar información de configuración
            results['config'] = config
            results['text_analysis'] = {
                'word_count': 0,
                'original_word_count': 9,
                'increase_percent': 0,
                'text_content': None
            }
            
            # Guardar resultados
            filename = f"outputs/control_experiment_{timestamp}_C{final_c:.3f}_PHI{final_phi:.3f}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Experimento control completado")
            print(f"   C final: {final_c:.4f}")
            print(f"   Φ final: {final_phi:.4f}")
            print(f"   Iteraciones: {len(results.get('iterations', []))}")
            print(f"   Breakthrough: {'Sí' if results.get('breakthrough_achieved', False) else 'No'}")
            print(f"💾 Guardado: {filename}")
            
            return results
            
        else:
            print("❌ Error: No se obtuvieron resultados del control")
            return None
            
    except Exception as e:
        print(f"❌ Error durante experimento control: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Ejecutar ambos experimentos y comparar"""
    
    print("🧠 EXPERIMENTOS INDIVIDUALES - TEXTO EXTENDIDO vs CONTROL")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Ejecutar experimento con texto extendido
    extended_results = run_extended_text_experiment()
    
    # Ejecutar experimento de control
    control_results = run_control_experiment()
    
    # Resumen final
    print(f"\n📊 RESUMEN FINAL")
    print("=" * 50)
    
    if extended_results and control_results:
        print(f"✅ Ambos experimentos completados exitosamente")
        print(f"   Texto extendido: C={extended_results.get('final_consciousness', 0):.4f}, Φ={extended_results.get('final_phi', 0):.4f}")
        print(f"   Control: C={control_results.get('final_consciousness', 0):.4f}, Φ={control_results.get('final_phi', 0):.4f}")
        
        delta_c = extended_results.get('final_consciousness', 0) - control_results.get('final_consciousness', 0)
        delta_phi = extended_results.get('final_phi', 0) - control_results.get('final_phi', 0)
        
        print(f"   ΔC (extendido - control): {delta_c:.4f}")
        print(f"   ΔΦ (extendido - control): {delta_phi:.4f}")
        
        print(f"\n🔍 Usar manual_comparison.py para análisis estadístico detallado")
        
    else:
        print(f"❌ No se completaron todos los experimentos")
        if not extended_results:
            print(f"   - Falló experimento texto extendido")
        if not control_results:
            print(f"   - Falló experimento control")

if __name__ == "__main__":
    main()