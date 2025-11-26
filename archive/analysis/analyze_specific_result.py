#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 ANALIZADOR ESPECÍFICO DE RESULTADOS
====================================

Examina en detalle archivos específicos de resultados.
"""

import os
import json
import torch
import sys

def analyze_best_model():
    """Analiza el mejor modelo en detalle."""
    
    print("🏆 ANÁLISIS DEL MEJOR MODELO")
    print("=" * 50)
    
    model_path = "models/checkpoints/infinito_v5.2_real_best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("📊 INFORMACIÓN GENERAL:")
        print(f"   Época: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        print(f"   Val Perplexity: {checkpoint.get('val_ppl', 'N/A'):.2f}")
        print()
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("⚙️ CONFIGURACIÓN:")
            for key, value in config.items():
                print(f"   {key}: {value}")
            print()
        
        if 'history' in checkpoint:
            history = checkpoint['history']
            print("📈 HISTORIAL DE ENTRENAMIENTO:")
            
            if 'val_perplexity' in history:
                val_ppls = history['val_perplexity']
                print(f"   Épocas: {len(val_ppls)}")
                print(f"   PPL inicial: {val_ppls[0]:.2f}")
                print(f"   PPL final: {val_ppls[-1]:.2f}")
                print(f"   Mejor PPL: {min(val_ppls):.2f} (época {val_ppls.index(min(val_ppls)) + 1})")
                
                improvement = ((val_ppls[0] - min(val_ppls)) / val_ppls[0]) * 100
                print(f"   Mejora: {improvement:.1f}%")
                print()
            
            if 'train_phi' in history:
                phi_values = history['train_phi']
                print("🧠 MÉTRICAS IIT:")
                print(f"   PHI inicial: {phi_values[0]:.4f}")
                print(f"   PHI final: {phi_values[-1]:.4f}")
                print(f"   Evolución PHI: {phi_values[-1] - phi_values[0]:+.4f}")
                print()
    
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_latest_training():
    """Analiza el entrenamiento más reciente."""
    
    print("📊 ANÁLISIS DEL ÚLTIMO ENTRENAMIENTO")
    print("=" * 50)
    
    # Buscar el archivo más reciente
    training_dir = "results/training"
    if not os.path.exists(training_dir):
        print(f"❌ Directorio no encontrado: {training_dir}")
        return
    
    files = [f for f in os.listdir(training_dir) if f.startswith('training_history_real_')]
    if not files:
        print("❌ No se encontraron historiales de entrenamiento")
        return
    
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(training_dir, f)))
    latest_path = os.path.join(training_dir, latest_file)
    
    print(f"📁 Archivo: {latest_file}")
    
    try:
        with open(latest_path, 'r') as f:
            data = json.load(f)
        
        print("📈 PROGRESO DEL ENTRENAMIENTO:")
        
        if 'val_perplexity' in data:
            val_ppls = data['val_perplexity']
            train_ppls = data.get('train_perplexity', [])
            
            print(f"   📊 Épocas completadas: {len(val_ppls)}")
            print()
            
            print("   📋 EVOLUCIÓN PPL:")
            for i, val_ppl in enumerate(val_ppls, 1):
                train_ppl = train_ppls[i-1] if i-1 < len(train_ppls) else "N/A"
                
                # Indicador de mejora
                if i > 1:
                    change = val_ppl - val_ppls[i-2]
                    if change < -1:
                        trend = "📉 Mejora significativa"
                    elif change < 0:
                        trend = "📉 Ligera mejora"
                    elif change > 1:
                        trend = "📈 Empeoramiento"
                    else:
                        trend = "➡️ Estable"
                else:
                    trend = "🚀 Inicio"
                
                print(f"      Época {i:2d}: Val={val_ppl:7.2f} | Train={train_ppl:>7} | {trend}")
            
            print()
            print("   📊 RESUMEN:")
            print(f"      Mejor época: {val_ppls.index(min(val_ppls)) + 1}")
            print(f"      Mejor PPL: {min(val_ppls):.2f}")
            print(f"      PPL final: {val_ppls[-1]:.2f}")
            
            improvement = ((val_ppls[0] - min(val_ppls)) / val_ppls[0]) * 100
            print(f"      Mejora total: {improvement:.1f}%")
            
            # Estado final
            if len(val_ppls) >= 3:
                recent = val_ppls[-3:]
                if recent[-1] > recent[0]:
                    status = "⚠️ OVERFITTING DETECTADO"
                elif recent[-1] < recent[0]:
                    status = "✅ SIGUE MEJORANDO"
                else:
                    status = "➡️ CONVERGENCIA"
                print(f"      Estado: {status}")
        
        # Métricas IIT
        if 'train_phi' in data and data['train_phi']:
            phi_values = data['train_phi']
            print()
            print("   🧠 INTEGRACIÓN IIT:")
            print(f"      PHI inicial: {phi_values[0]:.4f}")
            print(f"      PHI final: {phi_values[-1]:.4f}")
            print(f"      Tendencia: {'📈 Creciente' if phi_values[-1] > phi_values[0] else '📉 Decreciente'}")
        
        # Learning rate
        if 'learning_rate' in data and data['learning_rate']:
            lr_values = data['learning_rate']
            print()
            print("   📚 LEARNING RATE:")
            print(f"      LR inicial: {lr_values[0]:.2e}")
            print(f"      LR final: {lr_values[-1]:.2e}")
            if lr_values[-1] < lr_values[0]:
                reduction = ((lr_values[0] - lr_values[-1]) / lr_values[0]) * 100
                print(f"      Reducción: {reduction:.1f}% (scheduler activo)")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_model_comparison():
    """Analiza la comparación de modelos más reciente."""
    
    print("🏆 ANÁLISIS DE COMPARACIÓN DE MODELOS")
    print("=" * 50)
    
    # Buscar archivo de comparación más reciente
    comparison_files = []
    for root, dirs, files in os.walk("results"):
        for file in files:
            if 'comparison' in file.lower() and file.endswith('.json'):
                comparison_files.append(os.path.join(root, file))
    
    if not comparison_files:
        print("❌ No se encontraron archivos de comparación")
        return
    
    latest_comparison = max(comparison_files, key=os.path.getmtime)
    
    try:
        with open(latest_comparison, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📁 Archivo: {os.path.basename(latest_comparison)}")
        print()
        
        # Analizar cada modelo
        model_scores = {}
        
        for model_name, model_data in data.items():
            if isinstance(model_data, dict) and 'avg_perplexity' in model_data:
                avg_ppl = model_data['avg_perplexity']
                model_scores[model_name] = avg_ppl
                
                print(f"🤖 {model_name}:")
                print(f"   📊 Perplexity promedio: {avg_ppl:.2f}")
                
                if 'generations' in model_data and model_data['generations']:
                    successful = len(model_data['generations'])
                    print(f"   ✅ Generaciones exitosas: {successful}")
                    
                    # Mostrar mejor y peor generación
                    generations = model_data['generations']
                    perplexities = [g.get('perplexity', float('inf')) for g in generations]
                    
                    if perplexities:
                        best_idx = perplexities.index(min(perplexities))
                        worst_idx = perplexities.index(max(perplexities))
                        
                        print(f"   🏆 Mejor generación (PPL {perplexities[best_idx]:.2f}):")
                        best_gen = generations[best_idx]
                        print(f"      \"{best_gen.get('prompt', '')}\" →")
                        print(f"      \"{best_gen.get('generated', '')[:100]}...\"")
                        
                        print(f"   😵 Peor generación (PPL {perplexities[worst_idx]:.2f}):")
                        worst_gen = generations[worst_idx]
                        print(f"      \"{worst_gen.get('prompt', '')}\" →")
                        print(f"      \"{worst_gen.get('generated', '')[:100]}...\"")
                
                print()
        
        # Ranking
        if model_scores:
            print("🏆 RANKING DE MODELOS:")
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
            
            for i, (model_name, score) in enumerate(sorted_models, 1):
                medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
                print(f"   {medal} {model_name}: {score:.2f} PPL")
            
            # Diferencias
            print()
            print("📊 DIFERENCIAS:")
            best_score = sorted_models[0][1]
            for model_name, score in sorted_models[1:]:
                if best_score > 0:
                    factor = score / best_score
                    print(f"   {model_name} es {factor:.1f}x peor que el mejor")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def quick_summary():
    """Resumen rápido de todos los resultados importantes."""
    
    print("⚡ RESUMEN RÁPIDO DE RESULTADOS")
    print("=" * 50)
    
    # Mejor modelo
    best_model_path = "models/checkpoints/infinito_v5.2_real_best.pt"
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        print(f"🏆 MEJOR MODELO:")
        print(f"   Época: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Val PPL: {checkpoint.get('val_ppl', 'N/A'):.2f}")
        
        if 'history' in checkpoint and 'val_perplexity' in checkpoint['history']:
            val_ppls = checkpoint['history']['val_perplexity']
            improvement = ((val_ppls[0] - min(val_ppls)) / val_ppls[0]) * 100
            print(f"   Mejora: {improvement:.1f}%")
        print()
    
    # Último entrenamiento
    training_dir = "results/training"
    if os.path.exists(training_dir):
        files = [f for f in os.listdir(training_dir) if f.startswith('training_history_real_')]
        if files:
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(training_dir, f)))
            with open(os.path.join(training_dir, latest_file), 'r') as f:
                data = json.load(f)
            
            print(f"📈 ÚLTIMO ENTRENAMIENTO:")
            print(f"   Archivo: {latest_file}")
            if 'val_perplexity' in data:
                val_ppls = data['val_perplexity']
                print(f"   Épocas: {len(val_ppls)}")
                print(f"   PPL final: {val_ppls[-1]:.2f}")
                print(f"   Mejor PPL: {min(val_ppls):.2f}")
            print()
    
    # Comparación más reciente
    comparison_files = []
    for root, dirs, files in os.walk("results"):
        for file in files:
            if 'comparison' in file.lower() and file.endswith('.json'):
                comparison_files.append(os.path.join(root, file))
    
    if comparison_files:
        latest_comparison = max(comparison_files, key=os.path.getmtime)
        with open(latest_comparison, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"🏆 COMPARACIÓN DE MODELOS:")
        model_scores = {k: v.get('avg_perplexity', float('inf')) 
                       for k, v in data.items() 
                       if isinstance(v, dict) and 'avg_perplexity' in v}
        
        if model_scores:
            best_model = min(model_scores.items(), key=lambda x: x[1])
            print(f"   Mejor: {best_model[0]} (PPL: {best_model[1]:.2f})")
        print()
    
    print("💡 COMANDOS ÚTILES:")
    print("   python analyze_specific_result.py --best        # Analizar mejor modelo")
    print("   python analyze_specific_result.py --latest      # Último entrenamiento")
    print("   python analyze_specific_result.py --comparison  # Comparación de modelos")

def main():
    """Función principal."""
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == '--best':
            analyze_best_model()
        elif arg == '--latest':
            analyze_latest_training()
        elif arg == '--comparison':
            analyze_model_comparison()
        elif arg == '--summary':
            quick_summary()
        else:
            print(f"❌ Argumento no reconocido: {arg}")
            print("Uso: python analyze_specific_result.py [--best|--latest|--comparison|--summary]")
    else:
        # Por defecto, mostrar todo
        quick_summary()
        print()
        analyze_best_model()
        print("\n" + "="*70 + "\n")
        analyze_latest_training()
        print("\n" + "="*70 + "\n")
        analyze_model_comparison()

if __name__ == '__main__':
    main()