#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 VISOR INTERACTIVO DE RESULTADOS
================================

Herramienta para examinar cualquier archivo de resultados de manera detallada.
"""

import os
import json
import sys

def examine_file_interactive(file_path):
    """Examina un archivo de manera interactiva."""
    
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return
    
    print(f"🔍 EXAMINANDO: {file_path}")
    print("=" * 70)
    
    # Información del archivo
    file_size = os.path.getsize(file_path)
    mod_time = os.path.getmtime(file_path)
    
    print(f"📁 Información del archivo:")
    print(f"   Tamaño: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    print(f"   Modificado: {datetime.fromtimestamp(mod_time)}")
    print()
    
    if file_path.endswith('.json'):
        examine_json_file(file_path)
    elif file_path.endswith('.pt'):
        examine_model_file(file_path)
    else:
        examine_text_file(file_path)

def examine_json_file(file_path):
    """Examina un archivo JSON en detalle."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("📋 ESTRUCTURA JSON:")
        print_json_structure(data, indent=0, max_depth=3)
        print()
        
        # Detectar tipo de archivo
        if 'val_perplexity' in data:
            print("🔍 TIPO DETECTADO: Historial de Entrenamiento")
            examine_training_history_detailed(data)
        elif any('generated' in str(v) for v in data.values() if isinstance(v, dict)):
            print("🔍 TIPO DETECTADO: Comparación de Modelos")
            examine_model_comparison_detailed(data)
        else:
            print("🔍 TIPO DETECTADO: JSON Genérico")
            examine_generic_json(data)
    
    except Exception as e:
        print(f"❌ Error leyendo JSON: {e}")

def print_json_structure(obj, indent=0, max_depth=3, current_depth=0):
    """Imprime la estructura de un objeto JSON."""
    
    if current_depth > max_depth:
        print("  " * indent + "... (contenido truncado)")
        return
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            print("  " * indent + f"📂 {key}:", end="")
            
            if isinstance(value, dict):
                print(f" dict({len(value)} claves)")
                if current_depth < max_depth:
                    print_json_structure(value, indent + 1, max_depth, current_depth + 1)
            elif isinstance(value, list):
                print(f" list({len(value)} elementos)")
                if value and current_depth < max_depth:
                    print("  " * (indent + 1) + f"Ejemplo: {str(value[0])[:50]}...")
            else:
                print(f" {type(value).__name__}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
    
    elif isinstance(obj, list):
        print("  " * indent + f"📝 Lista con {len(obj)} elementos")
        if obj and current_depth < max_depth:
            print("  " * indent + f"Primer elemento:")
            print_json_structure(obj[0], indent + 1, max_depth, current_depth + 1)

def examine_training_history_detailed(data):
    """Examina un historial de entrenamiento en detalle extremo."""
    
    print("\n📈 ANÁLISIS DETALLADO DEL ENTRENAMIENTO:")
    print("=" * 50)
    
    # Métricas básicas
    if 'val_perplexity' in data:
        val_ppls = data['val_perplexity']
        train_ppls = data.get('train_perplexity', [])
        val_losses = data.get('val_loss', [])
        train_losses = data.get('train_loss', [])
        
        print("📊 ESTADÍSTICAS GENERALES:")
        print(f"   Total de épocas: {len(val_ppls)}")
        print(f"   Perplexity validación:")
        print(f"      Inicial: {val_ppls[0]:.4f}")
        print(f"      Final: {val_ppls[-1]:.4f}")
        print(f"      Mínima: {min(val_ppls):.4f} (época {val_ppls.index(min(val_ppls)) + 1})")
        print(f"      Máxima: {max(val_ppls):.4f} (época {val_ppls.index(max(val_ppls)) + 1})")
        print()
        
        # Análisis de tendencias
        print("📈 ANÁLISIS DE TENDENCIAS:")
        
        # Calcular derivadas (cambios época a época)
        if len(val_ppls) > 1:
            changes = [val_ppls[i] - val_ppls[i-1] for i in range(1, len(val_ppls))]
            
            improvements = [c for c in changes if c < 0]
            degradations = [c for c in changes if c > 0]
            
            print(f"   Épocas con mejora: {len(improvements)}")
            print(f"   Épocas con empeoramiento: {len(degradations)}")
            print(f"   Cambio promedio por época: {sum(changes)/len(changes):+.4f}")
            
            if improvements:
                print(f"   Mayor mejora: {min(improvements):+.4f}")
            if degradations:
                print(f"   Mayor empeoramiento: {max(degradations):+.4f}")
        print()
        
        # Tabla detallada época por época
        print("📋 TABLA DETALLADA:")
        print("   Época |   Val PPL |  Train PPL |   Val Loss |  Train Loss | Cambio | Estado")
        print("   ------|-----------|------------|------------|-------------|--------|--------")
        
        for i in range(len(val_ppls)):
            epoch = i + 1
            val_ppl = val_ppls[i]
            train_ppl = train_ppls[i] if i < len(train_ppls) else None
            val_loss = val_losses[i] if i < len(val_losses) else None
            train_loss = train_losses[i] if i < len(train_losses) else None
            
            # Calcular cambio
            if i > 0:
                change = val_ppl - val_ppls[i-1]
                if change < -5:
                    status = "🚀 Excelente"
                elif change < -1:
                    status = "✅ Buena"
                elif change < 0:
                    status = "📉 Mejora"
                elif change < 1:
                    status = "➡️ Estable"
                elif change < 5:
                    status = "⚠️ Empeora"
                else:
                    status = "🚨 Malo"
            else:
                change = 0
                status = "🎬 Inicio"
            
            print(f"   {epoch:5d} | {val_ppl:9.2f} | {train_ppl or 'N/A':>10} | "
                  f"{val_loss or 'N/A':>10} | {train_loss or 'N/A':>11} | {change:+6.2f} | {status}")
        print()
    
    # Métricas IIT si están disponibles
    if 'train_phi' in data and data['train_phi']:
        phi_values = data['train_phi']
        loss_phi_values = data.get('train_loss_phi', [])
        
        print("🧠 ANÁLISIS IIT DETALLADO:")
        print(f"   PHI Integration:")
        print(f"      Inicial: {phi_values[0]:.6f}")
        print(f"      Final: {phi_values[-1]:.6f}")
        print(f"      Evolución: {phi_values[-1] - phi_values[0]:+.6f}")
        print(f"      Promedio: {sum(phi_values)/len(phi_values):.6f}")
        
        if loss_phi_values:
            print(f"   PHI Loss:")
            print(f"      Inicial: {loss_phi_values[0]:.6f}")
            print(f"      Final: {loss_phi_values[-1]:.6f}")
            print(f"      Reducción: {((loss_phi_values[0] - loss_phi_values[-1])/loss_phi_values[0]*100):.2f}%")
        print()
    
    # Learning rate evolution
    if 'learning_rate' in data and data['learning_rate']:
        lr_values = data['learning_rate']
        print("📚 EVOLUCIÓN LEARNING RATE:")
        print(f"   Inicial: {lr_values[0]:.2e}")
        print(f"   Final: {lr_values[-1]:.2e}")
        
        # Detectar cambios de LR
        lr_changes = []
        for i in range(1, len(lr_values)):
            if lr_values[i] != lr_values[i-1]:
                lr_changes.append((i+1, lr_values[i-1], lr_values[i]))
        
        if lr_changes:
            print("   Cambios detectados:")
            for epoch, old_lr, new_lr in lr_changes:
                factor = new_lr / old_lr
                print(f"      Época {epoch}: {old_lr:.2e} → {new_lr:.2e} (factor: {factor:.2f})")
        else:
            print("   Sin cambios (LR constante)")
        print()

def examine_model_comparison_detailed(data):
    """Examina una comparación de modelos en detalle."""
    
    print("\n🏆 ANÁLISIS DETALLADO DE COMPARACIÓN:")
    print("=" * 50)
    
    models_data = {}
    
    for model_name, model_info in data.items():
        if isinstance(model_info, dict) and 'avg_perplexity' in model_info:
            models_data[model_name] = model_info
    
    print(f"📊 MODELOS ANALIZADOS: {len(models_data)}")
    print()
    
    for model_name, model_info in models_data.items():
        print(f"🤖 {model_name.upper()}:")
        print(f"   Perplexity promedio: {model_info['avg_perplexity']:.4f}")
        
        if 'generations' in model_info:
            generations = model_info['generations']
            print(f"   Generaciones: {len(generations)}")
            
            # Estadísticas de perplexity
            perplexities = [g.get('perplexity', 0) for g in generations]
            if perplexities:
                print(f"   PPL rango: {min(perplexities):.2f} - {max(perplexities):.2f}")
                print(f"   PPL mediana: {sorted(perplexities)[len(perplexities)//2]:.2f}")
            
            print("   📝 Todas las generaciones:")
            for i, gen in enumerate(generations, 1):
                prompt = gen.get('prompt', 'N/A')
                generated = gen.get('generated', 'N/A')
                ppl = gen.get('perplexity', 0)
                
                # Truncar texto para legibilidad
                prompt_short = prompt[:40] + "..." if len(prompt) > 40 else prompt
                generated_short = generated[:60] + "..." if len(generated) > 60 else generated
                
                quality = "🏆" if ppl < 50 else "👍" if ppl < 200 else "⚠️" if ppl < 1000 else "💥"
                
                print(f"      {i}. {quality} \"{prompt_short}\"")
                print(f"         → \"{generated_short}\" (PPL: {ppl:.2f})")
        
        print("-" * 50)
    
    # Ranking y comparaciones
    print("\n🏆 RANKING DETALLADO:")
    sorted_models = sorted(models_data.items(), key=lambda x: x[1]['avg_perplexity'])
    
    for i, (model_name, model_info) in enumerate(sorted_models, 1):
        score = model_info['avg_perplexity']
        medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
        
        # Calcular factor respecto al mejor
        best_score = sorted_models[0][1]['avg_perplexity']
        factor = score / best_score if best_score > 0 else float('inf')
        
        print(f"   {medal} {model_name}: {score:.4f} PPL (factor: {factor:.2f}x)")

def examine_model_file(file_path):
    """Examina un archivo de modelo (.pt)."""
    
    try:
        import torch
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        print("💾 ANÁLISIS DETALLADO DEL MODELO:")
        print("=" * 50)
        
        # Información básica
        print("📊 INFORMACIÓN GENERAL:")
        for key in ['epoch', 'val_loss', 'val_ppl']:
            if key in checkpoint:
                print(f"   {key}: {checkpoint[key]}")
        print()
        
        # Configuración
        if 'config' in checkpoint:
            print("⚙️ CONFIGURACIÓN COMPLETA:")
            config = checkpoint['config']
            for key, value in config.items():
                print(f"   {key}: {value}")
            print()
        
        # Análisis del state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            print("🏗️ ARQUITECTURA DEL MODELO:")
            print(f"   Total de parámetros: {sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel')):,}")
            print()
            
            # Agrupar por tipos de capas
            layer_types = {}
            for name, tensor in state_dict.items():
                layer_type = name.split('.')[0] if '.' in name else name
                if layer_type not in layer_types:
                    layer_types[layer_type] = []
                layer_types[layer_type].append((name, tensor.shape, tensor.numel()))
            
            print("   📋 CAPAS POR TIPO:")
            for layer_type, layers in layer_types.items():
                total_params = sum(params for _, _, params in layers)
                print(f"      {layer_type}: {len(layers)} capas, {total_params:,} parámetros")
                
                # Mostrar algunas capas representativas
                for name, shape, params in layers[:3]:
                    print(f"         {name}: {list(shape)} ({params:,} params)")
                if len(layers) > 3:
                    print(f"         ... y {len(layers) - 3} más")
                print()
        
        # Historial si está disponible
        if 'history' in checkpoint:
            print("📈 HISTORIAL EMBEBIDO:")
            history = checkpoint['history']
            if 'val_perplexity' in history:
                val_ppls = history['val_perplexity']
                print(f"   Épocas entrenadas: {len(val_ppls)}")
                print(f"   Progresión PPL: {val_ppls[0]:.2f} → {val_ppls[-1]:.2f}")
            print()
    
    except Exception as e:
        print(f"❌ Error analizando modelo: {e}")

def examine_text_file(file_path):
    """Examina un archivo de texto."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("📄 ANÁLISIS DE ARCHIVO DE TEXTO:")
        print("=" * 50)
        
        lines = content.split('\n')
        words = content.split()
        
        print(f"   Líneas: {len(lines)}")
        print(f"   Palabras: {len(words)}")
        print(f"   Caracteres: {len(content)}")
        print()
        
        # Mostrar primeras líneas
        print("📖 PRIMERAS 10 LÍNEAS:")
        for i, line in enumerate(lines[:10], 1):
            print(f"   {i:2d}: {line[:100]}{'...' if len(line) > 100 else ''}")
        
        if len(lines) > 10:
            print(f"   ... y {len(lines) - 10} líneas más")
    
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")

def main():
    """Función principal."""
    
    from datetime import datetime
    
    if len(sys.argv) != 2:
        print("❌ Uso: python examine_file.py <ruta_archivo>")
        print("\nEjemplos:")
        print("   python examine_file.py models/checkpoints/infinito_v5.2_real_best.pt")
        print("   python examine_file.py results/training/training_history_real_20251115_170102.json")
        print("   python examine_file.py results/model_comparison_20251115_180340.json")
        return
    
    file_path = sys.argv[1]
    examine_file_interactive(file_path)

if __name__ == '__main__':
    main()