#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba Directa de Arquitecturas Optimizadas
==========================================

Prueba las arquitecturas optimizadas del archivo advanced_model_architectures.py
para verificar que funcionan correctamente.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from advanced_model_architectures import create_optimized_model, OPTIMIZED_MODEL_CONFIGS

def test_model_creation():
    """Prueba la creación de modelos optimizados"""
    print("="*60)
    print("[TEST] CREACION DE MODELOS OPTIMIZADOS")
    print("="*60)
    
    results = {}
    
    for config_name in OPTIMIZED_MODEL_CONFIGS.keys():
        try:
            print(f"\n[CREATE] Creando modelo: {config_name}")
            
            # Crear modelo
            model = create_optimized_model(config_name)
            
            # Contar parámetros reales
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"[PARAMS] Total: {total_params:,}")
            print(f"[PARAMS] Entrenables: {trainable_params:,}")
            
            # Prueba de forward pass
            batch_size = 2
            seq_len = 64
            vocab_size = model.config['vocab_size']
            
            # Input de prueba
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids)
            
            print(f"[OUTPUT] Shape: {outputs.shape}")
            print(f"[OUTPUT] Expected: [{batch_size}, {seq_len}, {vocab_size}]")
            
            # Verificar que las dimensiones son correctas
            expected_shape = (batch_size, seq_len, vocab_size)
            if outputs.shape == expected_shape:
                print(f"[OK] Forward pass exitoso para {config_name}")
                results[config_name] = {
                    'success': True,
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'output_shape': list(outputs.shape)
                }
            else:
                print(f"[ERROR] Shape incorrecto para {config_name}")
                results[config_name] = {
                    'success': False,
                    'error': f"Shape incorrecto: {outputs.shape} != {expected_shape}",
                    'total_params': total_params,
                    'trainable_params': trainable_params
                }
            
        except Exception as e:
            print(f"[ERROR] Fallo al crear {config_name}: {str(e)}")
            results[config_name] = {
                'success': False,
                'error': str(e),
                'total_params': 0,
                'trainable_params': 0
            }
    
    return results

def test_generation_capability():
    """Prueba la capacidad de generación de un modelo"""
    print("\n" + "="*60)
    print("[TEST] CAPACIDAD DE GENERACION")
    print("="*60)
    
    try:
        # Usar el modelo más pequeño para la prueba
        model = create_optimized_model('ultra_efficient')
        
        # Input inicial
        input_ids = torch.randint(0, 1000, (1, 10))  # Secuencia de 10 tokens
        
        print(f"[INPUT] Input shape: {input_ids.shape}")
        
        # Generar texto
        with torch.no_grad():
            generated = model.generate_text(
                input_ids,
                max_length=20,
                temperature=1.0,
                do_sample=True
            )
        
        print(f"[OUTPUT] Generated shape: {generated.shape}")
        print(f"[OUTPUT] Tokens generados: {generated.shape[1] - input_ids.shape[1]}")
        
        return {
            'success': True,
            'input_length': input_ids.shape[1],
            'output_length': generated.shape[1],
            'tokens_generated': generated.shape[1] - input_ids.shape[1]
        }
        
    except Exception as e:
        print(f"[ERROR] Fallo en generación: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def test_memory_usage():
    """Prueba el uso de memoria de los modelos"""
    print("\n" + "="*60)
    print("[TEST] USO DE MEMORIA")
    print("="*60)
    
    memory_results = {}
    
    for config_name in ['ultra_efficient', 'balanced_performance']:
        try:
            print(f"\n[MEMORY] Probando {config_name}...")
            
            # Limpiar caché de GPU si está disponible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            else:
                initial_memory = 0
            
            # Crear modelo
            model = create_optimized_model(config_name)
            
            # Mover a GPU si está disponible
            if torch.cuda.is_available():
                model = model.cuda()
                model_memory = torch.cuda.memory_allocated() - initial_memory
                print(f"[GPU] Memoria del modelo: {model_memory / 1024**2:.2f} MB")
            else:
                model_memory = 0
                print(f"[CPU] Modelo creado en CPU")
            
            # Prueba de forward pass con batch más grande
            batch_size = 4
            seq_len = 128
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids)
            
            if torch.cuda.is_available():
                total_memory = torch.cuda.memory_allocated() - initial_memory
                print(f"[GPU] Memoria total con forward: {total_memory / 1024**2:.2f} MB")
            else:
                total_memory = 0
            
            memory_results[config_name] = {
                'success': True,
                'model_memory_mb': model_memory / 1024**2 if model_memory else 0,
                'total_memory_mb': total_memory / 1024**2 if total_memory else 0,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
        except Exception as e:
            print(f"[ERROR] Fallo en prueba de memoria para {config_name}: {str(e)}")
            memory_results[config_name] = {
                'success': False,
                'error': str(e)
            }
    
    return memory_results

def main():
    """Función principal"""
    print("[START] Iniciando pruebas de arquitecturas optimizadas...")
    
    # Información del sistema
    print(f"[SYSTEM] Python: {sys.version}")
    print(f"[SYSTEM] PyTorch: {torch.__version__}")
    print(f"[SYSTEM] CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")
    
    # Prueba 1: Creación de modelos
    creation_results = test_model_creation()
    
    # Prueba 2: Capacidad de generación
    generation_results = test_generation_capability()
    
    # Prueba 3: Uso de memoria
    memory_results = test_memory_usage()
    
    # Resumen final
    print("\n" + "="*60)
    print("[SUMMARY] RESUMEN DE PRUEBAS")
    print("="*60)
    
    successful_models = sum(1 for result in creation_results.values() if result['success'])
    total_models = len(creation_results)
    
    print(f"[MODELS] Modelos exitosos: {successful_models}/{total_models}")
    
    for config_name, result in creation_results.items():
        status = "OK" if result['success'] else "FAIL"
        params = result.get('total_params', 0)
        print(f"[{status}] {config_name}: {params:,} parámetros")
    
    print(f"\n[GENERATION] Generación: {'OK' if generation_results['success'] else 'FAIL'}")
    
    print(f"\n[MEMORY] Pruebas de memoria completadas")
    
    # Determinar éxito general
    overall_success = (
        successful_models >= 2 and  # Al menos 2 modelos funcionan
        generation_results['success']  # La generación funciona
    )
    
    if overall_success:
        print("\n[SUCCESS] Todas las pruebas principales exitosas!")
        print("[NEXT] Las arquitecturas optimizadas están listas para usar")
        return 0
    else:
        print("\n[WARNING] Algunas pruebas fallaron")
        print("[DEBUG] Revisar errores antes de continuar")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)