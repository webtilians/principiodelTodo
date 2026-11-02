#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación Científica Simplificada - Solo Arquitecturas
======================================================

Script simplificado que ejecuta directamente las pruebas de arquitecturas
sin depender de scripts externos con problemas de codificación.
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

def test_architecture(arch_name, epochs=3, batch_size=8, timeout_minutes=60):
    """
    Prueba una arquitectura específica
    """
    print(f"\n[TEST] Probando arquitectura: {arch_name}")
    
    command = [
        "python", "train_v5_2_wikitext_real.py",
        "--model-size", arch_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size)
    ]
    
    print(f"[CMD] {' '.join(command)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
            cwd=os.getcwd()
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[OK] {arch_name} completado en {duration:.2f} segundos")
            return {
                'success': True,
                'duration_seconds': duration,
                'architecture': arch_name,
                'stdout': result.stdout[-500:],  # Últimas 500 chars
                'stderr': result.stderr[-500:] if result.stderr else ""
            }
        else:
            print(f"[ERROR] {arch_name} fallo con codigo {result.returncode}")
            return {
                'success': False,
                'duration_seconds': duration,
                'architecture': arch_name,
                'error_code': result.returncode,
                'stdout': result.stdout[-500:],
                'stderr': result.stderr[-500:] if result.stderr else ""
            }
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {arch_name} timeout despues de {timeout_minutes} minutos")
        return {
            'success': False,
            'duration_seconds': timeout_minutes * 60,
            'architecture': arch_name,
            'error': 'timeout'
        }
    except Exception as e:
        print(f"[EXCEPTION] {arch_name} excepcion: {str(e)}")
        return {
            'success': False,
            'duration_seconds': 0,
            'architecture': arch_name,
            'error': str(e)
        }

def compare_architectures():
    """
    Compara las arquitecturas optimizadas
    """
    print("="*60)
    print("[START] COMPARACION DE ARQUITECTURAS OPTIMIZADAS")
    print("="*60)
    
    # Arquitecturas a probar
    architectures = ['tiny_iit', 'micro_iit', 'small_iit']
    
    results = {}
    total_start_time = time.time()
    
    for arch in architectures:
        result = test_architecture(arch, epochs=2, batch_size=4)  # Prueba rápida
        results[arch] = result
        
        # Pequeña pausa entre pruebas
        time.sleep(2)
    
    total_duration = time.time() - total_start_time
    
    # Generar reporte
    print("\n" + "="*60)
    print("[REPORT] RESUMEN DE RESULTADOS")
    print("="*60)
    
    successful_archs = []
    failed_archs = []
    
    for arch, result in results.items():
        if result['success']:
            successful_archs.append(arch)
            print(f"[OK] {arch}: {result['duration_seconds']:.1f}s")
        else:
            failed_archs.append(arch)
            print(f"[FAIL] {arch}: {result.get('error', 'error desconocido')}")
    
    # Estadísticas finales
    success_rate = (len(successful_archs) / len(architectures)) * 100
    
    print(f"\n[STATS] Arquitecturas exitosas: {len(successful_archs)}/{len(architectures)} ({success_rate:.1f}%)")
    print(f"[STATS] Tiempo total: {total_duration:.1f} segundos ({total_duration/60:.2f} minutos)")
    
    if successful_archs:
        print(f"[SUCCESS] Arquitecturas que funcionan: {', '.join(successful_archs)}")
    
    if failed_archs:
        print(f"[FAILED] Arquitecturas con problemas: {', '.join(failed_archs)}")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"architecture_comparison_{timestamp}.json"
    
    final_report = {
        'timestamp': timestamp,
        'total_duration_seconds': total_duration,
        'success_rate_percent': success_rate,
        'successful_architectures': successful_archs,
        'failed_architectures': failed_archs,
        'detailed_results': results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVE] Resultados guardados en: {results_file}")
    
    return final_report

def test_single_architecture(arch_name):
    """
    Prueba una sola arquitectura con más detalle
    """
    print("="*60)
    print(f"[TEST] PRUEBA INDIVIDUAL: {arch_name}")
    print("="*60)
    
    result = test_architecture(arch_name, epochs=5, batch_size=8, timeout_minutes=90)
    
    print("\n" + "="*60)
    print("[RESULT] RESULTADO DETALLADO")
    print("="*60)
    
    for key, value in result.items():
        if key in ['stdout', 'stderr']:
            if value.strip():
                print(f"[{key.upper()}] {value.strip()}")
        else:
            print(f"[{key.upper()}] {value}")
    
    # Guardar resultado individual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"single_test_{arch_name}_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVE] Resultado guardado en: {result_file}")
    
    return result

def quick_model_test():
    """
    Prueba rápida para verificar que el entrenamiento básico funciona
    """
    print("="*60)
    print("[QUICK] PRUEBA RAPIDA DE MODELO")
    print("="*60)
    
    result = test_architecture('tiny_iit', epochs=1, batch_size=2, timeout_minutes=30)
    
    if result['success']:
        print("[QUICK] Modelo funciona correctamente!")
        print("[NEXT] Puedes proceder con pruebas completas")
    else:
        print("[QUICK] Hay problemas con el modelo basico")
        print("[DEBUG] Revisar configuracion antes de continuar")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Validacion Simplificada de Arquitecturas')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['compare', 'single', 'quick'],
                       help='Modo de operacion (compare: todas, single: una, quick: rapida)')
    parser.add_argument('--architecture', type=str, default='tiny_iit',
                       choices=['tiny_iit', 'micro_iit', 'small_iit', 'large_iit'],
                       help='Arquitectura especifica para modo single')
    
    args = parser.parse_args()
    
    print("[START] Iniciando validacion simplificada...")
    print(f"[MODE] Modo: {args.mode}")
    
    if args.mode == 'compare':
        result = compare_architectures()
    elif args.mode == 'single':
        result = test_single_architecture(args.architecture)
    elif args.mode == 'quick':
        result = quick_model_test()
    else:
        print("[ERROR] Modo no reconocido")
        sys.exit(1)
    
    # Código de salida basado en éxito
    if isinstance(result, dict):
        if result.get('success', False):
            print("\n[EXIT] Saliendo con codigo 0 (exito)")
            sys.exit(0)
        else:
            print("\n[EXIT] Saliendo con codigo 1 (fallo)")
            sys.exit(1)
    else:
        # Para compare mode, revisar success rate
        success_rate = result.get('success_rate_percent', 0)
        if success_rate >= 50:
            print(f"\n[EXIT] Saliendo con codigo 0 (exito: {success_rate:.1f}%)")
            sys.exit(0)
        else:
            print(f"\n[EXIT] Saliendo con codigo 1 (fallo: {success_rate:.1f}%)")
            sys.exit(1)

if __name__ == "__main__":
    main()