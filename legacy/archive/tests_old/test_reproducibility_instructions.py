"""
Test de Reproducibilidad del Lenguaje Causal
============================================

Reutiliza el causal_architecture_analyzer.py para testear reproducibilidad
Objetivo: Verificar si "mi perro es rojo" genera arquitecturas reproducibles
         con diferentes seeds
"""

import subprocess
import json
import numpy as np
from datetime import datetime

def run_architecture_analysis_with_seed(text, seed, iterations=50):
    """
    Ejecuta el causal_architecture_analyzer con un seed específico
    """
    # Modificamos temporalmente el seed en el código y ejecutamos
    cmd = [
        'python',
        'causal_architecture_analyzer.py',
        '--text', text,
        '--iterations', str(iterations),
        '--seed', str(seed)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        
        # El analyzer genera un JSON con los resultados
        # Buscamos el archivo JSON más reciente
        import glob
        import os
        
        list_of_files = glob.glob('causal_vocabulary_*.json')
        if not list_of_files:
            return None
            
        latest_file = max(list_of_files, key=os.path.getctime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extraer la arquitectura del texto específico
        if 'architectures' in data and text in data['architectures']:
            return data['architectures'][text]
        
        return None
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def main():
    print("=" * 70)
    print("TEST DE REPRODUCIBILIDAD - MI PERRO ES ROJO")
    print("=" * 70)
    
    text = "mi perro es rojo"
    num_seeds = 10
    iterations = 50
    
    print(f"\nTexto: '{text}'")
    print(f"Seeds: {num_seeds}")
    print(f"Iteraciones por seed: {iterations}\n")
    
    # Por ahora, voy a ejecutar el analyzer que ya existe
    # y parsear los resultados manualmente
    
    print("INSTRUCCIONES:")
    print("=" * 70)
    print("\nDebes ejecutar manualmente 10 veces:")
    print(f"  python causal_architecture_analyzer.py")
    print("\nCon el texto '{text}' pero modificando el seed cada vez.")
    print("Seeds sugeridos: 1000, 1001, 1002, ... 1009")
    print("\nLuego analiza los archivos JSON generados para comparar:")
    print("  - phi_mean")
    print("  - dominant_state")
    print("  - phi_std")
    print("\nSi la varianza de phi_mean entre seeds < 0.05:")
    print("  → El lenguaje causal ES DETERMINISTA ✅")
    print("Sino:")
    print("  → El lenguaje causal tiene componente estocástico ⚠️")

if __name__ == "__main__":
    main()
