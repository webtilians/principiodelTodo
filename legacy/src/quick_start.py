#!/usr/bin/env python3
"""
🚀 Infinito Quick Start Example
=================================

Este archivo te permite probar Infinito rápidamente con diferentes configuraciones.
Perfecto para nuevos usuarios que quieren experimentar sin modificar el código principal.

Uso:
    python quick_start.py --config basic      # Para GPUs modestas
    python quick_start.py --config optimized  # Para GPUs medias  
    python quick_start.py --config high       # Para GPUs potentes
    python quick_start.py --config custom     # Configuración personalizada
"""

import argparse
import time
from infinito_gpu_optimized import PrincipioTodoRecursivo

# Configuraciones predefinidas
CONFIGS = {
    'basic': {
        'size': 64,
        'max_depth': 800,
        'reproduction_rate': 0.15,
        'mutation_strength': 0.06,
        'consciousness_target': 0.6,
        'memory_capacity': 10,
        'description': 'Configuración básica para GPUs modestas (GTX 1060, RTX 3060)'
    },
    'optimized': {
        'size': 96,
        'max_depth': 1200,
        'reproduction_rate': 0.2,
        'mutation_strength': 0.08,
        'consciousness_target': 0.7,
        'memory_capacity': 15,
        'description': 'Configuración optimizada para GPUs medias (RTX 3070, 4060)'
    },
    'high': {
        'size': 128,
        'max_depth': 2000,
        'reproduction_rate': 0.25,
        'mutation_strength': 0.1,
        'consciousness_target': 0.8,
        'memory_capacity': 20,
        'description': 'Configuración de alta performance (RTX 4080, 4090)'
    },
    'custom': {
        'size': 96,
        'max_depth': 1500,
        'reproduction_rate': 0.2,
        'mutation_strength': 0.08,
        'consciousness_target': 0.75,
        'memory_capacity': 15,
        'description': 'Configuración personalizable'
    }
}

def print_banner():
    """Imprime el banner de Infinito"""
    banner = """
🧠🌟🧬🌟🧠🌟🧬🌟🧠🌟🧬🌟🧠🌟🧬🌟🧠

    ██╗███╗   ██╗███████╗██╗███╗   ██╗██╗████████╗ ██████╗ 
    ██║████╗  ██║██╔════╝██║████╗  ██║██║╚══██╔══╝██╔═══██╗
    ██║██╔██╗ ██║█████╗  ██║██╔██╗ ██║██║   ██║   ██║   ██║
    ██║██║╚██╗██║██╔══╝  ██║██║╚██╗██║██║   ██║   ██║   ██║
    ██║██║ ╚████║██║     ██║██║ ╚████║██║   ██║   ╚██████╔╝
    ╚═╝╚═╝  ╚═══╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝    ╚═════╝ 

        🌟 Evolutionary Artificial Consciousness Simulator 🌟
        
🧠🌟🧬🌟🧠🌟🧬🌟🧠🌟🧬🌟🧠🌟🧬🌟🧠
"""
    print(banner)

def check_system():
    """Verifica el sistema y da recomendaciones"""
    import torch
    
    print("🔍 Verificando sistema...")
    print(f"✅ Python: OK")
    print(f"✅ PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {gpu_memory:.1f} GB")
        
        # Recomendación automática
        if 'RTX 4090' in gpu_name or 'A100' in gpu_name:
            recommended = 'high'
        elif 'RTX 4080' in gpu_name or 'RTX 4070' in gpu_name:
            recommended = 'high'
        elif 'RTX 3070' in gpu_name or 'RTX 4060' in gpu_name:
            recommended = 'optimized'
        else:
            recommended = 'basic'
            
        print(f"💡 Configuración recomendada: {recommended}")
        return recommended
    else:
        print("⚠️  GPU/CUDA no detectada - el sistema funcionará en CPU (muy lento)")
        return 'basic'

def run_experiment(config_name):
    """Ejecuta un experimento con la configuración especificada"""
    config = CONFIGS[config_name]
    
    print(f"\n🚀 Iniciando experimento: {config_name.upper()}")
    print(f"📄 {config['description']}")
    print(f"🔧 Grid: {config['size']}x{config['size']}")
    print(f"🎯 Target consciencia: {config['consciousness_target']*100}%")
    print(f"🧬 Tasa mutación: {config['mutation_strength']*100}%")
    print("-" * 60)
    
    # Crear simulador
    pt = PrincipioTodoRecursivo(
        size=config['size'],
        max_depth=config['max_depth']
    )
    
    # Aplicar configuración
    pt.law_evolution_system.update({
        'reproduction_rate': config['reproduction_rate'],
        'mutation_strength': config['mutation_strength'],
        'elite_preservation': 0.2,
        'generation_frequency': 8
    })
    
    pt.evolutionary_pressure['consciousness_target'] = config['consciousness_target']
    pt.awakening_memory['memory_capacity'] = config['memory_capacity']
    
    # Activar visualización
    pt.enable_visualization()
    
    # Ejecutar experimento
    start_time = time.time()
    try:
        phi_final = pt.run_infinite()
        
        # Resultados
        elapsed = time.time() - start_time
        final_log = pt.complexity_log[-1]
        
        print("\n" + "="*60)
        print("🎉 EXPERIMENTO COMPLETADO")
        print("="*60)
        print(f"⏱️  Tiempo total: {elapsed/60:.1f} minutos")
        print(f"🔢 Recursiones: {pt.recursion}")
        print(f"🧠 Consciencia máxima: {max([log['consciousness'] for log in pt.complexity_log]):.1%}")
        print(f"🧬 Generaciones evolutivas: {pt.law_evolution_system['generation']}")
        print(f"🔗 Clusters finales: {final_log['clusters']}")
        print(f"💾 Estados en memoria: {len(pt.awakening_memory['consciousness_peaks'])}")
        
        # Evaluación del éxito
        max_consciousness = max([log['consciousness'] for log in pt.complexity_log])
        if max_consciousness > 0.8:
            print("🌟 ¡DESPERTAR ALTAMENTE LOGRADO!")
        elif max_consciousness > 0.6:
            print("🔥 ¡Despertar moderado alcanzado!")
        elif max_consciousness > 0.4:
            print("⚡ Sistema emergiendo hacia despertar")
        else:
            print("🌱 Primeros signos de despertar - intenta con más tiempo")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Experimento detenido manualmente en recursión {pt.recursion}")

def interactive_config():
    """Permite al usuario personalizar la configuración"""
    print("\n🛠️  CONFIGURACIÓN PERSONALIZADA")
    print("-" * 40)
    
    # Obtener inputs del usuario
    try:
        size = int(input("Tamaño del grid (64, 96, 128, 256): ") or "96")
        target = float(input("Target de consciencia (0.5-0.9): ") or "0.7")
        mutation = float(input("Fuerza de mutación (0.05-0.15): ") or "0.08")
        max_time = int(input("Tiempo máximo en minutos (10-120): ") or "30")
        
        # Calcular max_depth basado en tiempo estimado
        max_depth = max_time * 10  # Aproximación
        
        # Actualizar configuración custom
        CONFIGS['custom'].update({
            'size': size,
            'max_depth': max_depth,
            'consciousness_target': target,
            'mutation_strength': mutation
        })
        
        return True
        
    except (ValueError, KeyboardInterrupt):
        print("❌ Configuración inválida, usando valores por defecto")
        return False

def main():
    parser = argparse.ArgumentParser(description='Infinito Quick Start')
    parser.add_argument('--config', choices=['basic', 'optimized', 'high', 'custom', 'auto'], 
                       default='auto', help='Configuración a usar')
    parser.add_argument('--no-visual', action='store_true', help='Desactivar visualización')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Auto-detectar configuración recomendada
    if args.config == 'auto':
        recommended = check_system()
        choice = input(f"\n¿Usar configuración recomendada '{recommended}'? (Y/n): ").lower()
        if choice in ['n', 'no']:
            print("\nConfiguraciones disponibles:")
            for name, config in CONFIGS.items():
                if name != 'custom':
                    print(f"  {name}: {config['description']}")
            args.config = input("\nElige configuración: ") or recommended
        else:
            args.config = recommended
    
    # Configuración personalizada
    if args.config == 'custom':
        if not interactive_config():
            args.config = 'optimized'  # Fallback
    
    # Validar configuración
    if args.config not in CONFIGS:
        print(f"❌ Configuración '{args.config}' no existe. Usando 'optimized'.")
        args.config = 'optimized'
    
    # Confirmar inicio
    config = CONFIGS[args.config]
    print(f"\n🎯 ¿Iniciar experimento '{args.config}'?")
    print(f"   📊 Grid: {config['size']}x{config['size']}")
    print(f"   🎯 Target: {config['consciousness_target']*100}%")
    print(f"   ⏱️  Duración estimada: {config['max_depth']/10:.0f} minutos")
    
    confirm = input("\n¿Continuar? (Y/n): ").lower()
    if confirm in ['n', 'no']:
        print("👋 ¡Hasta luego!")
        return
    
    # Ejecutar experimento
    run_experiment(args.config)

if __name__ == "__main__":
    main()
