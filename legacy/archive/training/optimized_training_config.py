#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración optimizada para entrenar INFINITO V5.2
Basada en el análisis de resultados previos
"""

import argparse
import os
import sys

def create_optimized_config():
    """Crear configuración optimizada basada en análisis de resultados."""
    
    config = {
        "model_config": {
            "d_model": 384,  # Modelo pequeño pero eficiente
            "n_layers": 3,
            "n_heads": 6,
            "d_ff": 1536,
            "max_length": 512,
            "vocab_size": 50257,
            "dropout": 0.25,  # Aumentado para combatir overfitting
        },
        
        "training_config": {
            "learning_rate": 1e-4,  # Reducido significativamente
            "batch_size": 16,
            "num_epochs": 15,  # Más épocas con LR bajo
            "warmup_steps": 500,  # Más warm-up
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
        },
        
        "iit_config": {
            "lambda_phi": 0.1,  # Reducido de 0.3
            "use_learnable_weights": True,
            "phi_update_frequency": 50,  # Más conservador
        },
        
        "optimization": {
            "early_stopping_patience": 4,  # Parar si no mejora en 4 épocas
            "lr_scheduler": "cosine",
            "min_lr": 1e-6,
        }
    }
    
    return config

def main():
    print("🔧 CONFIGURACIÓN OPTIMIZADA INFINITO V5.2")
    print("=" * 60)
    
    config = create_optimized_config()
    
    print("📋 Cambios principales vs configuración anterior:")
    print()
    print("🎯 Learning Rate:")
    print(f"   Anterior: 5e-4 → Nuevo: {config['training_config']['learning_rate']}")
    print("   Razón: Reducir oscilaciones y overfitting")
    print()
    
    print("🛡️  Dropout:")
    print(f"   Anterior: 0.15 → Nuevo: {config['model_config']['dropout']}")
    print("   Razón: Combatir overfitting más agresivamente")
    print()
    
    print("⚖️  Lambda Phi (IIT):")
    print(f"   Anterior: 0.3 → Nuevo: {config['iit_config']['lambda_phi']}")
    print("   Razón: Reducir interferencia de métricas IIT")
    print()
    
    print("⏰ Early Stopping:")
    print(f"   Nuevo: {config['optimization']['early_stopping_patience']} épocas de paciencia")
    print("   Razón: Parar automáticamente cuando empiece overfitting")
    print()
    
    print("🚀 COMANDO RECOMENDADO:")
    print("python train_v5_2_wikitext_real.py \\")
    print(f"  --model-size small_iit \\")
    print(f"  --learning-rate {config['training_config']['learning_rate']} \\")
    print(f"  --dropout {config['model_config']['dropout']} \\")
    print(f"  --lambda-phi {config['iit_config']['lambda_phi']} \\")
    print(f"  --epochs {config['training_config']['num_epochs']} \\")
    print(f"  --patience {config['optimization']['early_stopping_patience']} \\")
    print("  --output-dir results/optimized_training")
    print()
    
    print("📈 RESULTADOS ESPERADOS:")
    print("• PPL inicial: ~340")
    print("• PPL objetivo: <180 (mejor que 216.23 anterior)")
    print("• Entrenamiento estable sin degradación")
    print("• Parada automática en óptimo")

if __name__ == '__main__':
    main()