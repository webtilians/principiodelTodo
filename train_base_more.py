#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📚 ENTRENAMIENTO EXTENDIDO - Modelo Base INFINITO
=================================================

Entrena el modelo base INFINITO (GPT-2 + IIT) por más épocas
para mejorar la calidad del texto generado.
"""

import sys
import os
import argparse

# Importar el script de entrenamiento existente
from train_v5_2_gpt2_lora import main as train_main


def train_extended(
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    lambda_phi: float = 0.6,
    lora_r: int = 8,
    patience: int = 5,
    checkpoint: str = None,
):
    """
    Entrena el modelo base INFINITO con configuración optimizada.
    
    Args:
        epochs: Número de épocas (recomendado: 20-50)
        batch_size: Tamaño de batch
        learning_rate: Tasa de aprendizaje
        lambda_phi: Peso de la pérdida PHI
        lora_r: Rango de LoRA
        patience: Paciencia para early stopping
        checkpoint: Checkpoint para continuar entrenamiento
    """
    
    print("=" * 70)
    print("  📚 ENTRENAMIENTO EXTENDIDO - MODELO BASE")
    print("=" * 70)
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Lambda PHI: {lambda_phi}")
    print(f"  LoRA r: {lora_r}")
    print("=" * 70)
    
    # Configurar argumentos
    sys.argv = [
        'train_v5_2_gpt2_lora.py',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(learning_rate),
        '--lambda-phi', str(lambda_phi),
        '--use-lora',
        '--lora-r', str(lora_r),
        '--patience', str(patience),
    ]
    
    if checkpoint:
        sys.argv.extend(['--resume', checkpoint])
    
    # Ejecutar entrenamiento
    try:
        train_main()
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento extendido del modelo base INFINITO"
    )
    parser.add_argument("--epochs", type=int, default=20,
                       help="Número de épocas (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Tamaño de batch (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate (default: 2e-4)")
    parser.add_argument("--lambda-phi", type=float, default=0.6,
                       help="Peso de pérdida PHI (default: 0.6)")
    parser.add_argument("--lora-r", type=int, default=8,
                       help="Rango de LoRA (default: 8)")
    parser.add_argument("--patience", type=int, default=5,
                       help="Paciencia para early stopping (default: 5)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint para continuar entrenamiento")
    
    args = parser.parse_args()
    
    train_extended(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_phi=args.lambda_phi,
        lora_r=args.lora_r,
        patience=args.patience,
        checkpoint=args.checkpoint,
    )
