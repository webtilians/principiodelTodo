#!/usr/bin/env python3
"""
Monitor de entrenamiento baseline SIN IIT
"""
import time
import os

print("\n🔍 MONITOR DE ENTRENAMIENTO BASELINE (SIN IIT)")
print("="*70)
print("\nBuscando checkpoints y logs...")
print("\nPresiona Ctrl+C para salir")
print("="*70)

best_ppl = None
last_epoch = 0

try:
    while True:
        # Verificar checkpoint
        checkpoint_path = 'models/checkpoints/baseline_no_iit_best.pt'
        
        if os.path.exists(checkpoint_path):
            import torch
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                epoch = checkpoint.get('epoch', 0)
                val_ppl = checkpoint.get('val_ppl', 0)
                
                if epoch != last_epoch:
                    last_epoch = epoch
                    best_ppl = val_ppl
                    
                    print(f"\n{'='*70}")
                    print(f"📊 PROGRESO - Época {epoch}")
                    print(f"{'='*70}")
                    print(f"   Val PPL: {val_ppl:.2f}")
                    print(f"   Train PPL: {checkpoint.get('train_ppl', 0):.2f}")
                    
                    # Comparación
                    print(f"\n📈 COMPARACIÓN CON IIT:")
                    print(f"   Baseline (SIN IIT): {val_ppl:.2f} ← ACTUAL")
                    print(f"   Model A (CON IIT):  216.46")
                    print(f"   Model B (CON IIT):  207.15")
                    
                    if val_ppl > 220:
                        print(f"\n   ✅ IIT APORTA BENEFICIO (baseline peor)")
                    elif val_ppl < 210:
                        print(f"\n   ❌ IIT PERJUDICA (baseline mejor)")
                    else:
                        print(f"\n   ⚠️  RESULTADO INCIERTO (diferencia < 5%)")
                    
                    print(f"{'='*70}")
            except Exception as e:
                pass
        
        # Esperar 30 segundos
        time.sleep(30)
        
except KeyboardInterrupt:
    print("\n\n🛑 Monitor detenido")
    if best_ppl:
        print(f"\n📊 ÚLTIMO RESULTADO:")
        print(f"   Época: {last_epoch}")
        print(f"   Val PPL: {best_ppl:.2f}")
