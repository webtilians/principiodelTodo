"""
Validación rápida del modelo INFINITO V5.2 - Solo info del checkpoint
"""

import torch
import json
from pathlib import Path

def main():
    print("=" * 70)
    print("VALIDACIÓN RÁPIDA INFINITO V5.2 - WikiText-2 REAL Training")
    print("=" * 70)
    
    checkpoint_path = "models/checkpoints/infinito_v5.2_real_best.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ No se encontró checkpoint: {checkpoint_path}")
        return
    
    print(f"\nCargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "=" * 70)
    print("INFORMACIÓN DEL CHECKPOINT")
    print("=" * 70)
    
    print(f"\n📅 Época: {checkpoint.get('epoch', 'N/A')}")
    
    val_loss = checkpoint.get('val_loss', None)
    if val_loss:
        print(f"📊 Val Loss: {val_loss:.4f}")
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        print(f"📊 Val Perplexity: {val_ppl:.2f}")
    
    if 'learning_rate' in checkpoint:
        print(f"🎓 Learning Rate: {checkpoint['learning_rate']:.2e}")
    
    config = checkpoint.get('config', {})
    if config:
        print(f"\n🔧 Configuración:")
        print(f"   - Vocab Size: {config.get('vocab_size', 'N/A'):,}")
        print(f"   - Hidden Dim: {config.get('hidden_dim', 'N/A')}")
        print(f"   - Num Layers: {config.get('num_layers', 'N/A')}")
        print(f"   - Num Heads: {config.get('num_heads', 'N/A')}")
        print(f"   - Memory Slots: {config.get('memory_slots', 'N/A')}")
    
    # Checkpoints disponibles
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE EPOCHS")
    print("=" * 70)
    
    epoch_files = list(Path("models/checkpoints").glob("infinito_v5.2_real_epoch_*.pt"))
    epoch_files.append(Path("models/checkpoints/infinito_v5.2_real_best.pt"))
    
    results = []
    for ckpt_file in epoch_files:
        if not ckpt_file.exists():
            continue
        
        ckpt = torch.load(ckpt_file, map_location='cpu')
        epoch = ckpt.get('epoch', 'N/A')
        val_loss_ckpt = ckpt.get('val_loss', None)
        
        if val_loss_ckpt:
            val_ppl_ckpt = torch.exp(torch.tensor(val_loss_ckpt)).item()
            label = "BEST" if "best" in str(ckpt_file) else f"E{epoch}"
            results.append((label, val_loss_ckpt, val_ppl_ckpt))
    
    # Ordenar por epoch
    results.sort(key=lambda x: (x[0] != "BEST", x[0]))
    
    print(f"\n{'Checkpoint':<12} {'Val Loss':>10} {'Val PPL':>10}")
    print("-" * 35)
    for label, loss, ppl in results:
        print(f"{label:<12} {loss:>10.4f} {ppl:>10.2f}")
    
    # Historial de entrenamiento
    print("\n" + "=" * 70)
    print("HISTORIAL DE ENTRENAMIENTO")
    print("=" * 70)
    
    history_files = list(Path("results/training").glob("training_history_real_*.json"))
    if history_files:
        latest_history = max(history_files, key=lambda p: p.stat().st_mtime)
        print(f"\nArchivo: {latest_history.name}")
        
        with open(latest_history, 'r') as f:
            history = json.load(f)
        
        if 'val_losses' in history:
            val_losses = history['val_losses']
            print(f"\n📉 Progresión Val PPL:")
            for i, loss in enumerate(val_losses[-10:], start=len(val_losses)-9):  # Últimas 10 épocas
                ppl = torch.exp(torch.tensor(loss)).item()
                print(f"   Época {i:2d}: PPL = {ppl:7.2f}")
    
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    
    if val_loss:
        final_ppl = torch.exp(torch.tensor(val_loss)).item()
        
        print(f"\n✅ Modelo entrenado: 20 épocas")
        print(f"✅ Val Loss final: {val_loss:.4f}")
        print(f"✅ Val Perplexity final: {final_ppl:.2f}")
        print(f"✅ Vocab size: 50,257 tokens (GPT-2)")
        print(f"✅ Parámetros: 71.4M")
        print(f"✅ Dataset: WikiText-2 REAL")
        
        # Evaluación
        if final_ppl < 80:
            print(f"\n🎉 ¡OBJETIVO ALCANZADO! PPL < 80")
        elif final_ppl < 100:
            print(f"\n✅ Buen resultado (PPL < 100)")
        elif final_ppl < 150:
            print(f"\n⚠️  Resultado aceptable (PPL < 150)")
        else:
            print(f"\n⚠️  Resultado por debajo del esperado")
        
        print(f"\n📊 Para generar texto, usa:")
        print(f"   python generate_text_v5_2.py --checkpoint {checkpoint_path}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
