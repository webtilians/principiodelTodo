"""
Test de generación de texto con modelo INFINITO V5.2 entrenado
Carga en GPU y genera ejemplos
"""

import torch
import sys
from pathlib import Path
from transformers import GPT2Tokenizer

# Import directo del modelo
sys.path.append(str(Path(__file__).parent))
from src.infinito_v5_2_refactored import InfinitoV52Refactored

def load_model(checkpoint_path, device='cuda'):
    """Carga modelo desde checkpoint"""
    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extraer configuración
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', 50257)
    
    print(f"\n=== Creando Modelo ===")
    print(f"Vocab Size: {vocab_size:,}")
    print(f"Device: {device}")
    
    # Crear modelo con parámetros correctos de V5.2
    model = InfinitoV52Refactored(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        memory_slots=256,
        use_improved_memory=True,
        use_stochastic_exploration=True
    ).to(device)
    
    # Cargar pesos (strict=False para compatibilidad)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"Total parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Modelo cargado exitosamente en {device.upper()}")
    
    return model

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8, device='cuda'):
    """Genera texto desde un prompt con temperature sampling"""
    model.eval()
    
    # Tokenizar prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()
    
    print(f"\nGenerando (max {max_length} tokens, temp={temperature})...")
    
    with torch.no_grad():
        for i in range(max_length):
            # Forward pass
            logits, _ = model(generated)
            
            # Próximo token con temperature sampling
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop si EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decodificar
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

def main():
    print("=" * 70)
    print("TEST DE GENERACIÓN - INFINITO V5.2 (WikiText-2 REAL)")
    print("=" * 70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDispositivo: {device.upper()}")
    
    if device == 'cpu':
        print("⚠️  ADVERTENCIA: Corriendo en CPU (será lento)")
        print("    Considera activar CUDA para mejor rendimiento")
    
    # Checkpoint path
    checkpoint_path = "models/checkpoints/infinito_v5.2_real_best.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ No se encontró checkpoint: {checkpoint_path}")
        return
    
    # Cargar modelo
    model = load_model(checkpoint_path, device)
    
    # Tokenizer
    print("\nCargando tokenizer GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer cargado")
    
    # === PRUEBAS DE GENERACIÓN ===
    print("\n" + "=" * 70)
    print("EJEMPLOS DE GENERACIÓN DE TEXTO")
    print("=" * 70)
    
    test_prompts = [
        ("The theory of relativity", 0.7),
        ("In a world where", 0.9),
        ("Artificial intelligence is", 0.6),
        ("Once upon a time", 0.8),
        ("The future of technology", 0.7),
    ]
    
    for i, (prompt, temp) in enumerate(test_prompts, 1):
        print(f"\n{'─' * 70}")
        print(f"EJEMPLO {i}")
        print(f"{'─' * 70}")
        print(f"📝 Prompt: '{prompt}'")
        print(f"🌡️  Temperature: {temp}")
        
        try:
            import time
            start = time.time()
            
            generated = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_length=60,
                temperature=temp,
                device=device
            )
            
            elapsed = time.time() - start
            
            print(f"\n✨ Texto generado:")
            print(f"   {generated}")
            print(f"\n⏱️  Tiempo: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETADO")
    print("=" * 70)
    
    # Info del checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    val_loss = checkpoint.get('val_loss', None)
    
    if val_loss:
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        print(f"\n📊 Estadísticas del modelo:")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val Perplexity: {val_ppl:.2f}")
        print(f"   Época: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Vocab Size: 50,257 (GPT-2)")
        print(f"   Parámetros: 71.4M")

if __name__ == "__main__":
    main()
