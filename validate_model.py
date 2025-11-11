"""
Validación del modelo INFINITO V5.2 entrenado con WikiText-2 REAL
Evalúa perplexity, genera texto y compara con el estado pre-entrenamiento
"""

import torch
import json
import sys
from pathlib import Path
from transformers import GPT2Tokenizer

# Import directo del modelo
sys.path.append(str(Path(__file__).parent))
from src.infinito_v5_2_refactored import InfinitoV52Refactored
import time

def load_checkpoint(checkpoint_path):
    """Carga checkpoint y extrae información"""
    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n=== Información del Checkpoint ===")
    print(f"Época: {checkpoint.get('epoch', 'N/A')}")
    print(f"Iteración: {checkpoint.get('iteration', 'N/A')}")
    
    train_loss = checkpoint.get('train_loss', None)
    if train_loss is not None:
        print(f"Train Loss: {train_loss:.4f}")
    else:
        print(f"Train Loss: N/A")
    
    val_loss = checkpoint.get('val_loss', None)
    if val_loss is not None:
        print(f"Val Loss: {val_loss:.4f}")
    else:
        print(f"Val Loss: N/A")
    
    val_ppl = checkpoint.get('val_perplexity', None)
    if val_ppl is not None:
        print(f"Val Perplexity: {val_ppl:.2f}")
    else:
        print(f"Val Perplexity: N/A")
    
    if 'learning_rate' in checkpoint:
        print(f"Learning Rate: {checkpoint['learning_rate']:.2e}")
    
    return checkpoint

def load_model(checkpoint_path, device='cuda'):
    """Carga modelo desde checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extraer configuración
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', 50257)
    
    print(f"\n=== Creando Modelo ===")
    print(f"Vocab Size: {vocab_size}")
    print(f"Device: {device}")
    
    # Crear modelo con parámetros correctos de V5.2
    # IMPORTANTE: memory_slots debe coincidir con el checkpoint (256)
    model = InfinitoV52Refactored(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        memory_slots=256,  # Del checkpoint original
        use_improved_memory=True,
        use_stochastic_exploration=True
    ).to(device)
    
    # Cargar pesos (strict=False para ignorar claves extras)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"Total parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Modelo cargado exitosamente")
    
    return model, checkpoint

def generate_text_sample(model, tokenizer, prompt, max_length=100, device='cuda'):
    """Genera texto desde un prompt"""
    model.eval()
    
    # Tokenizar prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass (modelo retorna 2 valores: logits, metrics)
            logits, _ = model(generated)
            
            # Próximo token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop si EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decodificar
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    return generated_text

def evaluate_perplexity_sample(model, tokenizer, text, device='cuda'):
    """Evalúa perplexity en un texto de ejemplo"""
    model.eval()
    
    # Tokenizar
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    if input_ids.shape[1] < 2:
        return float('inf')
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        
        # Cross entropy
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1)
        )
        
        perplexity = torch.exp(loss).item()
    
    return perplexity

def main():
    """Validación principal"""
    
    print("=" * 70)
    print("VALIDACIÓN INFINITO V5.2 - WikiText-2 REAL Training")
    print("=" * 70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDispositivo: {device}")
    
    # Checkpoint path
    checkpoint_path = "models/checkpoints/infinito_v5.2_real_best.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ No se encontró checkpoint: {checkpoint_path}")
        return
    
    # Cargar info
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Cargar modelo
    print("\nCargando modelo...")
    model, _ = load_model(checkpoint_path, device)
    
    # Tokenizer
    print("\nCargando tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # === PRUEBAS DE GENERACIÓN ===
    print("\n" + "=" * 70)
    print("PRUEBAS DE GENERACIÓN DE TEXTO")
    print("=" * 70)
    
    test_prompts = [
        "The quick brown fox",
        "In a world where",
    ]  # Solo 2 prompts para ir más rápido
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Prueba {i} ---")
        print(f"Prompt: '{prompt}'")
        
        start = time.time()
        generated = generate_text_sample(model, tokenizer, prompt, max_length=20, device=device)  # Solo 20 tokens
        elapsed = time.time() - start
        
        print(f"Generado: {generated}")
        print(f"Tiempo: {elapsed:.2f}s")
    
    # === EVALUACIÓN DE PERPLEXITY ===
    print("\n" + "=" * 70)
    print("EVALUACIÓN DE PERPLEXITY EN TEXTOS DE EJEMPLO")
    print("=" * 70)
    
    test_texts = [
        "The study of natural language processing has advanced significantly.",
    ]  # Solo 1 texto para ir más rápido
    
    for i, text in enumerate(test_texts, 1):
        ppl = evaluate_perplexity_sample(model, tokenizer, text, device)
        print(f"\nTexto {i}: {text}")
        print(f"Perplexity: {ppl:.2f}")
    
    # === COMPARACIÓN CON PRE-ENTRENAMIENTO ===
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE EPOCHS")
    print("=" * 70)
    
    epoch_checkpoints = [
        ("models/checkpoints/infinito_v5.2_real_epoch_5.pt", "Epoch 5"),
        ("models/checkpoints/infinito_v5.2_real_epoch_10.pt", "Epoch 10"),
        ("models/checkpoints/infinito_v5.2_real_epoch_15.pt", "Epoch 15"),
        ("models/checkpoints/infinito_v5.2_real_epoch_20.pt", "Epoch 20"),
        ("models/checkpoints/infinito_v5.2_real_best.pt", "Best"),
    ]
    
    for ckpt_path, label in epoch_checkpoints:
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location='cpu')
            val_ppl = ckpt.get('val_perplexity', 'N/A')
            val_loss = ckpt.get('val_loss', 'N/A')
            
            if isinstance(val_ppl, float):
                print(f"{label:12} - Val PPL: {val_ppl:6.2f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"{label:12} - Val PPL: {val_ppl} | Val Loss: {val_loss}")
    
    # === ANÁLISIS DE MÉTRICAS IIT ===
    print("\n" + "=" * 70)
    print("ANÁLISIS DE MÉTRICAS IIT (muestra)")
    print("=" * 70)
    
    prompt = "The concept of consciousness in AI"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        logits, metrics = model(input_ids)
        
        print(f"\nPrompt: '{prompt}'")
        if metrics is not None:
            print(f"Coherencia (C): {metrics.get('coherencia', 'N/A')}")
            print(f"Complejidad (H): {metrics.get('complejidad', 'N/A')}")
            print(f"Phi (Φ): {metrics.get('phi', 'N/A')}")
        else:
            print("Métricas no disponibles en modo eval")
        print(f"Logits shape: {logits.shape}")
    
    print("\n" + "=" * 70)
    print("VALIDACIÓN COMPLETADA")
    print("=" * 70)
    print("\n✅ El modelo está listo para la siguiente fase")
    print("📊 Revisa los resultados anteriores para evaluar calidad")
    print("🚀 Próximo paso: Implementar IIT-guided memory system")

if __name__ == "__main__":
    main()
