"""
Generador de texto mejorado con repetition penalty y temperature sampling
"""

import torch
import sys
from pathlib import Path
from transformers import GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from src.infinito_v5_2_refactored import InfinitoV52Refactored

def generate_text_improved(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.8,
    repetition_penalty=1.2,
    top_k=50,
    top_p=0.9,
    device='cuda'
):
    """
    Genera texto con temperature sampling + repetition penalty + nucleus sampling
    
    Args:
        model: Modelo INFINITO V5.2
        tokenizer: GPT2Tokenizer
        prompt: Texto inicial
        max_length: Máximo de tokens a generar
        temperature: 0.1-2.0 (más alto = más creativo)
        repetition_penalty: >1.0 penaliza repeticiones (1.2 recomendado)
        top_k: Limita a los k tokens más probables (0 = desactivado)
        top_p: Nucleus sampling - acumula hasta p probabilidad
        device: 'cuda' o 'cpu'
    
    Returns:
        str: Texto generado
    """
    model.eval()
    
    # Tokenizar prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :]
            
            # === REPETITION PENALTY ===
            if repetition_penalty != 1.0:
                # Penalizar tokens ya generados
                for token_id in set(generated[0].tolist()):
                    # Si el token ya apareció, reducir su logit
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
            
            # === TEMPERATURE SAMPLING ===
            next_token_logits = next_token_logits / temperature
            
            # === TOP-K FILTERING ===
            if top_k > 0:
                # Mantener solo top_k logits, el resto a -inf
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # === TOP-P (NUCLEUS) FILTERING ===
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remover tokens con probabilidad acumulada > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Mantener al menos 1 token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Convertir a probabilidades
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Samplear siguiente token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop si EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decodificar
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


def load_model(checkpoint_path, device='cuda'):
    """Carga modelo desde checkpoint"""
    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', 50257)
    
    model = InfinitoV52Refactored(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        memory_slots=256,
        use_improved_memory=True,
        use_stochastic_exploration=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✅ Modelo cargado ({sum(p.numel() for p in model.parameters()):,} parámetros)")
    return model


def main():
    print("=" * 70)
    print("GENERACIÓN MEJORADA - INFINITO V5.2")
    print("Temperature + Repetition Penalty + Nucleus Sampling")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("⚠️  CPU detectado - será lento")
    
    # Cargar modelo
    checkpoint_path = "models/checkpoints/infinito_v5.2_real_best.pt"
    model = load_model(checkpoint_path, device)
    
    # Tokenizer
    print("\nCargando tokenizer GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # === PRUEBAS DE GENERACIÓN ===
    print("\n" + "=" * 70)
    print("EJEMPLOS DE GENERACIÓN")
    print("=" * 70)
    
    test_cases = [
        {
            'prompt': "The theory of relativity",
            'temp': 0.7,
            'rep_penalty': 1.2,
            'max_len': 50
        },
        {
            'prompt': "In a world where artificial intelligence",
            'temp': 0.9,
            'rep_penalty': 1.3,
            'max_len': 60
        },
        {
            'prompt': "Once upon a time",
            'temp': 1.0,
            'rep_penalty': 1.5,
            'max_len': 70
        }
    ]
    
    for i, config in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"EJEMPLO {i}")
        print(f"{'─' * 70}")
        print(f"📝 Prompt: '{config['prompt']}'")
        print(f"🌡️  Temperature: {config['temp']}")
        print(f"🔁 Repetition Penalty: {config['rep_penalty']}")
        print(f"🔄 Generando...")
        
        try:
            import time
            start = time.time()
            
            generated = generate_text_improved(
                model=model,
                tokenizer=tokenizer,
                prompt=config['prompt'],
                max_length=config['max_len'],
                temperature=config['temp'],
                repetition_penalty=config['rep_penalty'],
                top_k=50,
                top_p=0.95,
                device=device
            )
            
            elapsed = time.time() - start
            
            print(f"\n✨ RESULTADO:")
            print(f"   {generated}")
            print(f"\n⏱️  Tiempo: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETADO")
    print("=" * 70)
    
    # Info del modelo
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    val_loss = checkpoint.get('val_loss', None)
    
    if val_loss:
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        print(f"\n📊 Modelo:")
        print(f"   Val PPL: {val_ppl:.2f}")
        print(f"   Época: {checkpoint.get('epoch', 'N/A')}")
        
        print(f"\n💡 Mejoras aplicadas:")
        print(f"   ✅ Temperature sampling (vs greedy)")
        print(f"   ✅ Repetition penalty (evita repeticiones)")
        print(f"   ✅ Top-K filtering (k=50)")
        print(f"   ✅ Nucleus sampling (p=0.95)")

if __name__ == "__main__":
    main()
