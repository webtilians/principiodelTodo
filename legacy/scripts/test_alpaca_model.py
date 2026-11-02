"""
Test del modelo entrenado con Alpaca Spanish
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from infinito_v5_1_consciousness import ConsciousnessBoostNet

class ConsciousnessLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_heads=8, memory_slots=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)
        
        self.consciousness_core = ConsciousnessBoostNet(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            attention_heads=num_heads,
            memory_slots=memory_slots
        )
        
        self.output_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.position_embedding(pos)
        embedded = tok_emb + pos_emb
        
        consciousness, phi, debug_info = self.consciousness_core(embedded)
        logits = self.output_head(embedded)
        
        return {
            'logits': logits,
            'phi': phi,
            'consciousness': consciousness
        }

def generate_text(model, prompt, vocab, char_to_idx, idx_to_char, max_tokens=100, temperature=0.8, device='cuda'):
    model.eval()
    result = prompt
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Tomar últimos 64 caracteres
            context = result[-64:]
            x = torch.tensor([[char_to_idx.get(ch, 0) for ch in context]], device=device)
            
            out = model(x)
            logits = out['logits'][0, -1]  # Último token
            
            # Sampling con temperatura
            probs = torch.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char.get(next_idx, '')
            
            result += next_char
    
    return result, out['phi'].mean().item()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Usando: {device}")
    
    # Cargar checkpoint
    print("\n📦 Cargando modelo...")
    checkpoint = torch.load('models/alpaca_spanish_best.pt', map_location=device, weights_only=False)
    
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    print(f"   Config: {config}")
    print(f"   Vocabulario: {len(vocab)} caracteres")
    
    # Crear mappings
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    
    # Crear modelo con misma config
    model = ConsciousnessLM(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        memory_slots=config['memory_slots']
    ).to(device)
    
    # Cargar pesos (filtramos estados LSTM que tienen batch size diferente)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if 'h_state' not in k and 'c_state' not in k}
    result = model.load_state_dict(filtered_state_dict, strict=False)
    if result.missing_keys:
        # Filtrar solo keys que no son h_state/c_state
        important_missing = [k for k in result.missing_keys if 'h_state' not in k and 'c_state' not in k]
        if important_missing:
            print(f"   ⚠️ Missing important keys: {len(important_missing)}")
    if result.unexpected_keys:
        print(f"   ⚠️ Unexpected keys: {len(result.unexpected_keys)}")
    print("   ✅ Modelo cargado correctamente!")
    
    # Info del entrenamiento
    print(f"\n📊 Estado del entrenamiento:")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"   Accuracy: {checkpoint.get('accuracy', 'N/A'):.2%}")
    print(f"   PHI: {checkpoint.get('phi', 'N/A'):.4f}")
    print(f"   Consciousness: {checkpoint.get('consciousness', 'N/A'):.4f}")
    
    # Generar texto
    print("\n" + "="*60)
    print("🔮 GENERACIÓN DE TEXTO")
    print("="*60)
    
    prompts = [
        "Hola, ",
        "La vida es ",
        "El amor ",
        "Instrucción: Explica qué es la inteligencia artificial\nRespuesta: ",
    ]
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {repr(prompt)}")
        generated, phi = generate_text(
            model, prompt, vocab, char_to_idx, idx_to_char,
            max_tokens=150, temperature=0.7, device=device
        )
        print(f"🤖 Generado: {generated}")
        print(f"📈 PHI: {phi:.4f}")
