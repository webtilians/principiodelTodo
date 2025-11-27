"""
Entrenamiento rápido del Gate para distinguir trivialidades.
El Gate debe aprender a dar importance bajo a saludos/cortesías.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os

# ============================================================================
# DATOS DE ENTRENAMIENTO
# ============================================================================

TRIVIAL_PHRASES = [
    # Saludos
    "hola", "hello", "hi", "hey", "buenas", "buenos días", "buenas tardes", 
    "buenas noches", "qué tal", "cómo estás", "cómo va", "qué hay",
    # Cortesías
    "gracias", "de nada", "por favor", "perdón", "disculpa", "lo siento",
    # Afirmaciones simples
    "ok", "vale", "sí", "no", "claro", "entendido", "perfecto", "genial",
    "bien", "mal", "regular", "más o menos", "ya", "ajá", "mmm", "ah", "oh",
    # Despedidas
    "adiós", "chao", "bye", "hasta luego", "nos vemos", "hasta mañana",
    # Relleno
    "pues", "bueno", "entonces", "a ver", "veamos", "oye", "mira",
]

IMPORTANT_PHRASES = [
    # Identidad
    "me llamo Juan", "mi nombre es María", "soy Pedro García", "tengo 25 años",
    # Contacto
    "mi teléfono es 666123456", "mi email es juan@gmail.com", "vivo en Madrid",
    "mi dirección es Calle Mayor 5",
    # Familia
    "mi hermano se llama Pedro", "mi madre es profesora", "mi padre trabaja en banco",
    "mi esposa es doctora", "tengo dos hijos",
    # Preferencias
    "me gusta el fútbol", "prefiero el café", "mi color favorito es azul",
    "odio las espinacas", "me encanta la música",
    # Recordatorios
    "mañana tengo cita con el médico", "el viernes es mi cumpleaños",
    "recuerda llamar a Juan", "no olvides comprar leche",
    # Información personal
    "trabajo como ingeniero", "estudio medicina", "mi coche es rojo",
    "tengo un perro llamado Max", "nací en Barcelona",
]

# ============================================================================
# MODELO
# ============================================================================

class TrivialityGate(nn.Module):
    """Gate especializado en detectar trivialidades."""
    
    def __init__(self, vocab_size=256, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        
        # Transformer ligero
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2,
            dropout=0.1, batch_first=True
        )
        
        # Gate de importancia
        self.importance_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        x = self.embedding(input_ids)
        seq_len = input_ids.size(1)
        x = x + self.position_embedding[:, :seq_len, :]
        x = self.encoder(x)
        x = x.mean(dim=1)  # [batch, hidden]
        importance = self.importance_gate(x)  # [batch, 1]
        return importance.squeeze(-1)


def text_to_ids(text, max_len=64):
    ids = [ord(c) % 256 for c in text.lower()[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids)


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Dispositivo: {device}")
    
    model = TrivialityGate().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCELoss()
    
    # Preparar datos
    trivial_ids = torch.stack([text_to_ids(p) for p in TRIVIAL_PHRASES]).to(device)
    important_ids = torch.stack([text_to_ids(p) for p in IMPORTANT_PHRASES]).to(device)
    
    trivial_labels = torch.zeros(len(TRIVIAL_PHRASES)).to(device)  # 0 = trivial
    important_labels = torch.ones(len(IMPORTANT_PHRASES)).to(device)  # 1 = importante
    
    all_ids = torch.cat([trivial_ids, important_ids])
    all_labels = torch.cat([trivial_labels, important_labels])
    
    print(f"📊 Datos: {len(trivial_ids)} triviales + {len(important_ids)} importantes")
    print(f"📈 Entrenando...")
    
    best_acc = 0
    for epoch in range(200):
        model.train()
        
        # Shuffle
        perm = torch.randperm(len(all_ids))
        ids = all_ids[perm]
        labels = all_labels[perm]
        
        # Forward
        predictions = model(ids)
        loss = criterion(predictions, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluar
        model.eval()
        with torch.no_grad():
            preds = model(all_ids)
            predicted = (preds > 0.5).float()
            correct = (predicted == all_labels).sum().item()
            acc = correct / len(all_labels)
            
            # Scores por clase
            trivial_scores = model(trivial_ids)
            important_scores = model(important_ids)
        
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': acc,
                'epoch': epoch
            }, 'models/triviality_gate.pt')
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, acc={acc*100:.1f}%, "
                  f"trivial={trivial_scores.mean().item():.3f}, important={important_scores.mean().item():.3f}")
    
    print(f"\n✅ Mejor accuracy: {best_acc*100:.1f}%")
    print(f"💾 Modelo guardado en: models/triviality_gate.pt")
    
    # Test final
    print("\n🧪 Test final:")
    model.eval()
    with torch.no_grad():
        test_phrases = [
            ("hola", "trivial"),
            ("ok", "trivial"),
            ("buenos días", "trivial"),
            ("me llamo Enrique", "importante"),
            ("mi teléfono es 123456", "importante"),
            ("mañana tengo cita", "importante"),
        ]
        for phrase, expected in test_phrases:
            ids = text_to_ids(phrase).unsqueeze(0).to(device)
            score = model(ids).item()
            pred = "importante" if score > 0.5 else "trivial"
            status = "✓" if pred == expected else "✗"
            print(f"  {phrase:25} → {score:.3f} ({pred}) {status}")


if __name__ == "__main__":
    train()
