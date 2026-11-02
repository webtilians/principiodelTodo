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
    # Saludos básicos
    "hola", "hello", "hi", "hey", "buenas", "buenos días", "buenas tardes", 
    "buenas noches", "qué tal", "cómo estás", "cómo va", "qué hay",
    "hola qué tal", "hey qué pasa", "buenas qué tal",
    # Cortesías
    "gracias", "de nada", "por favor", "perdón", "disculpa", "lo siento",
    "muchas gracias", "gracias por todo", "te lo agradezco",
    # Afirmaciones simples
    "ok", "vale", "sí", "no", "claro", "entendido", "perfecto", "genial",
    "bien", "mal", "regular", "más o menos", "ya", "ajá", "mmm", "ah", "oh",
    "de acuerdo", "está bien", "vale vale", "okey", "oki",
    # Despedidas
    "adiós", "chao", "bye", "hasta luego", "nos vemos", "hasta mañana",
    "que te vaya bien", "cuídate", "hasta pronto",
    # Relleno
    "pues", "bueno", "entonces", "a ver", "veamos", "oye", "mira",
    "vamos a ver", "déjame ver", "espera", "un momento",
    # Preguntas vacías (no aportan info nueva)
    "cómo", "qué", "cuál", "dónde", "cuándo", "por qué",
    "qué pasa", "qué tal", "cómo vas", "qué dices",
    "cuéntame algo sobre ti", "háblame de ti", "qué me cuentas",
    "cuentame algo sobre ti", "hablame de ti", "que me cuentas",  # sin tildes
    "cuentame sobre ti", "hablame sobre ti", "cuentame de ti",
    "y tú qué", "qué opinas", "tú qué dices",
    # Preguntas genéricas sin contexto personal
    "qué hora es", "qué día es hoy", "qué tiempo hace",
    "qué prioridades debería tener", "qué debería hacer",
    # Referencias vagas sin información (preguntas sobre personas sin dar info nueva)
    "y eso", "y qué más", "algo más", "qué más",
    "y mi primo andrés", "y mi padre", "y mi madre",  # preguntas sin info nueva
    "y mi primo andres", "y mi hermano", "y mi hermana",  # sin tilde también
    "y respecto a mi padre", "y qué pasa con mi madre",
    "y mi primo", "y mi tío", "y mi abuelo", "y tu familia",
    "qué sabes de mi padre", "qué sabes de mi madre",
    "y juan", "y pedro", "y maría",  # nombres solos como pregunta
]

IMPORTANT_PHRASES = [
    # Identidad con datos concretos
    "me llamo Juan", "mi nombre es María", "soy Pedro García", "tengo 25 años",
    "soy Enrique", "me llamo Ana López", "mi apellido es Martínez",
    "hola infinito soy enrique", "hola me llamo carlos",
    # Contacto
    "mi teléfono es 666123456", "mi email es juan@gmail.com", "vivo en Madrid",
    "mi dirección es Calle Mayor 5", "mi móvil es 612345678",
    # Familia con información concreta
    "mi hermano se llama Pedro", "mi madre es profesora", "mi padre trabaja en banco",
    "mi esposa es doctora", "tengo dos hijos", "mi hijo tiene 5 años",
    "mi primo andres monta en bici", "mi hermana estudia medicina",
    "mi padre tiene una tienda", "mi abuela vive en el pueblo",
    # Preferencias y gustos
    "me gusta el fútbol", "prefiero el café", "mi color favorito es azul",
    "odio las espinacas", "me encanta la música", "hago descenso en bici",
    "mi deporte favorito es el ciclismo", "me gusta correr por las mañanas",
    # Recordatorios y eventos
    "mañana tengo cita con el médico", "el viernes es mi cumpleaños",
    "recuerda llamar a Juan", "no olvides comprar leche",
    "el sábado vamos a la playa", "la semana que viene tengo examen",
    # Actividades y logros
    "hoy he igualado mi mejor tiempo", "esta mañana he montado la suspensión",
    "ayer terminé el proyecto", "he conseguido el trabajo",
    "hoy he corrido 10 kilómetros", "acabo de aprobar el examen",
    # Información personal específica
    "trabajo como ingeniero", "estudio medicina", "mi coche es rojo",
    "tengo un perro llamado Max", "nací en Barcelona",
    "peso 75 kilos", "mido 1.80 metros", "mi bici es una Scott",
    # Datos técnicos/específicos
    "el sag ideal es 25 por ciento", "uso una horquilla de 160mm",
    "mi presupuesto es 500 euros", "necesito 8 horas de sueño",
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
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
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
    best_separation = 0
    for epoch in range(500):  # Más épocas
        model.train()
        
        # Shuffle
        perm = torch.randperm(len(all_ids))
        ids = all_ids[perm]
        labels = all_labels[perm]
        
        # Forward
        predictions = model(ids)
        loss = criterion(predictions, labels)
        
        # Añadir margin loss para mejor separación
        trivial_preds = predictions[:len(TRIVIAL_PHRASES)]
        important_preds = predictions[len(TRIVIAL_PHRASES):]
        margin_loss = torch.relu(0.3 - (important_preds.mean() - trivial_preds.mean()))
        total_loss = loss + 0.5 * margin_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
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
            separation = important_scores.mean().item() - trivial_scores.mean().item()
        
        # Guardar si tiene buena accuracy Y buena separación
        if acc >= best_acc and separation > best_separation:
            best_acc = acc
            best_separation = separation
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': acc,
                'separation': separation,
                'epoch': epoch
            }, 'models/triviality_gate.pt')
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, acc={acc*100:.1f}%, "
                  f"trivial={trivial_scores.mean().item():.3f}, important={important_scores.mean().item():.3f}, "
                  f"sep={separation:.3f}")
    
    print(f"\n✅ Mejor accuracy: {best_acc*100:.1f}%")
    print(f"📊 Mejor separación: {best_separation:.3f}")
    print(f"💾 Modelo guardado en: models/triviality_gate.pt")
    
    # Test final con casos problemáticos
    print("\n🧪 Test final (casos problemáticos):")
    model.eval()
    with torch.no_grad():
        test_phrases = [
            ("hola", "trivial"),
            ("cuentame algo sobre ti", "trivial"),
            ("que pasa", "trivial"),
            ("como estas", "trivial"),
            ("y mi primo andres", "trivial"),
            ("hola infinito soy enrique", "importante"),
            ("mi primo andres monta en bici", "importante"),
            ("hoy he igualado mi mejor tiempo", "importante"),
            ("peso 75 kilos", "importante"),
        ]
        correct = 0
        for phrase, expected in test_phrases:
            ids = text_to_ids(phrase).unsqueeze(0).to(device)
            score = model(ids).item()
            pred = "importante" if score > 0.5 else "trivial"
            status = "✓" if pred == expected else "✗"
            if pred == expected:
                correct += 1
            print(f"  {phrase:35} → {score:.3f} ({pred}) {status}")
        print(f"\n  Casos test: {correct}/{len(test_phrases)} correctos")


if __name__ == "__main__":
    train()
