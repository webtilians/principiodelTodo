import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

# Añadir el directorio src al path para importar el modelo
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from infinito_v5_2_refactored import InfinitoV52Refactored


class SimpleTextDataset(Dataset):
    """
    Dataset simple que tokeniza texto por espacios y construye secuencias de longitud fija.

    Cada palabra se asigna a un índice en un vocabulario limitado. Se generan pares
    (input_ids, target_ids) desplazando una posición la secuencia de destino.
    """

    def __init__(self, text: str, seq_len: int = 128, vocab_size: int = 5000):
        self.seq_len = seq_len
        words = text.lower().split()
        from collections import Counter
        counts = Counter(words)
        most_common = counts.most_common(vocab_size - 3)
        # Vocabulario con tokens especiales
        self.vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)
        # Convertir palabras a índices
        self.tokens = [self.vocab.get(w, 1) for w in words]

    def __len__(self) -> int:
        # Devuelve el número de secuencias disponibles en el corpus
        return max(1, (len(self.tokens) - self.seq_len) // self.seq_len)

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        seq = self.tokens[start:start + self.seq_len + 1]
        # Rellenar con padding si la secuencia es demasiado corta
        if len(seq) < self.seq_len + 1:
            seq += [0] * (self.seq_len + 1 - len(seq))
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids


def train_one_epoch(model, loader, optimizer, device):
    """Entrena el modelo durante una época y devuelve (loss_promedio, perplexity)."""
    model.train()
    total_loss = 0.0
    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad()
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        batch_size, seq_len, vocab_size = logits.shape
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=0
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(min(avg_loss, 20))


def evaluate(model, loader, device):
    """Evalúa el modelo en el conjunto de validación y devuelve (loss_promedio, perplexity)."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            output = model(input_ids)
            logits = output[0] if isinstance(output, tuple) else output
            batch_size, seq_len, vocab_size = logits.shape
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
                ignore_index=0
            )
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(min(avg_loss, 20))


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento personalizado para Infinito V5.2 refactorizado con búsqueda de hiperparámetros."
    )
    parser.add_argument('--text_file', type=str, default=None, help='Ruta a un archivo de texto para entrenar')
    parser.add_argument('--epochs', type=int, default=2, help='Número de épocas por configuración')
    parser.add_argument('--seq_len', type=int, default=128, help='Longitud de las secuencias')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamaño del batch')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Dispositivo de cálculo')
    args = parser.parse_args()

    # Cargar corpus
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Corpus artificial si no se proporciona archivo
        text = "La inteligencia artificial generativa puede crear secuencias complejas y coherentes. " * 100

    dataset = SimpleTextDataset(text, seq_len=args.seq_len, vocab_size=5000)
    # Dividir dataset en entrenamiento (90%) y validación (10%)
    n = len(dataset)
    val_size = max(1, n // 10)
    train_size = n - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Definir espacio de búsqueda para hidden_dim y num_layers
    hidden_dims = [256, 512]
    num_layers_options = [4, 6]
    best_ppl = float('inf')
    best_config = None

    for hidden_dim in hidden_dims:
        for num_layers in num_layers_options:
            print(f"\nEvaluando configuración: hidden_dim={hidden_dim}, num_layers={num_layers}")
            model = InfinitoV52Refactored(
                vocab_size=5000,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=8,
                use_improved_memory=True,
                use_improved_iit=True,
                use_learnable_phi=True,
                use_stochastic_exploration=True
            )
            model.to(args.device)
            optimizer = optim.Adam(model.parameters(), lr=3e-4)

            # Entrenar durante 'epochs' épocas
            for epoch in range(1, args.epochs + 1):
                train_loss, train_ppl = train_one_epoch(model, train_loader, optimizer, args.device)
                val_loss, val_ppl = evaluate(model, val_loader, args.device)
                print(
                    f"  Época {epoch}: loss={train_loss:.4f}, perplexity={train_ppl:.2f} | "
                    f"val_loss={val_loss:.4f}, val_perplexity={val_ppl:.2f}"
                )

            # Actualizar mejor configuración
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                best_config = {'hidden_dim': hidden_dim, 'num_layers': num_layers}
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs('models/custom', exist_ok=True)
                model_path = (
                    f"models/custom/infinito_v5_2_custom_hd{hidden_dim}_nl{num_layers}_{timestamp}.pt"
                )
                torch.save(model.state_dict(), model_path)
                print(f"  Nueva mejor perplexity: {val_ppl:.2f} guardado en {model_path}")

    print(
        f"\nMejor configuración encontrada: {best_config} con perplexity {best_ppl:.2f}"
    )


if __name__ == '__main__':
    main()
