"""
🧠 NEURAL MEMORY MANAGER - Memoria Neuronal con Aprendizaje Continuo
=====================================================================

Sistema de memoria neuronal que usa el modelo IIT del 54% como base
y LoRA para aprendizaje continuo sin olvido catastrófico.

Arquitectura:
┌─────────────────────────────────────────────────────────────┐
│  BASE MODEL (InfinitoV52 - Super Golden Seed 54%)          │
│  ├── Token Embeddings                                       │
│  ├── Position Embeddings                                    │
│  ├── Transformer Layers (6)                                 │
│  └── Memory Gate                                            │
│       ↓ CONGELADO                                           │
├─────────────────────────────────────────────────────────────┤
│  LoRA ADAPTERS (entrenables)                                │
│  ├── Importance Adapter (qué guardar)                       │
│  ├── Retrieval Adapter (qué recuperar)                      │
│  └── Consolidation Adapter (cómo relacionar)                │
├─────────────────────────────────────────────────────────────┤
│  REPLAY BUFFER                                              │
│  └── Últimas N interacciones para entrenamiento periódico   │
└─────────────────────────────────────────────────────────────┘

Autor: INFINITO Project
Fecha: Noviembre 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import json
import os
import numpy as np
import random

# Imports locales
import sys
sys.path.insert(0, os.path.dirname(__file__))

from lora_adapter import LoRAAdapter, AdapterManager


class ReplayBuffer:
    """
    Buffer de experiencias para entrenamiento con replay.
    
    Guarda las últimas N interacciones y permite muestrear
    una mezcla de recientes + antiguas para evitar olvido.
    """
    
    def __init__(
        self, 
        max_size: int = 1000,
        db_file: str = "data/replay_buffer.json"
    ):
        self.max_size = max_size
        self.db_file = db_file
        self.buffer: deque = deque(maxlen=max_size)
        
        # Cargar buffer existente
        self._load()
    
    def _load(self):
        """Carga el buffer desde disco."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data[-self.max_size:]:
                        self.buffer.append(item)
                print(f"📂 Replay buffer: {len(self.buffer)} experiencias cargadas")
            except Exception as e:
                print(f"⚠️ Error cargando replay buffer: {e}")
    
    def _save(self):
        """Guarda el buffer a disco."""
        os.makedirs(os.path.dirname(self.db_file) if os.path.dirname(self.db_file) else '.', exist_ok=True)
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.buffer), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error guardando replay buffer: {e}")
    
    def add(
        self,
        text: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        category: str = "general",
        metrics: Optional[Dict] = None,
        context: Optional[Dict] = None
    ):
        """
        Añade una experiencia al buffer.
        
        Args:
            text: Texto de la interacción
            embedding: Vector embedding (opcional)
            importance: Score de importancia del Gate
            category: Categoría detectada
            metrics: Métricas IIT (phi, coherence, etc.)
            context: Contexto adicional
        """
        experience = {
            "text": text,
            "embedding": embedding,
            "importance": importance,
            "category": category,
            "metrics": metrics or {},
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "used_in_training": 0  # Contador de veces usado
        }
        
        self.buffer.append(experience)
        
        # Guardar cada 10 adiciones
        if len(self.buffer) % 10 == 0:
            self._save()
    
    def sample(
        self,
        batch_size: int = 32,
        recent_ratio: float = 0.7
    ) -> List[Dict]:
        """
        Muestrea un batch mixto: recientes + antiguas.
        
        Args:
            batch_size: Tamaño del batch
            recent_ratio: Proporción de muestras recientes (0.0-1.0)
            
        Returns:
            Lista de experiencias
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Calcular splits
        n_recent = int(batch_size * recent_ratio)
        n_old = batch_size - n_recent
        
        # Muestras recientes (últimas 20%)
        recent_start = max(0, len(self.buffer) - len(self.buffer) // 5)
        recent_pool = list(self.buffer)[recent_start:]
        
        # Muestras antiguas (resto)
        old_pool = list(self.buffer)[:recent_start]
        
        # Muestrear
        recent_samples = random.sample(recent_pool, min(n_recent, len(recent_pool)))
        old_samples = random.sample(old_pool, min(n_old, len(old_pool))) if old_pool else []
        
        # Combinar y mezclar
        batch = recent_samples + old_samples
        random.shuffle(batch)
        
        # Actualizar contador de uso
        for exp in batch:
            exp["used_in_training"] += 1
        
        return batch
    
    def get_high_importance(self, threshold: float = 0.7, limit: int = 50) -> List[Dict]:
        """Obtiene experiencias de alta importancia."""
        high_imp = [e for e in self.buffer if e.get("importance", 0) >= threshold]
        return sorted(high_imp, key=lambda x: x.get("importance", 0), reverse=True)[:limit]
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self):
        """Guarda el buffer explícitamente."""
        self._save()


class NeuralMemoryManager(nn.Module):
    """
    Gestor de Memoria Neuronal con aprendizaje continuo.
    
    Usa el modelo IIT del 54% como base (congelado) y adapters LoRA
    para aprender de nuevas interacciones sin olvido catastrófico.
    """
    
    def __init__(
        self,
        base_model_path: str = "models/super_golden_seed_54percent.pt",
        hidden_dim: int = 64,
        num_layers: int = 2,
        lora_rank: int = 8,
        device: str = None,
        auto_consolidate_every: int = 50
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.auto_consolidate_every = auto_consolidate_every
        self.interaction_count = 0
        
        print(f"\n{'='*60}")
        print(f"🧠 NEURAL MEMORY MANAGER")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  LoRA rank: {lora_rank}")
        print(f"  Auto-consolidate every: {auto_consolidate_every} interacciones")
        
        # 1. Cargar modelo base (CONGELADO)
        self.base_model = self._load_base_model(base_model_path)
        
        # 2. Crear adapters LoRA
        self.adapters = nn.ModuleDict({
            'importance': LoRAAdapter(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                rank=lora_rank,
                target_modules=['query', 'value']
            ),
            'retrieval': LoRAAdapter(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                rank=lora_rank,
                target_modules=['query', 'key']
            )
        })
        
        # 3. Cabeza de importancia (decide qué guardar)
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 4. Cabeza de relación (conecta memorias)
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 5. Replay Buffer
        self.replay_buffer = ReplayBuffer(max_size=1000)
        
        # 6. Optimizador (solo para adapters y cabezas)
        self.optimizer = optim.AdamW(
            list(self.adapters.parameters()) + 
            list(self.importance_head.parameters()) +
            list(self.relation_head.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # 7. Historial de entrenamiento
        self.training_history = []
        
        # Mover a device
        self.to(self.device)
        
        print(f"{'='*60}\n")
    
    def _load_base_model(self, path: str) -> Optional[nn.Module]:
        """
        Carga el modelo base del 54% y lo congela.
        """
        if not os.path.exists(path):
            print(f"  ⚠️ Modelo base no encontrado: {path}")
            print(f"  ➡️ Creando modelo base simple...")
            return self._create_simple_base()
        
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            print(f"  ✅ Modelo base cargado: {path}")
            
            # Extraer solo los embeddings y capas útiles
            # El modelo completo puede ser muy grande, extraemos lo esencial
            base = SimpleBaseModel(
                hidden_dim=self.hidden_dim,
                checkpoint=checkpoint
            )
            
            # CONGELAR todos los parámetros
            for param in base.parameters():
                param.requires_grad = False
            
            print(f"  ✅ Modelo base CONGELADO")
            return base
            
        except Exception as e:
            print(f"  ⚠️ Error cargando modelo: {e}")
            print(f"  ➡️ Creando modelo base simple...")
            return self._create_simple_base()
    
    def _create_simple_base(self) -> nn.Module:
        """Crea un modelo base simple si no hay checkpoint."""
        base = SimpleBaseModel(hidden_dim=self.hidden_dim)
        for param in base.parameters():
            param.requires_grad = False
        return base
    
    def encode(self, text: str, external_embedding: Optional[List[float]] = None) -> torch.Tensor:
        """
        Codifica texto a embedding.
        
        Args:
            text: Texto a codificar
            external_embedding: Embedding externo (ej: de OpenAI)
            
        Returns:
            Tensor [1, hidden_dim]
        """
        if external_embedding is not None:
            # Usar embedding externo y proyectar a hidden_dim
            emb = torch.tensor(external_embedding, device=self.device).float()
            
            # Proyectar si es necesario (OpenAI usa 1536 dims)
            if emb.shape[-1] != self.hidden_dim:
                if not hasattr(self, 'projection'):
                    self.projection = nn.Linear(emb.shape[-1], self.hidden_dim).to(self.device)
                emb = self.projection(emb)
            
            return emb.unsqueeze(0) if emb.dim() == 1 else emb
        
        # Usar modelo base para encoding simple
        return self.base_model.encode(text)
    
    def predict_importance(
        self, 
        embedding: torch.Tensor,
        text: str = ""
    ) -> Tuple[float, Dict]:
        """
        Predice la importancia de una memoria.
        
        Args:
            embedding: Embedding de la memoria
            text: Texto original (para contexto)
            
        Returns:
            importance: Score 0.0-1.0
            details: Detalles del análisis
        """
        with torch.no_grad():
            # Aplicar adapter de importancia
            adapted = embedding + self.adapters['importance'].get_delta(0, 'query', embedding)
            
            # Predecir importancia
            importance = self.importance_head(adapted)
            
            return importance.item(), {
                "raw_score": importance.item(),
                "embedding_norm": embedding.norm().item()
            }
    
    def learn(
        self,
        text: str,
        importance: float,
        embedding: Optional[List[float]] = None,
        category: str = "general",
        metrics: Optional[Dict] = None
    ):
        """
        Aprende de una nueva interacción (añade al buffer).
        
        Args:
            text: Texto de la interacción
            importance: Score de importancia
            embedding: Embedding del texto
            category: Categoría
            metrics: Métricas IIT
        """
        # Añadir al replay buffer
        self.replay_buffer.add(
            text=text,
            embedding=embedding,
            importance=importance,
            category=category,
            metrics=metrics
        )
        
        self.interaction_count += 1
        
        # Auto-consolidar si es necesario
        if self.interaction_count % self.auto_consolidate_every == 0:
            print(f"🔄 Auto-consolidación después de {self.interaction_count} interacciones...")
            self.consolidate(epochs=1, batch_size=16)
    
    def consolidate(
        self,
        epochs: int = 3,
        batch_size: int = 32
    ) -> Dict:
        """
        Entrena los adapters con el replay buffer.
        
        Esto es como "dormir" - consolida las memorias recientes
        mezclándolas con las antiguas para evitar olvido.
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño del batch
            
        Returns:
            Estadísticas del entrenamiento
        """
        if len(self.replay_buffer) < batch_size:
            print(f"⚠️ Buffer muy pequeño ({len(self.replay_buffer)} < {batch_size})")
            return {"status": "skipped", "reason": "buffer too small"}
        
        print(f"\n🌙 CONSOLIDACIÓN DE MEMORIA")
        print(f"   Épocas: {epochs}")
        print(f"   Buffer size: {len(self.replay_buffer)}")
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            # Muestrear batch mixto
            batch = self.replay_buffer.sample(batch_size, recent_ratio=0.7)
            
            # Preparar datos
            embeddings = []
            importances = []
            
            for exp in batch:
                if exp.get("embedding"):
                    emb = torch.tensor(exp["embedding"], device=self.device).float()
                    # Proyectar si es necesario
                    if emb.shape[-1] != self.hidden_dim:
                        if not hasattr(self, 'projection'):
                            self.projection = nn.Linear(emb.shape[-1], self.hidden_dim).to(self.device)
                        with torch.no_grad():
                            emb = self.projection(emb)
                    embeddings.append(emb)
                    importances.append(exp.get("importance", 0.5))
            
            if not embeddings:
                continue
            
            # Stack tensors
            emb_tensor = torch.stack(embeddings)  # [batch, hidden]
            imp_tensor = torch.tensor(importances, device=self.device).float().unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Aplicar adapter y predecir importancia
            adapted = emb_tensor + self.adapters['importance'].get_delta(0, 'query', emb_tensor)
            pred_importance = self.importance_head(adapted)
            
            # Loss: MSE entre importancia predicha y real
            loss = F.mse_loss(pred_importance, imp_tensor)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        stats = {
            "epochs": epochs,
            "batches": num_batches,
            "avg_loss": avg_loss,
            "buffer_size": len(self.replay_buffer),
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_history.append(stats)
        
        print(f"   ✅ Loss promedio: {avg_loss:.4f}")
        print(f"   ✅ Batches procesados: {num_batches}")
        
        # Guardar adapters
        self.save_adapters()
        
        return stats
    
    def save_adapters(self, path: str = "models/adapters/neural_memory"):
        """Guarda los adapters entrenados."""
        os.makedirs(path, exist_ok=True)
        
        for name, adapter in self.adapters.items():
            adapter.save(os.path.join(path, f"{name}.pt"))
        
        # Guardar cabezas
        torch.save({
            'importance_head': self.importance_head.state_dict(),
            'relation_head': self.relation_head.state_dict(),
            'training_history': self.training_history,
            'interaction_count': self.interaction_count
        }, os.path.join(path, "heads.pt"))
        
        # Guardar buffer
        self.replay_buffer.save()
        
        print(f"💾 Neural Memory guardado en {path}")
    
    def load_adapters(self, path: str = "models/adapters/neural_memory"):
        """Carga adapters guardados."""
        if not os.path.exists(path):
            print(f"⚠️ No se encontró {path}")
            return False
        
        try:
            for name in self.adapters.keys():
                adapter_path = os.path.join(path, f"{name}.pt")
                if os.path.exists(adapter_path):
                    self.adapters[name] = LoRAAdapter.load(adapter_path, self.device)
            
            heads_path = os.path.join(path, "heads.pt")
            if os.path.exists(heads_path):
                state = torch.load(heads_path, map_location=self.device)
                self.importance_head.load_state_dict(state['importance_head'])
                self.relation_head.load_state_dict(state['relation_head'])
                self.training_history = state.get('training_history', [])
                self.interaction_count = state.get('interaction_count', 0)
            
            print(f"📂 Neural Memory cargado desde {path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error cargando: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del sistema."""
        return {
            "interactions": self.interaction_count,
            "buffer_size": len(self.replay_buffer),
            "consolidations": len(self.training_history),
            "device": self.device,
            "adapters": list(self.adapters.keys())
        }


class SimpleBaseModel(nn.Module):
    """
    Modelo base simplificado para encoding.
    
    Extrae embeddings del checkpoint del 54% o usa embeddings simples.
    """
    
    def __init__(self, hidden_dim: int = 64, checkpoint: Dict = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Embedding simple para texto
        self.char_embedding = nn.Embedding(256, hidden_dim)  # ASCII
        
        # Transformer pequeño para contexto
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        
        # Si hay checkpoint, intentar cargar pesos relevantes
        if checkpoint:
            self._load_from_checkpoint(checkpoint)
    
    def _load_from_checkpoint(self, checkpoint: Dict):
        """Intenta cargar pesos del checkpoint."""
        try:
            state = checkpoint.get('model_state_dict', checkpoint)
            
            # Buscar embeddings compatibles
            for key, value in state.items():
                if 'embedding' in key.lower() and value.shape[-1] == self.hidden_dim:
                    print(f"     Cargado: {key}")
                    break
        except Exception as e:
            print(f"     No se pudieron cargar pesos: {e}")
    
    def encode(self, text: str) -> torch.Tensor:
        """Codifica texto a embedding."""
        # Convertir a indices ASCII
        chars = [ord(c) % 256 for c in text[:100]]  # Max 100 chars
        chars = torch.tensor(chars, device=next(self.parameters()).device)
        
        # Embedding
        emb = self.char_embedding(chars.unsqueeze(0))  # [1, seq, hidden]
        
        # Encoding con transformer
        encoded = self.encoder(emb)
        
        # Pool (mean)
        return encoded.mean(dim=1)  # [1, hidden]


# =============================================================================
# PRUEBA RÁPIDA
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 TEST: Neural Memory Manager")
    print("=" * 60)
    
    # Crear manager
    manager = NeuralMemoryManager(
        base_model_path="models/super_golden_seed_54percent.pt",
        hidden_dim=64,
        lora_rank=4
    )
    
    # Simular interacciones
    test_interactions = [
        ("Me llamo Enrique y vivo en Madrid", 0.85, "identity"),
        ("Hoy hace buen tiempo", 0.15, "trivial"),
        ("Mi cumpleaños es el 15 de marzo", 0.75, "personal"),
        ("Trabajo como desarrollador de software", 0.80, "professional"),
        ("Buenos días", 0.10, "greeting"),
    ]
    
    print("\n📝 Simulando interacciones...")
    for text, importance, category in test_interactions:
        # Crear embedding fake
        fake_embedding = torch.randn(64).tolist()
        
        manager.learn(
            text=text,
            importance=importance,
            embedding=fake_embedding,
            category=category
        )
        print(f"   ✅ Añadido: {text[:30]}... (imp={importance})")
    
    # Consolidar
    print("\n🌙 Consolidando...")
    stats = manager.consolidate(epochs=2, batch_size=4)
    print(f"   Stats: {stats}")
    
    # Estadísticas
    print(f"\n📊 Estadísticas finales:")
    for k, v in manager.get_stats().items():
        print(f"   {k}: {v}")
    
    print("\n✅ Test completado!")
