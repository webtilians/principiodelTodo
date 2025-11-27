"""
🔧 LoRA ADAPTER - Low-Rank Adaptation para Aprendizaje Continuo
================================================================

Implementación de LoRA (Low-Rank Adaptation) para entrenar adapters
pequeños sobre un modelo base congelado.

Ventajas:
- No hay olvido catastrófico (modelo base intacto)
- Entrenamiento rápido (pocos parámetros)
- Múltiples adapters por usuario/tarea
- Fácil rollback

Referencias:
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)

Autor: INFINITO Project
Fecha: Noviembre 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import os
import json


class LoRALayer(nn.Module):
    """
    Capa LoRA individual que se añade a una capa Linear existente.
    
    En lugar de modificar W directamente, añadimos:
        W' = W + (B @ A) * scaling
    
    Donde:
        - W: Pesos originales (congelados)
        - A: Matriz de rank bajo [in_features, rank]
        - B: Matriz de rank bajo [rank, out_features]
        - scaling: alpha / rank
    
    Parámetros entrenables: A y B (mucho menos que W)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrices de bajo rango (ESTAS se entrenan)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout para regularización
        self.dropout = nn.Dropout(dropout)
        
        # Inicialización (Kaiming para A, zeros para B)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Contador de actualizaciones
        self.update_count = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula el delta que se suma a la salida de la capa original.
        
        Args:
            x: Input tensor [batch, ..., in_features]
            
        Returns:
            delta: [batch, ..., out_features] para sumar a la salida original
        """
        # Aplicar dropout al input
        x = self.dropout(x)
        
        # LoRA: x @ A @ B * scaling
        # Usamos matmul para soportar batches
        delta = x @ self.lora_A @ self.lora_B * self.scaling
        
        return delta
    
    def get_num_params(self) -> int:
        """Retorna el número de parámetros entrenables."""
        return self.rank * (self.in_features + self.out_features)
    
    def merge_weights(self, original_weight: torch.Tensor) -> torch.Tensor:
        """
        Fusiona los pesos LoRA con los pesos originales.
        Útil para inferencia rápida (sin overhead de LoRA).
        
        Args:
            original_weight: Pesos originales [out_features, in_features]
            
        Returns:
            merged: Pesos fusionados [out_features, in_features]
        """
        with torch.no_grad():
            # W' = W + (A @ B)^T * scaling
            delta = (self.lora_A @ self.lora_B).T * self.scaling
            return original_weight + delta


class LoRALinear(nn.Module):
    """
    Wrapper que combina una capa Linear original con LoRA.
    
    La capa original se congela y solo se entrena el adapter LoRA.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Guardar referencia a la capa original (CONGELADA)
        self.original = original_layer
        for param in self.original.parameters():
            param.requires_grad = False
        
        # Crear adapter LoRA
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        self.merged = False  # Flag para modo inferencia
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Original + LoRA delta."""
        if self.merged:
            # En modo merged, usamos pesos fusionados (más rápido)
            return F.linear(x, self.merged_weight, self.original.bias)
        else:
            # En modo normal, calculamos original + delta
            original_output = self.original(x)
            lora_delta = self.lora(x)
            return original_output + lora_delta
    
    def merge(self):
        """Fusiona pesos para inferencia rápida."""
        if not self.merged:
            self.merged_weight = self.lora.merge_weights(self.original.weight)
            self.merged = True
    
    def unmerge(self):
        """Deshace la fusión para seguir entrenando."""
        self.merged = False
        if hasattr(self, 'merged_weight'):
            del self.merged_weight


class LoRAAdapter(nn.Module):
    """
    Adapter LoRA completo para un modelo Transformer.
    
    Aplica LoRA a las proyecciones Q, K, V y/o las capas FFN.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 6,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        target_modules: List[str] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rank = rank
        self.alpha = alpha
        
        # Por defecto, aplicar a Q, K, V
        if target_modules is None:
            target_modules = ['query', 'key', 'value']
        self.target_modules = target_modules
        
        # Crear adapters para cada capa
        self.adapters = nn.ModuleDict()
        
        for layer_idx in range(num_layers):
            layer_adapters = nn.ModuleDict()
            
            for module_name in target_modules:
                layer_adapters[module_name] = LoRALayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
            
            self.adapters[f'layer_{layer_idx}'] = layer_adapters
        
        # Estadísticas
        self.total_params = self._count_params()
        print(f"🔧 LoRA Adapter creado:")
        print(f"   - Rank: {rank}")
        print(f"   - Target modules: {target_modules}")
        print(f"   - Parámetros entrenables: {self.total_params:,}")
    
    def _count_params(self) -> int:
        """Cuenta parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_delta(
        self, 
        layer_idx: int, 
        module_name: str, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Obtiene el delta LoRA para una capa específica.
        
        Args:
            layer_idx: Índice de la capa transformer
            module_name: 'query', 'key', 'value', etc.
            x: Input tensor
            
        Returns:
            delta: Tensor para sumar a la salida original
        """
        layer_key = f'layer_{layer_idx}'
        if layer_key in self.adapters and module_name in self.adapters[layer_key]:
            return self.adapters[layer_key][module_name](x)
        return torch.zeros_like(x)
    
    def save(self, path: str):
        """Guarda solo los pesos del adapter (muy pequeño)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        state = {
            'state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'rank': self.rank,
                'alpha': self.alpha,
                'target_modules': self.target_modules
            }
        }
        torch.save(state, path)
        print(f"💾 Adapter guardado en {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'LoRAAdapter':
        """Carga un adapter guardado."""
        state = torch.load(path, map_location=device)
        
        adapter = cls(
            hidden_dim=state['config']['hidden_dim'],
            num_layers=state['config']['num_layers'],
            rank=state['config']['rank'],
            alpha=state['config']['alpha'],
            target_modules=state['config']['target_modules']
        )
        
        adapter.load_state_dict(state['state_dict'])
        print(f"📂 Adapter cargado desde {path}")
        return adapter


class AdapterManager:
    """
    Gestor de múltiples adapters (por usuario, tarea, etc.)
    """
    
    def __init__(self, adapters_dir: str = "models/adapters"):
        self.adapters_dir = adapters_dir
        os.makedirs(adapters_dir, exist_ok=True)
        
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.active_adapter: Optional[str] = None
        
        # Cargar índice de adapters existentes
        self.index_file = os.path.join(adapters_dir, "index.json")
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Carga el índice de adapters."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {"adapters": {}}
    
    def _save_index(self):
        """Guarda el índice de adapters."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def create_adapter(
        self,
        name: str,
        hidden_dim: int,
        num_layers: int = 6,
        rank: int = 8,
        description: str = ""
    ) -> LoRAAdapter:
        """Crea y registra un nuevo adapter."""
        adapter = LoRAAdapter(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rank=rank
        )
        
        self.adapters[name] = adapter
        self.index["adapters"][name] = {
            "path": os.path.join(self.adapters_dir, f"{name}.pt"),
            "description": description,
            "created": str(torch.datetime.datetime.now()) if hasattr(torch, 'datetime') else "unknown",
            "rank": rank,
            "num_layers": num_layers
        }
        self._save_index()
        
        return adapter
    
    def load_adapter(self, name: str, device: str = 'cpu') -> Optional[LoRAAdapter]:
        """Carga un adapter por nombre."""
        if name in self.adapters:
            return self.adapters[name]
        
        if name in self.index["adapters"]:
            path = self.index["adapters"][name]["path"]
            if os.path.exists(path):
                adapter = LoRAAdapter.load(path, device)
                self.adapters[name] = adapter
                return adapter
        
        return None
    
    def save_adapter(self, name: str):
        """Guarda un adapter específico."""
        if name in self.adapters:
            path = os.path.join(self.adapters_dir, f"{name}.pt")
            self.adapters[name].save(path)
            self.index["adapters"][name]["path"] = path
            self._save_index()
    
    def set_active(self, name: str) -> bool:
        """Establece el adapter activo."""
        if name in self.adapters or name in self.index["adapters"]:
            self.active_adapter = name
            if name not in self.adapters:
                self.load_adapter(name)
            return True
        return False
    
    def get_active(self) -> Optional[LoRAAdapter]:
        """Retorna el adapter activo."""
        if self.active_adapter:
            return self.adapters.get(self.active_adapter)
        return None
    
    def list_adapters(self) -> List[str]:
        """Lista todos los adapters disponibles."""
        return list(self.index["adapters"].keys())


# =============================================================================
# PRUEBA RÁPIDA
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 TEST: LoRA Adapter")
    print("=" * 60)
    
    # Crear adapter
    adapter = LoRAAdapter(
        hidden_dim=512,
        num_layers=6,
        rank=8
    )
    
    # Test forward
    x = torch.randn(2, 10, 512)  # [batch, seq, hidden]
    delta = adapter.get_delta(0, 'query', x)
    print(f"\n✅ Delta shape: {delta.shape}")
    print(f"✅ Delta mean: {delta.mean().item():.6f}")
    print(f"✅ Delta std: {delta.std().item():.6f}")
    
    # Test save/load
    adapter.save("test_adapter.pt")
    loaded = LoRAAdapter.load("test_adapter.pt")
    print(f"\n✅ Adapter cargado correctamente")
    
    # Limpiar
    os.remove("test_adapter.pt")
    print("\n✅ Test completado!")
