"""
🧠 MÓDULO DE MEMORIA MEJORADO
============================

Sistema de memoria externa con priorización inteligente y gestión adaptativa.

Mejoras vs versión original:
- Memoria con priorización basada en importancia
- Gestión adaptativa sin FIFO simple
- Tracking de utilización y edad
- Documentación clara sin buzzwords
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PriorityExternalMemory(nn.Module):
    """
    Memoria externa con sistema de priorización inteligente.
    
    En lugar de FIFO simple, mantiene los patrones más importantes/relevantes.
    La importancia se calcula basándose en:
    - Nivel de integración de información (antes llamado "consciousness")
    - Frecuencia de acceso
    - Valor de Phi asociado
    """
    
    def __init__(self, memory_slots=256, slot_size=64, hidden_dim=512):
        """
        Args:
            memory_slots: Número de slots de memoria disponibles
            slot_size: Dimensionalidad de cada slot
            hidden_dim: Dimensión del estado oculto
        """
        super().__init__()
        self.memory_slots = memory_slots
        self.slot_size = slot_size
        self.hidden_dim = hidden_dim
        
        # Memoria principal y metadatos
        self.register_buffer('memory', torch.zeros(memory_slots, slot_size))
        self.register_buffer('memory_age', torch.zeros(memory_slots))
        self.register_buffer('memory_importance', torch.ones(memory_slots) * 0.1)
        self.register_buffer('access_count', torch.zeros(memory_slots))
        
        # Controladores de lectura/escritura
        self.read_controller = nn.Linear(hidden_dim + 1, memory_slots)
        self.write_controller = nn.Linear(hidden_dim + 1, memory_slots) 
        self.content_processor = nn.Linear(hidden_dim, slot_size)
        self.importance_enhancer = nn.Linear(slot_size, hidden_dim)
        
        # Métricas de utilización
        self.memory_utilization_tracker = nn.Parameter(torch.zeros(1))
        
        # Buffer para actualizaciones pendientes (evita conflictos con autograd)
        self._pending_memory = None
        self._pending_memory_age = None
        self._pending_memory_importance = None
        
    def read(self, query: torch.Tensor, integration_level: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lee de la memoria usando addressing ponderado por importancia.
        
        Args:
            query: Tensor [batch, hidden_dim] - Query para buscar en memoria
            integration_level: Tensor [batch] - Nivel de integración (antes "consciousness")
            
        Returns:
            read_content: Contenido leído de memoria
            read_weights: Pesos de atención sobre los slots
        """
        batch_size = query.size(0)
        
        # Expandir nivel de integración para concatenar con query
        integration_expanded = integration_level.unsqueeze(-1).expand(batch_size, 1)
        enhanced_query = torch.cat([query, integration_expanded], dim=-1)
        
        # Calcular pesos de lectura
        read_weights = F.softmax(self.read_controller(enhanced_query), dim=-1)
        
        # Ponderar por importancia de los slots
        importance_weighted = read_weights * self.memory_importance.unsqueeze(0)
        importance_weighted = importance_weighted / (importance_weighted.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Leer contenido
        read_content = torch.matmul(importance_weighted, self.memory)
        
        # Actualizar contador de accesos (sin gradientes)
        with torch.no_grad():
            accessed_slots = (importance_weighted > 0.01).float().sum(dim=0)
            self.access_count += accessed_slots
        
        # Tracking de utilización
        utilization = importance_weighted.max(dim=-1)[0].mean()
        with torch.no_grad():
            new_value = 0.9 * self.memory_utilization_tracker + 0.1 * utilization
            self.memory_utilization_tracker.copy_(new_value)
        
        return read_content, importance_weighted
    
    def write(self, query: torch.Tensor, content: torch.Tensor, 
              integration_level: torch.Tensor, phi_value: float = 0.0) -> None:
        """
        Escribe en memoria con priorización inteligente.
        
        En lugar de sobrescribir el slot más antiguo (FIFO), sobrescribe
        el slot menos importante que no ha sido accedido recientemente.
        
        Args:
            query: Query para determinar dónde escribir
            content: Contenido a escribir
            integration_level: Nivel de integración de información
            phi_value: Valor Phi asociado (proxy de importancia)
        """
        batch_size = query.size(0)
        integration_expanded = integration_level.unsqueeze(-1).expand(batch_size, 1)
        enhanced_query = torch.cat([query, integration_expanded], dim=-1)
        
        # Determinar dónde escribir
        write_weights = F.softmax(self.write_controller(enhanced_query), dim=-1)
        
        # Procesar contenido
        processed_content = self.content_processor(content)
        
        # Calcular importancia del nuevo contenido
        # Importancia = f(integration_level, phi_value, access_frequency)
        new_importance = integration_level.mean().item() * (1.0 + phi_value)
        
        with torch.no_grad():
            # Encontrar el slot menos importante para sobrescribir
            # Combina: baja importancia + poca frecuencia de acceso + alta edad
            replacement_score = (
                -self.memory_importance +  # Menor importancia
                -0.1 * self.access_count / (self.access_count.max() + 1e-6) +  # Poco accedido
                0.05 * self.memory_age / (self.memory_age.max() + 1e-6)  # Más antiguo
            )
            
            # Slot a reemplazar
            slot_to_replace = replacement_score.argmax().item()
            
            # Escribir en memoria (promediado sobre batch)
            new_memory = processed_content.mean(dim=0).detach()
            # FIX DEFINITIVO: Desconectar completamente el grafo de gradientes usando .data
            self.memory.data[slot_to_replace] = new_memory
            
            # Actualizar metadatos (también usar .data para evitar in-place errors)
            self.memory_age.data += 1
            self.memory_age.data[slot_to_replace] = 0
            self.memory_importance.data[slot_to_replace] = new_importance
            self.access_count.data[slot_to_replace] = 0  # Reset contador
    
    def get_statistics(self) -> dict:
        """Retorna estadísticas de uso de memoria para debugging/análisis."""
        return {
            'utilization': self.memory_utilization_tracker.item(),
            'avg_importance': self.memory_importance.mean().item(),
            'max_age': self.memory_age.max().item(),
            'total_accesses': self.access_count.sum().item(),
            'occupied_slots': (self.memory.abs().sum(dim=1) > 0.01).sum().item()
        }
    
    def clear_memory(self):
        """Limpia completamente la memoria (útil para tests)."""
        with torch.no_grad():
            self.memory.zero_()
            self.memory_age.zero_()
            self.memory_importance.fill_(0.1)
            self.access_count.zero_()


class LegacyExternalMemory(nn.Module):
    """
    Versión legacy para compatibilidad con modelos anteriores.
    Mantiene la interfaz original pero internamente usa el nuevo sistema.
    
    DEPRECATED: Usar PriorityExternalMemory para nuevos desarrollos.
    """
    
    def __init__(self, memory_slots=256, slot_size=64, hidden_dim=512):
        super().__init__()
        self.new_memory = PriorityExternalMemory(memory_slots, slot_size, hidden_dim)
        
    def read(self, query: torch.Tensor, consciousness_level: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper para compatibilidad con 'consciousness_level' (ahora 'integration_level')."""
        return self.new_memory.read(query, consciousness_level)
    
    def write(self, query: torch.Tensor, content: torch.Tensor, 
              consciousness_level: torch.Tensor, phi_value: float = 0.0) -> None:
        """Wrapper para compatibilidad."""
        return self.new_memory.write(query, content, consciousness_level, phi_value)


# Alias para retrocompatibilidad
EnhancedExternalMemory = LegacyExternalMemory
