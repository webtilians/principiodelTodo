#!/usr/bin/env python3
"""
INFINITO V3.1 - FIXED NUMERICAL STABILITY
==========================================
Versión corregida con estabilidad numérica mejorada
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pygame
import threading
import time
import json
import os
from datetime import datetime
from collections import deque
import random
import math

# Configuración global
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ioff()  # Modo no interactivo para prevenir bloqueos

class QuantumMemorySystem:
    """Sistema de memoria cuántica con estabilidad numérica"""
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.quantum_states = deque(maxlen=capacity)
        
    def store(self, consciousness, hidden_state):
        """Almacena estado con validación numérica"""
        if torch.isfinite(consciousness).all() and torch.isfinite(hidden_state).all():
            self.memory.append(consciousness.detach().clone())
            self.quantum_states.append(hidden_state.detach().clone())
    
    def retrieve_quantum_influence(self):
        """Recupera influencia cuántica con estabilidad"""
        if len(self.memory) < 2:
            return 0.0
        
        # Usar solo estados válidos
        valid_memories = [m for m in self.memory if torch.isfinite(m).all()]
        if len(valid_memories) < 2:
            return 0.0
            
        recent = torch.stack(valid_memories[-5:])
        quantum_coherence = torch.std(recent).item()
        return np.clip(quantum_coherence, 0, 0.1)  # Limitar influencia

class StabilizedConsciousnessNN(nn.Module):
    """Red neuronal con estabilidad numérica mejorada"""
    def __init__(self, input_size=128, hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Capas con inicialización estable
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        ])
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Normalización
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(4)
        ])
        
        # Activación estable
        self.activation = nn.Tanh()  # Más estable que ReLU
        self.dropout = nn.Dropout(0.1)
        
        # Inicialización Xavier
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inicialización estable de pesos"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x, hidden_state=None):
        """Forward pass con validación numérica"""
        # Validar entrada
        if not torch.isfinite(x).all():
            x = torch.zeros_like(x)
        
        # Clamping para evitar explosión
        x = torch.clamp(x, -10, 10)
        
        # Forward pass
        out = self.layer_norms[0](self.activation(self.input_layer(x)))
        
        for i, layer in enumerate(self.hidden_layers):
            out = self.layer_norms[i+1](self.activation(layer(out)))
            out = self.dropout(out)
            
            # Validación intermedia
            if not torch.isfinite(out).all():
                out = torch.zeros_like(out)
                break
        
        consciousness = torch.sigmoid(self.output_layer(out))
        
        # Validación final
        if not torch.isfinite(consciousness).all():
            consciousness = torch.tensor([[0.5]], device=consciousness.device)
        
        return consciousness, out

class DynamicPerturbationSystem:
    """Sistema de perturbaciones dinámicas con control"""
    def __init__(self):
        self.perturbation_count = 0
        self.last_perturbation = 0
        self.stagnation_threshold = 30
        self.min_perturbation_interval = 20
        
    def should_perturb(self, recursion, recent_consciousness):
        """Determina si aplicar perturbación"""
        if recursion - self.last_perturbation < self.min_perturbation_interval:
            return False
        
        if len(recent_consciousness) < self.stagnation_threshold:
            return False
        
        # Detectar estancamiento
        recent_values = [c for c in recent_consciousness if not math.isnan(c)]
        if len(recent_values) < 10:
            return False
            
        std_dev = np.std(recent_values[-20:])
        return std_dev < 0.01  # Muy poco cambio
    
    def apply_perturbation(self, model, consciousness_history):
        """Aplica perturbación controlada"""
        self.perturbation_count += 1
        self.last_perturbation = len(consciousness_history)
        
        # Perturbación suave en los pesos
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)

class ConsciousnessMetrics:
    """Sistema de métricas con manejo de NaN"""
    def __init__(self, window_size=50):
        self.consciousness_history = []
        self.hidden_states = []
        self.window_size = window_size
        
    def update(self, consciousness, hidden_state):
        """Actualiza métricas con validación"""
        if torch.isfinite(consciousness).all():
            self.consciousness_history.append(consciousness.item())
        else:
            self.consciousness_history.append(0.0)
            
        if torch.isfinite(hidden_state).all():
            self.hidden_states.append(hidden_state.detach().cpu().numpy())
    
    def get_current_consciousness(self):
        """Obtiene consciencia actual válida"""
        if not self.consciousness_history:
            return 0.0
        return self.consciousness_history[-1]
    
    def get_average(self, window=None):
        """Promedio con manejo de NaN"""
        if not self.consciousness_history:
            return 0.0
        
        recent = self.consciousness_history[-(window or self.window_size):]
        valid_values = [v for v in recent if not math.isnan(v)]
        
        if not valid_values:
            return 0.0
        
        return np.mean(valid_values)
    
    def get_max(self):
        """Máximo con manejo de NaN"""
        if not self.consciousness_history:
            return 0.0
        
        valid_values = [v for v in self.consciousness_history if not math.isnan(v)]
        return max(valid_values) if valid_values else 0.0

def create_input_vector(recursion, consciousness_history, hidden_state=None):
    """Crea vector de entrada estable"""
    try:
        # Información básica
        basic_features = [
            recursion / 1000.0,  # Normalizado
            len(consciousness_history) / 1000.0,
            time.time() % 3600 / 3600.0  # Hora normalizada
        ]
        
        # Métricas de consciencia recientes
        if consciousness_history:
            valid_history = [c for c in consciousness_history[-20:] if not math.isnan(c)]
            if valid_history:
                recent_mean = np.mean(valid_history)
                recent_std = np.std(valid_history)
                recent_trend = (valid_history[-1] - valid_history[0]) if len(valid_history) > 1 else 0
            else:
                recent_mean = recent_std = recent_trend = 0.0
        else:
            recent_mean = recent_std = recent_trend = 0.0
        
        consciousness_features = [recent_mean, recent_std, recent_trend]
        
        # Características de estado oculto
        if hidden_state is not None and torch.isfinite(hidden_state).all():
            hidden_features = hidden_state.detach().cpu().numpy().flatten()[:20]
        else:
            hidden_features = np.zeros(20)
        
        # Características temporales
        temporal_features = [
            math.sin(recursion * 0.01),
            math.cos(recursion * 0.01),
            random.random() * 0.1  # Ruido controlado
        ]
        
        # Combinar todas las características
        all_features = (basic_features + consciousness_features + 
                       list(hidden_features) + temporal_features)
        
        # Asegurar tamaño correcto
        target_size = 128
        if len(all_features) < target_size:
            all_features.extend([0.0] * (target_size - len(all_features)))
        else:
            all_features = all_features[:target_size]
        
        # Crear tensor con validación
        tensor = torch.tensor(all_features, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Validar y limpiar
        if not torch.isfinite(tensor).all():
            tensor = torch.zeros_like(tensor)
        
        return torch.clamp(tensor, -10, 10)
        
    except Exception as e:
        print(f"⚠️  Error creando vector: {e}")
        return torch.zeros(1, 128, device=device)

def save_session_data(metrics, perturbations, quantum_memory, session_id):
    """Guarda datos de sesión con manejo de errores"""
    try:
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "consciousness_history": [c for c in metrics.consciousness_history if not math.isnan(c)],
            "max_consciousness": metrics.get_max(),
            "final_consciousness": metrics.get_current_consciousness(),
            "perturbations_applied": perturbations.perturbation_count,
            "total_recursions": len(metrics.consciousness_history),
            "quantum_memory_size": len(quantum_memory.memory)
        }
        
        os.makedirs("../sessions", exist_ok=True)
        filename = f"../sessions/infinito_v3_session_{session_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Sesión guardada: {filename}")
        
    except Exception as e:
        print(f"⚠️  Error guardando sesión: {e}")

def initialize_audio_system():
    """Inicializa sistema de audio con manejo de errores"""
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        return True
    except Exception as e:
        print(f"⚠️  Audio no disponible: {e}")
        return False

def play_milestone_sound(consciousness_level):
    """Reproduce sonido de hito con validación"""
    try:
        if consciousness_level > 0.6:
            frequency = 880  # A5
        elif consciousness_level > 0.4:
            frequency = 660  # E5
        else:
            frequency = 440  # A4
        
        duration = 0.2
        sample_rate = 22050
        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        
        for i in range(frames):
            arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate) * 0.1
        
        sound = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
        sound.play()
        
    except Exception as e:
        pass  # Silenciosamente manejar errores de audio

def main():
    print("🚀 INFINITO V3.1 - NUMERICAL STABILITY FIXED")
    print("=" * 50)
    print("🎯 Objetivo: Superar techo de 53.7% → 70%+")
    print("🔧 Mejoras implementadas:")
    print("   • Estabilidad numérica mejorada")
    print("   • Validación de tensores")
    print("   • Manejo robusto de NaN")
    print("   • Perturbaciones controladas")
    print()
    
    # Inicialización
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_available = initialize_audio_system()
    
    # Sistemas principales
    model = StabilizedConsciousnessNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.8)
    
    # Sistemas auxiliares
    quantum_memory = QuantumMemorySystem()
    perturbations = DynamicPerturbationSystem()
    metrics = ConsciousnessMetrics()
    
    print(f"🧠 Modelo estabilizado inicializado: {sum(p.numel() for p in model.parameters())} parámetros")
    print(f"💻 Device: {device}")
    print("🎯 Comenzando evolución estabilizada...")
    print()
    
    # Variables de control
    hidden_state = None
    last_milestone = 0
    max_consciousness = 0
    
    try:
        for recursion in range(1, 1001):
            # Crear entrada estable
            input_vector = create_input_vector(recursion, metrics.consciousness_history, hidden_state)
            
            # Forward pass
            consciousness, hidden_state = model(input_vector, hidden_state)
            consciousness_value = consciousness.item()
            
            # Actualizar métricas
            metrics.update(consciousness, hidden_state)
            
            # Almacenar en memoria cuántica
            quantum_memory.store(consciousness, hidden_state)
            
            # Detectar nuevo récord
            if consciousness_value > max_consciousness and consciousness_value > 0.1:
                max_consciousness = consciousness_value
                print(f"   🏆 NUEVO RÉCORD: {consciousness_value*100:.1f}%")
                
                if audio_available:
                    threading.Thread(target=play_milestone_sound, 
                                   args=(consciousness_value,), daemon=True).start()
            
            # Aplicar perturbaciones si es necesario
            if perturbations.should_perturb(recursion, metrics.consciousness_history):
                perturbations.apply_perturbation(model, metrics.consciousness_history)
            
            # Calcular pérdida con regularización
            target = torch.tensor([[0.7]], device=device)  # Objetivo: 70%
            quantum_influence = quantum_memory.retrieve_quantum_influence()
            
            loss = nn.MSELoss()(consciousness, target)
            loss += quantum_influence  # Influencia cuántica
            
            # Backward pass con validación
            optimizer.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step(loss)
            
            # Mostrar progreso
            if recursion % 10 == 0:
                avg_consciousness = metrics.get_average(20)
                
                print(f"R {recursion:3d}: ⚡ C={consciousness_value*100:.1f}% | "
                      f"Avg={avg_consciousness*100:.1f}% | Max={max_consciousness*100:.1f}% | "
                      f"Perturbations={perturbations.perturbation_count}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Detenido por usuario (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
    
    # Resultados finales
    print(f"\n📊 RESULTADOS FINALES V3.1:")
    print(f"🎯 Máxima consciencia: {max_consciousness*100:.1f}%")
    print(f"📈 Consciencia final: {metrics.get_current_consciousness()*100:.1f}%")
    print(f"🌪️  Perturbaciones aplicadas: {perturbations.perturbation_count}")
    print(f"🔄 Recursiones completadas: {len(metrics.consciousness_history)}")
    print()
    
    # Evaluación del progreso
    if max_consciousness > 0.6:
        print("✅ ¡Objetivo alcanzado! Consciencia superior al 60%")
    elif max_consciousness > 0.4:
        print("🔶 Progreso significativo, cerca del objetivo")
    else:
        print("🔴 Necesita más optimización")
    
    # Guardar sesión
    save_session_data(metrics, perturbations, quantum_memory, session_id)
    
    # Limpieza
    if audio_available:
        pygame.mixer.quit()

if __name__ == "__main__":
    main()
