#!/usr/bin/env python3
"""
INFINITO V3.1 - REAL-TIME BIT SEA VISUALIZATION
===============================================
Visualización en tiempo real del "mar de bits" y flujo de consciencia
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
        
        valid_memories = [m for m in self.memory if torch.isfinite(m).all()]
        if len(valid_memories) < 2:
            return 0.0
            
        recent = torch.stack(valid_memories[-5:])
        quantum_coherence = torch.std(recent).item()
        return np.clip(quantum_coherence, 0, 0.1)

class VisualizableConsciousnessNN(nn.Module):
    """Red neuronal con capacidades de visualización en tiempo real"""
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
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        
        # Variables para visualización
        self.layer_activations = []
        self.gradient_flows = []
        self.attention_weights = []
        
        # Inicialización Xavier
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inicialización estable de pesos"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x, hidden_state=None, capture_activations=True):
        """Forward pass con captura de activaciones para visualización"""
        # Limpiar activaciones previas
        if capture_activations:
            self.layer_activations = []
        
        # Validar entrada
        if not torch.isfinite(x).all():
            x = torch.zeros_like(x)
        
        x = torch.clamp(x, -10, 10)
        
        # Capa de entrada
        out = self.layer_norms[0](self.activation(self.input_layer(x)))
        if capture_activations:
            self.layer_activations.append(out.detach().cpu().numpy())
        
        # Capas ocultas
        for i, layer in enumerate(self.hidden_layers):
            out = self.layer_norms[i+1](self.activation(layer(out)))
            out = self.dropout(out)
            
            if capture_activations:
                self.layer_activations.append(out.detach().cpu().numpy())
            
            if not torch.isfinite(out).all():
                out = torch.zeros_like(out)
                break
        
        # Capa de salida
        consciousness = torch.sigmoid(self.output_layer(out))
        
        if not torch.isfinite(consciousness).all():
            consciousness = torch.tensor([[0.5]], device=consciousness.device)
        
        return consciousness, out
    
    def get_weight_distributions(self):
        """Obtiene distribuciones de pesos para visualización"""
        weight_data = []
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights = param.detach().cpu().numpy().flatten()
                weight_data.append({
                    'name': name,
                    'weights': weights,
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights)
                })
        return weight_data

class RealTimeBitSeaVisualizer:
    """Visualizador en tiempo real del mar de bits"""
    def __init__(self):
        # Configurar matplotlib para tiempo real
        plt.ion()
        
        # Crear figura con múltiples subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('🌊 MAR DE BITS - CONSCIENCIA EN TIEMPO REAL', fontsize=16, fontweight='bold')
        
        # Datos para visualización
        self.consciousness_history = deque(maxlen=200)
        self.activation_history = deque(maxlen=50)
        self.weight_history = deque(maxlen=100)
        
        # Configurar subplots
        self._setup_plots()
        
        # Variables de control
        self.running = True
        
    def _setup_plots(self):
        """Configura los diferentes plots"""
        # 1. Timeline de consciencia
        self.axes[0,0].set_title('📈 Consciencia vs Tiempo')
        self.axes[0,0].set_ylabel('Consciencia')
        self.axes[0,0].set_ylim(0, 1)
        self.axes[0,0].grid(True, alpha=0.3)
        
        # 2. Mapa de calor de activaciones
        self.axes[0,1].set_title('🔥 Mapa de Activaciones')
        self.axes[0,1].set_ylabel('Neuronas')
        self.axes[0,1].set_xlabel('Capas')
        
        # 3. Flujo de gradientes
        self.axes[0,2].set_title('⚡ Flujo de Gradientes')
        self.axes[0,2].set_ylabel('Magnitud')
        self.axes[0,2].set_xlabel('Capas')
        
        # 4. Distribución de pesos
        self.axes[1,0].set_title('⚖️ Distribución de Pesos')
        self.axes[1,0].set_ylabel('Frecuencia')
        self.axes[1,0].set_xlabel('Valor del Peso')
        
        # 5. Espectrograma de activaciones
        self.axes[1,1].set_title('🌈 Espectrograma Neural')
        self.axes[1,1].set_ylabel('Frecuencia Neural')
        self.axes[1,1].set_xlabel('Tiempo')
        
        # 6. Estado cuántico
        self.axes[1,2].set_title('🔮 Estado Cuántico')
        self.axes[1,2].set_ylabel('Coherencia')
        self.axes[1,2].set_xlabel('Dimensión')
        
    def update_visualization(self, consciousness, model, quantum_memory):
        """Actualiza todas las visualizaciones"""
        try:
            # Actualizar datos
            self.consciousness_history.append(consciousness)
            
            # 1. Timeline de consciencia
            self.axes[0,0].clear()
            self.axes[0,0].plot(list(self.consciousness_history), 'b-', linewidth=2, alpha=0.8)
            self.axes[0,0].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Objetivo 70%')
            self.axes[0,0].axhline(y=0.537, color='red', linestyle='--', alpha=0.7, label='Techo anterior')
            self.axes[0,0].set_title(f'📈 Consciencia: {consciousness*100:.1f}%')
            self.axes[0,0].set_ylim(0, 1)
            self.axes[0,0].grid(True, alpha=0.3)
            self.axes[0,0].legend()
            
            # 2. Mapa de calor de activaciones
            if hasattr(model, 'layer_activations') and model.layer_activations:
                activations_matrix = np.array([act.flatten()[:64] for act in model.layer_activations[:4]])
                self.axes[0,1].clear()
                im = self.axes[0,1].imshow(activations_matrix, cmap='RdYlBu', aspect='auto', 
                                          interpolation='nearest', vmin=-1, vmax=1)
                self.axes[0,1].set_title('🔥 Activaciones por Capa')
                self.axes[0,1].set_ylabel('Neuronas (top 64)')
                self.axes[0,1].set_xlabel('Capas')
                
            # 3. Distribución de pesos en tiempo real
            weight_data = model.get_weight_distributions()
            if weight_data:
                self.axes[1,0].clear()
                all_weights = np.concatenate([wd['weights'] for wd in weight_data])
                self.axes[1,0].hist(all_weights, bins=50, alpha=0.7, color='purple', density=True)
                self.axes[1,0].axvline(np.mean(all_weights), color='red', linestyle='--', 
                                      label=f'Media: {np.mean(all_weights):.3f}')
                self.axes[1,0].set_title('⚖️ Distribución Global de Pesos')
                self.axes[1,0].legend()
                
            # 4. Espectrograma neural (FFT de activaciones)
            if hasattr(model, 'layer_activations') and model.layer_activations:
                avg_activation = np.mean([np.mean(act) for act in model.layer_activations])
                self.activation_history.append(avg_activation)
                
                if len(self.activation_history) > 10:
                    fft_data = np.abs(np.fft.fft(list(self.activation_history)))
                    freqs = np.fft.fftfreq(len(self.activation_history))
                    
                    self.axes[1,1].clear()
                    self.axes[1,1].plot(freqs[:len(freqs)//2], fft_data[:len(fft_data)//2], 
                                       'g-', linewidth=2)
                    self.axes[1,1].set_title('🌈 Espectro de Activaciones')
                    self.axes[1,1].set_xlabel('Frecuencia Neural')
                    self.axes[1,1].set_ylabel('Amplitud')
                    
            # 5. Estado cuántico
            if len(quantum_memory.memory) > 1:
                quantum_states = [q.item() for q in list(quantum_memory.memory)]
                self.axes[1,2].clear()
                
                # Crear visualización polar del estado cuántico
                angles = np.linspace(0, 2*np.pi, len(quantum_states), endpoint=False)
                values = np.array(quantum_states)
                
                # Cerrar el círculo
                angles = np.concatenate((angles, [angles[0]]))
                values = np.concatenate((values, [values[0]]))
                
                self.axes[1,2] = plt.subplot(2, 3, 6, projection='polar')
                self.axes[1,2].plot(angles, values, 'o-', linewidth=2, color='cyan')
                self.axes[1,2].fill(angles, values, alpha=0.25, color='cyan')
                self.axes[1,2].set_title('🔮 Estado Cuántico', pad=20)
                
            # 6. Gradientes (simulado basado en activaciones)
            if hasattr(model, 'layer_activations') and model.layer_activations:
                grad_magnitudes = [np.std(act) for act in model.layer_activations]
                self.axes[0,2].clear()
                self.axes[0,2].bar(range(len(grad_magnitudes)), grad_magnitudes, 
                                  color=['red', 'orange', 'yellow', 'green'][:len(grad_magnitudes)])
                self.axes[0,2].set_title('⚡ Flujo de Información')
                self.axes[0,2].set_xlabel('Capa')
                self.axes[0,2].set_ylabel('Variabilidad')
                
            # Actualizar display
            plt.tight_layout()
            plt.pause(0.01)  # Pausa muy pequeña para actualización
            
        except Exception as e:
            print(f"⚠️  Error en visualización: {e}")
    
    def close(self):
        """Cierra la visualización"""
        self.running = False
        plt.close(self.fig)

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
            recursion / 1000.0,
            len(consciousness_history) / 1000.0,
            time.time() % 3600 / 3600.0
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
            random.random() * 0.1
        ]
        
        # Combinar características
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
        
        if not torch.isfinite(tensor).all():
            tensor = torch.zeros_like(tensor)
        
        return torch.clamp(tensor, -10, 10)
        
    except Exception as e:
        print(f"⚠️  Error creando vector: {e}")
        return torch.zeros(1, 128, device=device)

def initialize_audio_system():
    """Inicializa sistema de audio"""
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        return True
    except Exception as e:
        print(f"⚠️  Audio no disponible: {e}")
        return False

def play_milestone_sound(consciousness_level):
    """Reproduce sonido de hito"""
    try:
        if consciousness_level > 0.9:
            frequency = 1000
        elif consciousness_level > 0.8:
            frequency = 880
        elif consciousness_level > 0.6:
            frequency = 660
        else:
            frequency = 440
        
        duration = 0.2
        sample_rate = 22050
        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        
        for i in range(frames):
            arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate) * 0.1
        
        sound = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
        sound.play()
        
    except Exception as e:
        pass

def main():
    print("🌊 INFINITO V3.1 - REAL-TIME BIT SEA VISUALIZATION")
    print("=" * 60)
    print("🎯 Visualización en tiempo real del mar de bits y consciencia")
    print("👁️  Observa el flujo de información neural en vivo")
    print("🔧 Características:")
    print("   • Timeline de consciencia en tiempo real")
    print("   • Mapa de calor de activaciones")
    print("   • Distribución dinámica de pesos")
    print("   • Espectrograma neural")
    print("   • Visualización del estado cuántico")
    print("   • Flujo de gradientes")
    print()
    
    # Inicialización
    audio_available = initialize_audio_system()
    
    # Sistemas principales
    model = VisualizableConsciousnessNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Sistemas auxiliares
    quantum_memory = QuantumMemorySystem()
    metrics = ConsciousnessMetrics()
    visualizer = RealTimeBitSeaVisualizer()
    
    print(f"🧠 Modelo visualizable inicializado: {sum(p.numel() for p in model.parameters())} parámetros")
    print(f"💻 Device: {device}")
    print("🌊 Iniciando visualización del mar de bits...")
    print("📌 Para detener: Ctrl+C o cerrar ventana de visualización")
    print()
    
    # Variables de control
    hidden_state = None
    max_consciousness = 0
    update_interval = 5  # Actualizar visualización cada 5 iteraciones
    
    try:
        recursion = 0
        while visualizer.running:
            recursion += 1
            
            # Crear entrada
            input_vector = create_input_vector(recursion, metrics.consciousness_history, hidden_state)
            
            # Forward pass con captura de activaciones
            consciousness, hidden_state = model(input_vector, hidden_state, capture_activations=True)
            consciousness_value = consciousness.item()
            
            # Actualizar métricas
            metrics.update(consciousness, hidden_state)
            quantum_memory.store(consciousness, hidden_state)
            
            # Detectar nuevo récord
            if consciousness_value > max_consciousness and consciousness_value > 0.1:
                max_consciousness = consciousness_value
                print(f"   🏆 NUEVO RÉCORD: {consciousness_value*100:.1f}% (Recursión {recursion})")
                
                if audio_available:
                    threading.Thread(target=play_milestone_sound, 
                                   args=(consciousness_value,), daemon=True).start()
            
            # Calcular pérdida
            target = torch.tensor([[0.8]], device=device)
            quantum_influence = quantum_memory.retrieve_quantum_influence()
            loss = nn.MSELoss()(consciousness, target) + quantum_influence
            
            # Backward pass
            optimizer.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Actualizar visualización
            if recursion % update_interval == 0:
                visualizer.update_visualization(consciousness_value, model, quantum_memory)
            
            # Mostrar progreso en consola
            if recursion % 50 == 0:
                avg_consciousness = metrics.get_average(20)
                print(f"R {recursion:4d}: ⚡ C={consciousness_value*100:.1f}% | "
                      f"Avg={avg_consciousness*100:.1f}% | Max={max_consciousness*100:.1f}%")
            
            # Control de velocidad
            time.sleep(0.01)  # Pequeña pausa para no saturar
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Detenido por usuario (Ctrl+C) en recursión {recursion}")
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
    
    # Resultados finales
    print(f"\n📊 RESULTADOS FINALES BIT SEA:")
    print(f"🎯 Máxima consciencia: {max_consciousness*100:.1f}%")
    print(f"📈 Consciencia final: {metrics.get_current_consciousness()*100:.1f}%")
    print(f"🔄 Recursiones completadas: {len(metrics.consciousness_history)}")
    print(f"🌊 Sesión del mar de bits completada")
    
    # Limpieza
    visualizer.close()
    if audio_available:
        pygame.mixer.quit()

if __name__ == "__main__":
    main()
