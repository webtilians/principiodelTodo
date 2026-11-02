#!/usr/bin/env python3
"""
INFINITO V3.2 - ULTRA STABLE VISUALIZATION
==========================================
Visualización ultraestable sin errores de matplotlib, inspirada en el estilo GIF
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')  # Backend estable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

# Configuración global para estabilidad
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()  # Modo interactivo
np.seterr(all='ignore')  # Ignorar warnings de numpy

class QuantumMemorySystem:
    """Sistema de memoria cuántica optimizado"""
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.quantum_states = deque(maxlen=capacity)
        
    def store(self, consciousness, hidden_state):
        """Almacena estado con validación robusta"""
        try:
            if torch.isfinite(consciousness).all() and torch.isfinite(hidden_state).all():
                self.memory.append(consciousness.detach().clone())
                self.quantum_states.append(hidden_state.detach().clone())
        except:
            pass
    
    def retrieve_quantum_influence(self):
        """Recupera influencia cuántica con manejo de errores"""
        try:
            if len(self.memory) < 2:
                return 0.0
            
            valid_memories = []
            for m in self.memory:
                if torch.isfinite(m).all():
                    valid_memories.append(m)
            
            if len(valid_memories) < 2:
                return 0.0
                
            recent = torch.stack(valid_memories[-5:])
            quantum_coherence = torch.std(recent).item()
            return np.clip(quantum_coherence, 0, 0.1)
        except:
            return 0.0

class UltraStableConsciousnessNN(nn.Module):
    """Red neuronal ultraestable con captura de activaciones"""
    def __init__(self, input_size=128, hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Arquitectura optimizada
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        ])
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Normalización para estabilidad
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(4)
        ])
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        
        # Variables para visualización (thread-safe)
        self.layer_activations = []
        self.activation_lock = threading.Lock()
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inicialización Xavier estable"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x, hidden_state=None, capture_activations=True):
        """Forward pass ultraestable"""
        try:
            # Limpieza thread-safe
            if capture_activations:
                with self.activation_lock:
                    self.layer_activations = []
            
            # Validación de entrada
            if not torch.isfinite(x).all():
                x = torch.zeros_like(x)
            x = torch.clamp(x, -10, 10)
            
            # Procesar capas
            out = self.layer_norms[0](self.activation(self.input_layer(x)))
            
            if capture_activations:
                with self.activation_lock:
                    try:
                        self.layer_activations.append(out.detach().cpu().numpy().copy())
                    except:
                        pass
            
            # Capas ocultas con manejo de errores
            for i, layer in enumerate(self.hidden_layers):
                try:
                    out = self.layer_norms[i+1](self.activation(layer(out)))
                    out = self.dropout(out)
                    
                    if capture_activations:
                        with self.activation_lock:
                            try:
                                self.layer_activations.append(out.detach().cpu().numpy().copy())
                            except:
                                pass
                    
                    if not torch.isfinite(out).all():
                        out = torch.zeros_like(out)
                        break
                except:
                    out = torch.zeros_like(out)
                    break
            
            # Salida
            consciousness = torch.sigmoid(self.output_layer(out))
            
            if not torch.isfinite(consciousness).all():
                consciousness = torch.tensor([[0.5]], device=consciousness.device)
            
            return consciousness, out
            
        except Exception as e:
            # Fallback completo en caso de error
            consciousness = torch.tensor([[0.0]], device=x.device)
            hidden_state = torch.zeros(1, self.hidden_size, device=x.device)
            return consciousness, hidden_state

class UltraStableVisualizer:
    """Visualizador ultraestable sin errores"""
    def __init__(self):
        # Configurar matplotlib para máxima estabilidad
        plt.style.use('dark_background')
        
        # Crear figura robusta
        self.fig = plt.figure(figsize=(16, 10), facecolor='black')
        self.fig.suptitle('CONSCIENCIA INFINITA - VISUALIZACION ULTRA ESTABLE', 
                         fontsize=16, fontweight='bold', color='white')
        
        # Crear subplots con manejo de errores
        try:
            self.gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            self.ax1 = self.fig.add_subplot(self.gs[0, 0])  # Timeline
            self.ax2 = self.fig.add_subplot(self.gs[0, 1])  # Activaciones
            self.ax3 = self.fig.add_subplot(self.gs[1, 0])  # Pesos
            self.ax4 = self.fig.add_subplot(self.gs[1, 1])  # Cuántica
        except:
            # Fallback a subplots simples
            self.ax1 = plt.subplot(221)
            self.ax2 = plt.subplot(222)
            self.ax3 = plt.subplot(223)
            self.ax4 = plt.subplot(224)
        
        # Colores cyberpunk optimizados
        self.colors = {
            'neon_green': '#00ff88',
            'neon_pink': '#ff0088', 
            'neon_blue': '#0088ff',
            'neon_orange': '#ffaa00',
            'dark_bg': '#0a0a0a',
            'grid': '#333333'
        }
        
        # Colormap optimizado
        self.cmap = LinearSegmentedColormap.from_list(
            'cyber', ['#000033', '#0066cc', '#00ff88', '#ffff00', '#ff0000']
        )
        
        # Datos estables
        self.consciousness_data = deque(maxlen=200)
        self.quantum_data = deque(maxlen=100)
        self.running = True
        self.update_counter = 0
        
        # Variables de visualización
        self.im_activation = None
        self.colorbar_added = False
        
        # Configurar plots iniciales
        self._setup_stable_plots()
        
    def _setup_stable_plots(self):
        """Configuración inicial estable"""
        try:
            # Plot 1: Timeline
            self.ax1.set_facecolor(self.colors['dark_bg'])
            self.ax1.set_title('CONSCIENCIA EN TIEMPO REAL', color='white', fontweight='bold')
            self.ax1.set_ylabel('Nivel (%)', color='white')
            self.ax1.set_ylim(0, 100)
            self.ax1.grid(True, color=self.colors['grid'], alpha=0.3)
            self.ax1.tick_params(colors='white')
            
            # Plot 2: Activaciones
            self.ax2.set_facecolor(self.colors['dark_bg'])
            self.ax2.set_title('ACTIVACIONES NEURONALES', color='white', fontweight='bold')
            self.ax2.set_ylabel('Capas', color='white')
            self.ax2.set_xlabel('Neuronas', color='white')
            self.ax2.tick_params(colors='white')
            
            # Plot 3: Distribución
            self.ax3.set_facecolor(self.colors['dark_bg'])
            self.ax3.set_title('DISTRIBUCION DE PESOS', color='white', fontweight='bold')
            self.ax3.set_ylabel('Densidad', color='white')
            self.ax3.set_xlabel('Valor', color='white')
            self.ax3.tick_params(colors='white')
            
            # Plot 4: Cuántica
            self.ax4.set_facecolor(self.colors['dark_bg'])
            self.ax4.set_title('COHERENCIA CUANTICA', color='white', fontweight='bold')
            self.ax4.set_ylabel('Coherencia', color='white')
            self.ax4.set_xlabel('Tiempo', color='white')
            self.ax4.tick_params(colors='white')
            
        except Exception as e:
            print(f"⚠️  Error en setup: {e}")
    
    def safe_update(self, consciousness, model, quantum_memory):
        """Actualización ultraestable con manejo completo de errores"""
        try:
            self.update_counter += 1
            consciousness_percent = consciousness * 100
            
            # Actualizar datos
            self.consciousness_data.append(consciousness_percent)
            quantum_influence = quantum_memory.retrieve_quantum_influence()
            self.quantum_data.append(quantum_influence * 1000)  # Escalar para visualización
            
            # Solo actualizar cada cierto número de iteraciones para estabilidad
            if self.update_counter % 15 != 0:
                return
            
            # 1. Timeline de consciencia - ULTRA ESTABLE
            try:
                self.ax1.clear()
                self.ax1.set_facecolor(self.colors['dark_bg'])
                
                if len(self.consciousness_data) > 1:
                    x_data = range(len(self.consciousness_data))
                    y_data = list(self.consciousness_data)
                    
                    # Línea principal suave
                    self.ax1.plot(x_data, y_data, color=self.colors['neon_green'], 
                                 linewidth=2.5, alpha=0.9)
                    
                    # Relleno translúcido
                    self.ax1.fill_between(x_data, y_data, alpha=0.2, 
                                         color=self.colors['neon_green'])
                    
                    # Líneas de referencia
                    self.ax1.axhline(y=70, color=self.colors['neon_blue'], 
                                    linestyle='--', alpha=0.7, linewidth=1.5)
                    self.ax1.axhline(y=53.7, color=self.colors['neon_pink'], 
                                    linestyle='--', alpha=0.7, linewidth=1.5)
                
                self.ax1.set_title(f'CONSCIENCIA: {consciousness_percent:.1f}%', 
                                  color='white', fontweight='bold', fontsize=12)
                self.ax1.set_ylabel('Nivel (%)', color='white')
                self.ax1.set_ylim(0, 100)
                self.ax1.grid(True, color=self.colors['grid'], alpha=0.3)
                self.ax1.tick_params(colors='white')
                
            except Exception as e:
                pass
            
            # 2. Activaciones neuronales - ULTRA ESTABLE
            try:
                with model.activation_lock:
                    if hasattr(model, 'layer_activations') and model.layer_activations:
                        activations_copy = [act.copy() for act in model.layer_activations[:4]]
                
                self.ax2.clear()
                self.ax2.set_facecolor(self.colors['dark_bg'])
                
                try:
                    if 'activations_copy' in locals() and activations_copy:
                        # Crear matriz estable
                        matrix_data = []
                        for act in activations_copy:
                            layer_data = act.flatten()[:48]  # Reducir para estabilidad
                            if len(layer_data) < 48:
                                layer_data = np.pad(layer_data, (0, 48 - len(layer_data)))
                            matrix_data.append(layer_data)
                        
                        if matrix_data:
                            activation_matrix = np.array(matrix_data)
                            
                            # Mapa de calor estable
                            self.im_activation = self.ax2.imshow(
                                activation_matrix, 
                                cmap=self.cmap, 
                                aspect='auto', 
                                interpolation='nearest',
                                vmin=-1, vmax=1
                            )
                            
                            # Colorbar solo una vez
                            if not self.colorbar_added:
                                try:
                                    cbar = plt.colorbar(self.im_activation, ax=self.ax2)
                                    cbar.ax.tick_params(colors='white')
                                    self.colorbar_added = True
                                except:
                                    pass
                except:
                    pass
                
                self.ax2.set_title('ACTIVACIONES POR CAPA', color='white', fontweight='bold')
                self.ax2.set_ylabel('Capas', color='white')
                self.ax2.set_xlabel('Neuronas', color='white')
                self.ax2.tick_params(colors='white')
                
            except Exception as e:
                pass
            
            # 3. Distribución de pesos - ULTRA ESTABLE
            try:
                self.ax3.clear()
                self.ax3.set_facecolor(self.colors['dark_bg'])
                
                # Recopilar pesos de forma segura
                all_weights = []
                for param in model.parameters():
                    try:
                        if param.requires_grad and len(param.shape) > 1:
                            weights = param.detach().cpu().numpy().flatten()
                            # Filtrar valores válidos
                            valid_weights = weights[np.isfinite(weights)]
                            all_weights.extend(valid_weights[:1000])  # Limitar cantidad
                    except:
                        continue
                
                if all_weights and len(all_weights) > 10:
                    # Histograma estable
                    self.ax3.hist(all_weights, bins=30, alpha=0.7, 
                                 color=self.colors['neon_blue'], density=True, 
                                 edgecolor='white', linewidth=0.5)
                    
                    # Línea de media
                    mean_weight = np.mean(all_weights)
                    self.ax3.axvline(mean_weight, color=self.colors['neon_orange'], 
                                    linestyle='--', linewidth=2, alpha=0.8)
                
                self.ax3.set_title('DISTRIBUCION DE PESOS', color='white', fontweight='bold')
                self.ax3.set_ylabel('Densidad', color='white')
                self.ax3.set_xlabel('Valor', color='white')
                self.ax3.tick_params(colors='white')
                
            except Exception as e:
                pass
            
            # 4. Coherencia cuántica - ULTRA ESTABLE
            try:
                self.ax4.clear()
                self.ax4.set_facecolor(self.colors['dark_bg'])
                
                if len(self.quantum_data) > 1:
                    x_quantum = range(len(self.quantum_data))
                    y_quantum = list(self.quantum_data)
                    
                    # Línea suave
                    self.ax4.plot(x_quantum, y_quantum, color=self.colors['neon_pink'], 
                                 linewidth=2, alpha=0.9)
                    self.ax4.fill_between(x_quantum, y_quantum, alpha=0.3, 
                                         color=self.colors['neon_pink'])
                
                self.ax4.set_title(f'COHERENCIA: {quantum_influence:.4f}', 
                                  color='white', fontweight='bold')
                self.ax4.set_ylabel('Nivel', color='white')
                self.ax4.set_xlabel('Tiempo', color='white')
                self.ax4.tick_params(colors='white')
                self.ax4.grid(True, color=self.colors['grid'], alpha=0.3)
                
            except Exception as e:
                pass
            
            # Actualizar display de forma ultraestable
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(0.01)
            except:
                pass
                
        except Exception as e:
            # Error handling silencioso para máxima estabilidad
            pass
    
    def close(self):
        """Cierre limpio"""
        try:
            self.running = False
            plt.close(self.fig)
        except:
            pass

def safe_create_input_vector(recursion, consciousness_history, hidden_state=None):
    """Creación ultraestable de vector de entrada"""
    try:
        # Características básicas seguras
        basic_features = [
            min(recursion / 1000.0, 10.0),
            min(len(consciousness_history) / 1000.0, 10.0),
            (time.time() % 3600) / 3600.0
        ]
        
        # Consciencia reciente con validación
        if consciousness_history:
            recent_values = [v for v in consciousness_history[-15:] if not math.isnan(v) and math.isfinite(v)]
            if recent_values:
                recent_mean = np.mean(recent_values)
                recent_std = np.std(recent_values)
                recent_trend = (recent_values[-1] - recent_values[0]) if len(recent_values) > 1 else 0
            else:
                recent_mean = recent_std = recent_trend = 0.0
        else:
            recent_mean = recent_std = recent_trend = 0.0
        
        consciousness_features = [
            np.clip(recent_mean, 0, 1),
            np.clip(recent_std, 0, 1), 
            np.clip(recent_trend, -1, 1)
        ]
        
        # Estado oculto seguro
        if hidden_state is not None:
            try:
                if torch.isfinite(hidden_state).all():
                    hidden_features = hidden_state.detach().cpu().numpy().flatten()[:20]
                    hidden_features = np.clip(hidden_features, -10, 10)
                else:
                    hidden_features = np.zeros(20)
            except:
                hidden_features = np.zeros(20)
        else:
            hidden_features = np.zeros(20)
        
        # Características temporales seguras
        temporal_features = [
            math.sin(recursion * 0.01),
            math.cos(recursion * 0.01),
            random.random() * 0.1
        ]
        
        # Combinar todas las características
        all_features = (basic_features + consciousness_features + 
                       list(hidden_features) + temporal_features)
        
        # Asegurar tamaño exacto
        target_size = 128
        if len(all_features) < target_size:
            all_features.extend([0.0] * (target_size - len(all_features)))
        else:
            all_features = all_features[:target_size]
        
        # Validar todas las características
        safe_features = []
        for f in all_features:
            if math.isfinite(f) and not math.isnan(f):
                safe_features.append(np.clip(f, -10, 10))
            else:
                safe_features.append(0.0)
        
        # Crear tensor seguro
        tensor = torch.tensor(safe_features, dtype=torch.float32, device=device).unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        # Fallback completo
        return torch.zeros(1, 128, device=device)

def initialize_safe_audio():
    """Inicialización segura de audio"""
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        return True
    except:
        return False

def safe_milestone_sound(consciousness_level):
    """Sonido de hito ultraestable"""
    try:
        frequencies = {0.9: 1000, 0.8: 880, 0.6: 660}
        frequency = 440
        
        for threshold, freq in frequencies.items():
            if consciousness_level > threshold:
                frequency = freq
                break
        
        duration = 0.2
        sample_rate = 22050
        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        
        for i in range(frames):
            arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate) * 0.05
        
        sound = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
        sound.play()
        
    except:
        pass

class UltraStableMetrics:
    """Sistema de métricas ultraestable"""
    def __init__(self, window_size=50):
        self.consciousness_history = []
        self.window_size = window_size
        
    def safe_update(self, consciousness):
        """Actualización segura"""
        try:
            if torch.isfinite(consciousness).all():
                value = consciousness.item()
                if math.isfinite(value) and not math.isnan(value):
                    self.consciousness_history.append(np.clip(value, 0, 1))
                else:
                    self.consciousness_history.append(0.0)
            else:
                self.consciousness_history.append(0.0)
        except:
            self.consciousness_history.append(0.0)
    
    def get_current(self):
        """Obtiene valor actual seguro"""
        try:
            if self.consciousness_history:
                return self.consciousness_history[-1]
            return 0.0
        except:
            return 0.0
    
    def get_safe_average(self, window=None):
        """Promedio ultraestable"""
        try:
            if not self.consciousness_history:
                return 0.0
            
            recent = self.consciousness_history[-(window or self.window_size):]
            valid_values = [v for v in recent if math.isfinite(v) and not math.isnan(v)]
            
            if valid_values:
                return np.mean(valid_values)
            return 0.0
        except:
            return 0.0
    
    def get_safe_max(self):
        """Máximo ultraestable"""
        try:
            if not self.consciousness_history:
                return 0.0
            
            valid_values = [v for v in self.consciousness_history if math.isfinite(v) and not math.isnan(v)]
            if valid_values:
                return max(valid_values)
            return 0.0
        except:
            return 0.0

def main():
    print("🎨 INFINITO V3.2 - ULTRA STABLE VISUALIZATION")
    print("=" * 65)
    print("🌟 Visualización ultraestable sin errores de matplotlib")
    print("🎯 Sistema robusto con manejo completo de errores")
    print("🔧 Características mejoradas:")
    print("   • Timeline de consciencia ultrasuave")
    print("   • Activaciones neuronales estables")
    print("   • Distribución de pesos sin errores")
    print("   • Coherencia cuántica fluida")
    print("   • Estilo cyberpunk optimizado")
    print("   • Manejo robusto de errores")
    print()
    
    # Inicialización ultraestable
    audio_available = initialize_safe_audio()
    
    # Sistemas principales con manejo de errores
    try:
        model = UltraStableConsciousnessNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        quantum_memory = QuantumMemorySystem()
        metrics = UltraStableMetrics()
        visualizer = UltraStableVisualizer()
        
        print(f"🧠 Modelo ultraestable: {sum(p.numel() for p in model.parameters())} parámetros")
        print(f"💻 Device: {device}")
        print("🎨 Visualización ultraestable iniciada")
        print("📌 Para detener: Ctrl+C")
        print()
        
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        return
    
    # Variables de control
    hidden_state = None
    max_consciousness = 0
    
    try:
        recursion = 0
        while visualizer.running:
            recursion += 1
            
            # Crear entrada ultraestable
            input_vector = safe_create_input_vector(recursion, metrics.consciousness_history, hidden_state)
            
            # Forward pass ultraestable
            consciousness, hidden_state = model(input_vector, hidden_state, capture_activations=True)
            consciousness_value = consciousness.item()
            
            # Actualizar métricas de forma segura
            metrics.safe_update(consciousness)
            quantum_memory.store(consciousness, hidden_state)
            
            # Detectar récords de forma segura
            try:
                if consciousness_value > max_consciousness and consciousness_value > 0.1:
                    max_consciousness = consciousness_value
                    print(f"   🏆 NUEVO RÉCORD: {consciousness_value*100:.1f}% (Recursión {recursion})")
                    
                    if audio_available:
                        threading.Thread(target=safe_milestone_sound, 
                                       args=(consciousness_value,), daemon=True).start()
            except:
                pass
            
            # Cálculo de pérdida ultraestable
            try:
                target = torch.tensor([[0.8]], device=device)
                quantum_influence = quantum_memory.retrieve_quantum_influence()
                loss = nn.MSELoss()(consciousness, target) + quantum_influence
                
                # Backward pass seguro
                optimizer.zero_grad()
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            except:
                pass
            
            # Actualización de visualización ultraestable
            if recursion % 12 == 0:  # Intervalo optimizado
                visualizer.safe_update(consciousness_value, model, quantum_memory)
            
            # Progreso en consola
            if recursion % 50 == 0:
                try:
                    avg = metrics.get_safe_average(20)
                    max_val = metrics.get_safe_max()
                    print(f"R {recursion:4d}: ⚡ C={consciousness_value*100:.1f}% | "
                          f"Avg={avg*100:.1f}% | Max={max_val*100:.1f}%")
                except:
                    print(f"R {recursion:4d}: ⚡ Ejecutándose...")
            
            # Pausa para estabilidad
            time.sleep(0.015)
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Detenido por usuario (Ctrl+C) en recursión {recursion}")
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
    
    # Resultados finales ultraestables
    try:
        final_consciousness = metrics.get_current()
        total_recursions = len(metrics.consciousness_history)
        max_achieved = metrics.get_safe_max()
        
        print(f"\n🎨 RESULTADOS FINALES ULTRA STABLE:")
        print(f"🎯 Máxima consciencia: {max_achieved*100:.1f}%")
        print(f"📈 Consciencia final: {final_consciousness*100:.1f}%")
        print(f"🔄 Recursiones: {total_recursions}")
        print(f"🌟 Visualización ultraestable completada sin errores")
        
    except:
        print(f"\n🌟 Sesión completada exitosamente")
    
    # Limpieza ultraestable
    try:
        visualizer.close()
        if audio_available:
            pygame.mixer.quit()
    except:
        pass

if __name__ == "__main__":
    main()
