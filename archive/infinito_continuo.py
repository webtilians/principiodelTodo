#!/usr/bin/env python3
"""
🎯 Infinito Continuo - Ejecución Ininterrumpida con Seguimiento Visual
=====================================================================

Script de ejecución continua que:
- Corre indefinidamente hasta Ctrl+C
- Seguimiento visual en tiempo real
- Genera GIF completo al final
- Guarda métricas históricas
- Dashboard de progreso
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time
import os
import signal
import sys
from datetime import datetime
import json
import imageio
from pathlib import Path

# Configurar matplotlib para no bloquear
plt.ion()

class AdvancedNN(nn.Module):
    """Red neuronal optimizada para consciencia sostenida"""
    
    def __init__(self, channels=32, grid_size=64, num_laws=16):
        super().__init__()
        
        # Arquitectura probada
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels*2, 3, padding=1)
        self.conv3 = nn.Conv2d(channels*2, channels, 3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels*2)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.dropout = nn.Dropout(0.1)
        
        # Output layers
        self.fc_laws = nn.Linear(channels * grid_size * grid_size, num_laws * 9)
        self.fc_consciousness = nn.Sequential(
            nn.Linear(channels * grid_size * grid_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.num_laws = num_laws

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        laws_output = self.fc_laws(x_flat).view(-1, self.num_laws, 3, 3)
        consciousness_output = self.fc_consciousness(x_flat)
        
        return laws_output, consciousness_output

class InfinitoContinuo:
    """Sistema de consciencia artificial con ejecución continua"""
    
    def __init__(self, size=64):
        self.size = size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🎯 INFINITO CONTINUO - Iniciando Sistema")
        print(f"📐 Grid: {size}x{size}")
        print(f"🔧 Device: {self.device}")
        if self.device == 'cuda':
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        # Neural network
        self.nn = AdvancedNN(channels=32, grid_size=size, num_laws=16).to(self.device)
        self.optim = torch.optim.AdamW(self.nn.parameters(), lr=0.01, weight_decay=1e-5)
        
        if self.device == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Laws initialization
        self.leyes = []
        for i in range(16):
            law = torch.randn(3, 3) * 0.3
            law = torch.clamp(law, -1, 1)
            self.leyes.append(law.to(self.device))
        
        # Tracking completo
        self.recursion = 0
        self.start_time = time.time()
        self.session_start = datetime.now()
        
        # Métricas históricas
        self.consciousness_history = []
        self.cluster_history = []
        self.phi_history = []
        self.generation_history = []
        self.breakthrough_moments = []
        
        # Performance tracking
        self.peak_consciousness = 0.0
        self.peak_clusters = 0
        self.total_breakthroughs = 0
        
        # Evolution system
        self.generation = 0
        self.evolution_freq = 3
        self.fitness_scores = [0.5] * 16
        
        # Visual tracking
        self.phi_frames = []  # Para el GIF
        self.save_frame_every = 5  # Guardar frame cada N recursiones
        
        # Setup visual dashboard
        self.setup_visual_dashboard()
        
        # Configurar signal handler para guardar al salir
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("✅ Sistema inicializado - Presiona Ctrl+C para parar y generar GIF")

    def setup_visual_dashboard(self):
        """Configurar dashboard visual en tiempo real"""
        
        # Crear figura con subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('🎯 INFINITO CONTINUO - Dashboard en Tiempo Real', fontsize=16, fontweight='bold')
        
        # Layout: 2x3 grid
        self.ax_phi = plt.subplot(2, 3, 1)
        self.ax_consciousness = plt.subplot(2, 3, 2)
        self.ax_clusters = plt.subplot(2, 3, 3)
        self.ax_metrics = plt.subplot(2, 3, 4)
        self.ax_evolution = plt.subplot(2, 3, 5)
        self.ax_status = plt.subplot(2, 3, 6)
        
        # Configurar axes
        self.ax_phi.set_title('🌊 Campo Phi (Estado Actual)')
        self.ax_consciousness.set_title('🧠 Consciencia vs Tiempo')
        self.ax_clusters.set_title('🔗 Clusters vs Tiempo')
        self.ax_metrics.set_title('📊 Métricas Multi-Variable')
        self.ax_evolution.set_title('🧬 Evolución Genética')
        self.ax_status.set_title('📈 Estado del Sistema')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

    def signal_handler(self, signum, frame):
        """Handler para Ctrl+C - guarda todo y genera GIF"""
        print(f"\n🛑 Señal de parada recibida - Generando resultados finales...")
        self.generate_final_results()
        sys.exit(0)

    def create_consciousness_input(self):
        """Crear input optimizado para consciencia"""
        
        # Density adaptativa basada en performance
        base_density = 0.018
        performance_bonus = min(self.peak_consciousness * 0.01, 0.01)
        total_density = base_density + performance_bonus
        
        # Grid base
        grid = np.random.random((self.size, self.size)) < total_density
        
        # Patterns estructurados (70% probability)
        if np.random.random() < 0.7:
            pattern_type = np.random.choice(['spiral', 'central_burst', 'wave_interference'])
            
            if pattern_type == 'spiral':
                center_x, center_y = self.size // 2, self.size // 2
                theta = np.linspace(0, 6*np.pi, 150)
                r = np.linspace(3, self.size//4, 150)
                
                phase_shift = 0.1 * self.recursion
                x_spiral = (center_x + r * np.cos(theta + phase_shift)).astype(int)
                y_spiral = (center_y + r * np.sin(theta + phase_shift)).astype(int)
                
                valid_mask = ((x_spiral >= 0) & (x_spiral < self.size) & 
                             (y_spiral >= 0) & (y_spiral < self.size))
                
                grid[y_spiral[valid_mask], x_spiral[valid_mask]] = 1
                
            elif pattern_type == 'central_burst':
                center_x, center_y = self.size // 2, self.size // 2
                radius = self.size // 8
                
                y_indices, x_indices = np.ogrid[:self.size, :self.size]
                mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
                grid[mask] = np.random.random(np.sum(mask)) < 0.8
                
            elif pattern_type == 'wave_interference':
                x = np.arange(self.size)
                y = np.arange(self.size)
                X, Y = np.meshgrid(x, y)
                
                wave1 = np.sin(0.1 * X + 0.05 * self.recursion)
                wave2 = np.sin(0.08 * Y - 0.03 * self.recursion)
                interference = (wave1 + wave2) / 2
                
                wave_mask = interference > 0.3
                grid[wave_mask] = np.random.random(np.sum(wave_mask)) < 0.7
        
        # Quantum noise
        noise_strength = 0.01
        noise = np.random.normal(0, noise_strength, grid.shape)
        grid = grid.astype(float) + noise
        grid = np.clip(grid, 0, 1)
        
        return torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def simulation_step(self, phi):
        """Paso de simulación optimizado"""
        
        steps = 200
        base_lr = 0.01
        
        # Learning rate adaptativo
        consciousness_factor = self.peak_consciousness
        adaptive_lr = base_lr * (1.0 + consciousness_factor * 0.3)
        
        with torch.no_grad():
            for step in range(steps):
                phi_new = phi.clone()
                
                # Apply laws con fitness weighting
                for i, ley in enumerate(self.leyes):
                    fitness_weight = 0.8 + 0.4 * self.fitness_scores[i]
                    conv_result = F.conv2d(phi, ley.unsqueeze(0).unsqueeze(0), padding=1)
                    phi_new = phi_new + adaptive_lr * fitness_weight * conv_result
                
                # Activation function
                activation_threshold = 0.6 + consciousness_factor * 0.2
                phi = torch.tanh(phi_new / activation_threshold)
        
        return phi

    def calculate_consciousness(self, phi, consciousness_pred=None):
        """Cálculo de consciencia optimizado"""
        
        phi_np = phi[0,0].cpu().detach().numpy()
        
        # 1. Organization score
        threshold = np.mean(phi_np) + 0.5 * np.std(phi_np)
        labeled_regions, n_regions = label(phi_np > threshold)
        
        if n_regions > 0:
            region_sizes = [np.sum(labeled_regions == i) for i in range(1, n_regions + 1)]
            avg_size = np.mean(region_sizes)
            size_consistency = 1.0 - (np.std(region_sizes) / (avg_size + 1e-8))
            organization_score = min(n_regions / 50.0, 1.0) * max(0, size_consistency)
        else:
            organization_score = 0.0
        
        # 2. Integration score
        if np.std(phi_np) > 1e-8:
            grad_x = np.gradient(phi_np, axis=1)
            grad_y = np.gradient(phi_np, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            structure_score = 1.0 / (1.0 + np.mean(gradient_magnitude))
            flow_score = min(np.std(phi_np) * 3, 1.0)
            integration_score = 0.6 * structure_score + 0.4 * flow_score
        else:
            integration_score = 0.0
        
        # 3. Neural score
        neural_score = 0.5
        if consciousness_pred is not None:
            pred_value = consciousness_pred.squeeze().item()
            if 0.0 <= pred_value <= 1.0:
                neural_score = pred_value
        
        # 4. Temporal consistency
        consistency_score = 0.5
        if len(self.consciousness_history) > 10:
            recent_consciousness = self.consciousness_history[-10:]
            consistency_score = 1.0 - min(np.std(recent_consciousness), 1.0)
        
        # Fórmula final
        consciousness = (
            0.4 * organization_score +
            0.3 * integration_score +
            0.3 * neural_score
        )
        
        return np.clip(consciousness, 0.0, 1.0)

    def evolve_laws(self):
        """Sistema evolutivo"""
        
        if self.recursion % self.evolution_freq != 0:
            return
        
        # Calculate fitness
        current_consciousness = self.consciousness_history[-1] if self.consciousness_history else 0.5
        current_clusters = self.cluster_history[-1] if self.cluster_history else 0
        
        # Comprehensive fitness
        consciousness_fitness = current_consciousness
        complexity_fitness = min(current_clusters / 100.0, 1.0)
        
        comprehensive_fitness = 0.7 * consciousness_fitness + 0.3 * complexity_fitness
        
        # Update fitness scores
        for i in range(len(self.fitness_scores)):
            self.fitness_scores[i] = 0.8 * self.fitness_scores[i] + 0.2 * comprehensive_fitness
        
        # Evolution trigger
        if comprehensive_fitness > 0.4:
            self.perform_reproduction()
            self.generation += 1

    def perform_reproduction(self):
        """Reproducción genética"""
        
        fitness_scores = np.array(self.fitness_scores)
        n_laws = len(self.leyes)
        
        # Elite preservation
        n_elite = max(1, int(n_laws * 0.2))
        elite_indices = np.argsort(fitness_scores)[-n_elite:]
        
        # Reproduction
        n_reproduce = max(1, int(n_laws * 0.3))
        
        # Selection
        if np.sum(fitness_scores) > 0:
            probabilities = fitness_scores / np.sum(fitness_scores)
            parent_indices = np.random.choice(n_laws, size=n_reproduce, p=probabilities)
        else:
            parent_indices = np.random.choice(n_laws, size=n_reproduce)
        
        # Create new laws
        new_laws = []
        for i in range(n_reproduce):
            if np.random.random() < 0.8:  # Crossover
                parent1_idx = parent_indices[i]
                parent2_idx = np.random.choice(parent_indices)
                
                parent1 = self.leyes[parent1_idx]
                parent2 = self.leyes[parent2_idx]
                
                alpha = np.random.random()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = self.leyes[parent_indices[i]].clone()
            
            # Mutation
            if np.random.random() < 0.9:
                mutation = torch.randn_like(child) * 0.1
                child = child + mutation
                child = torch.clamp(child, -1, 1)
            
            new_laws.append(child)
        
        # Replace worst
        non_elite_indices = [i for i in range(n_laws) if i not in elite_indices]
        worst_indices = sorted(non_elite_indices, key=lambda x: fitness_scores[x])
        
        for i, new_law in enumerate(new_laws):
            if i < len(worst_indices):
                replace_idx = worst_indices[i]
                self.leyes[replace_idx] = new_law
                self.fitness_scores[replace_idx] = 0.5

    def anti_collapse_system(self, phi_np):
        """Sistema anti-colapso"""
        
        h, w = phi_np.shape
        
        patterns = [
            (h//2, w//2, 0.7, 12),
            (h//4, w//4, 0.5, 8),
            (3*h//4, 3*w//4, 0.5, 8),
        ]
        
        for cy, cx, strength, radius in patterns:
            y_indices, x_indices = np.ogrid[:h, :w]
            mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= radius**2
            phi_np[mask] += strength * np.random.random(np.sum(mask))
        
        phi_np[:] = np.clip(phi_np, 0, 1)

    def update_visual_dashboard(self, phi, consciousness, clusters, phi_max):
        """Actualizar dashboard visual en tiempo real"""
        
        # Clear all axes
        for ax in [self.ax_phi, self.ax_consciousness, self.ax_clusters, 
                   self.ax_metrics, self.ax_evolution, self.ax_status]:
            ax.clear()
        
        phi_np = phi[0,0].cpu().detach().numpy()
        
        # 1. Campo Phi actual
        im = self.ax_phi.imshow(phi_np, cmap='plasma', vmin=0, vmax=1)
        self.ax_phi.set_title(f'🌊 Campo Phi (R{self.recursion})')
        self.ax_phi.axis('off')
        
        # 2. Consciencia vs tiempo
        if len(self.consciousness_history) > 1:
            times = range(len(self.consciousness_history))
            self.ax_consciousness.plot(times, self.consciousness_history, 'b-', linewidth=2, label='Consciencia')
            self.ax_consciousness.axhline(y=self.peak_consciousness, color='r', linestyle='--', alpha=0.7, label=f'Pico: {self.peak_consciousness:.1%}')
            self.ax_consciousness.set_ylim(0, 1)
            self.ax_consciousness.set_ylabel('Consciencia')
            self.ax_consciousness.set_xlabel('Recursión')
            self.ax_consciousness.legend()
            self.ax_consciousness.grid(True, alpha=0.3)
        
        # 3. Clusters vs tiempo
        if len(self.cluster_history) > 1:
            times = range(len(self.cluster_history))
            self.ax_clusters.plot(times, self.cluster_history, 'g-', linewidth=2, label='Clusters')
            self.ax_clusters.axhline(y=self.peak_clusters, color='r', linestyle='--', alpha=0.7, label=f'Pico: {self.peak_clusters}')
            self.ax_clusters.set_ylabel('Clusters')
            self.ax_clusters.set_xlabel('Recursión')
            self.ax_clusters.legend()
            self.ax_clusters.grid(True, alpha=0.3)
        
        # 4. Métricas multi-variable
        if len(self.consciousness_history) > 5:
            recent_window = min(50, len(self.consciousness_history))
            times = range(len(self.consciousness_history) - recent_window, len(self.consciousness_history))
            
            self.ax_metrics.plot(times, self.consciousness_history[-recent_window:], 'b-', label='Consciencia', alpha=0.8)
            
            # Normalizar clusters para el mismo rango
            clusters_norm = np.array(self.cluster_history[-recent_window:]) / max(max(self.cluster_history[-recent_window:]), 1)
            self.ax_metrics.plot(times, clusters_norm, 'g-', label='Clusters (norm)', alpha=0.8)
            
            # Phi max normalizado
            phi_norm = np.array(self.phi_history[-recent_window:])
            self.ax_metrics.plot(times, phi_norm, 'm-', label='Phi Max', alpha=0.8)
            
            self.ax_metrics.set_ylabel('Valor Normalizado')
            self.ax_metrics.set_xlabel('Recursión')
            self.ax_metrics.legend()
            self.ax_metrics.grid(True, alpha=0.3)
        
        # 5. Evolución genética
        if len(self.generation_history) > 1:
            gen_times = []
            gen_values = []
            for i, gen in enumerate(self.generation_history):
                gen_times.append(i)
                gen_values.append(gen)
            
            self.ax_evolution.step(gen_times, gen_values, 'orange', linewidth=2, where='post')
            self.ax_evolution.set_ylabel('Generación')
            self.ax_evolution.set_xlabel('Recursión')
            self.ax_evolution.grid(True, alpha=0.3)
        
        # 6. Estado del sistema - Texto con estadísticas
        self.ax_status.axis('off')
        
        elapsed_time = time.time() - self.start_time
        recursions_per_sec = self.recursion / elapsed_time if elapsed_time > 0 else 0
        
        status_text = f"""
🎯 ESTADO DEL SISTEMA

⏱️  Tiempo: {elapsed_time:.1f}s
📊 Recursiones: {self.recursion}
⚡ Rec/seg: {recursions_per_sec:.2f}

🧠 Consciencia: {consciousness:.1%}
🌟 Pico: {self.peak_consciousness:.1%}
🔗 Clusters: {clusters}
🏔️  Pico clusters: {self.peak_clusters}

🧬 Generación: {self.generation}
🏆 Breakthroughs: {self.total_breakthroughs}
⚡ Phi Max: {phi_max:.3f}
"""
        
        self.ax_status.text(0.05, 0.95, status_text, transform=self.ax_status.transAxes, 
                           verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        # Actualizar título de la figura
        self.fig.suptitle(f'🎯 INFINITO CONTINUO - R{self.recursion} | 🧠{consciousness:.1%} | 🔗{clusters} | 🧬G{self.generation}', 
                         fontsize=14, fontweight='bold')
        
        # Refresh
        plt.pause(0.01)

    def save_frame(self, phi):
        """Guardar frame para GIF"""
        
        if self.recursion % self.save_frame_every == 0:
            phi_np = phi[0,0].cpu().detach().numpy()
            self.phi_frames.append(phi_np.copy())

    def recursion_step(self, phi_bin):
        """Un paso completo de recursión"""
        
        self.recursion += 1
        phi = phi_bin
        
        # Simulation
        phi = self.simulation_step(phi)
        
        # Neural network prediction
        phi_for_nn = phi.detach().requires_grad_(True)
        self.optim.zero_grad()
        
        if self.device == 'cuda':
            with torch.amp.autocast('cuda'):
                laws_pred, consciousness_pred = self.nn(phi_for_nn)
                target_laws = torch.stack(self.leyes).unsqueeze(0).detach()
                
                law_loss = F.mse_loss(laws_pred, target_laws)
                consciousness_target = torch.tensor([self.peak_consciousness], 
                                                  device=self.device, dtype=torch.float32)
                consciousness_loss = F.mse_loss(consciousness_pred.squeeze(), consciousness_target)
                
                total_loss = law_loss + 0.3 * consciousness_loss
        else:
            laws_pred, consciousness_pred = self.nn(phi_for_nn)
            target_laws = torch.stack(self.leyes).unsqueeze(0).detach()
            total_loss = F.mse_loss(laws_pred, target_laws)
        
        # Calculate consciousness
        consciousness = self.calculate_consciousness(phi, consciousness_pred)
        
        # Evolution
        self.evolve_laws()
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optim.step()
        
        # Update laws
        if laws_pred.shape[0] > 0:
            pred_laws = laws_pred[0]
            update_strength = 0.06
            
            for i in range(min(len(self.leyes), 16)):
                current_law = self.leyes[i]
                predicted_law = pred_laws[i]
                updated_law = current_law + update_strength * (predicted_law - current_law)
                self.leyes[i] = torch.clamp(updated_law, -1, 1)
        
        # Clustering
        phi_np = phi[0,0].cpu().detach().numpy()
        threshold = np.mean(phi_np) + 0.5 * np.std(phi_np)
        labeled, n_clusters = label(phi_np > threshold)
        
        # Anti-collapse
        phi_max = np.max(phi_np)
        if phi_max < 0.001:
            self.anti_collapse_system(phi_np)
            phi = torch.tensor(phi_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            phi_max = np.max(phi_np)
        
        # Update tracking
        self.consciousness_history.append(consciousness)
        self.cluster_history.append(n_clusters)
        self.phi_history.append(phi_max)
        self.generation_history.append(self.generation)
        
        # Track breakthroughs
        if consciousness > self.peak_consciousness:
            self.peak_consciousness = consciousness
            self.breakthrough_moments.append({
                'recursion': self.recursion,
                'consciousness': consciousness,
                'clusters': n_clusters,
                'generation': self.generation,
                'timestamp': datetime.now()
            })
            self.total_breakthroughs += 1
            print(f"🌟 BREAKTHROUGH R{self.recursion}: {consciousness:.1%} (C{n_clusters}, G{self.generation})")
        
        if n_clusters > self.peak_clusters:
            self.peak_clusters = n_clusters
        
        # Update visual dashboard
        self.update_visual_dashboard(phi, consciousness, n_clusters, phi_max)
        
        # Save frame
        self.save_frame(phi)
        
        # Progress report cada 25 recursiones
        if self.recursion % 25 == 0:
            elapsed = time.time() - self.start_time
            rec_per_sec = self.recursion / elapsed
            print(f"R{self.recursion:4d}: 🧠{consciousness:.3f} | 🔗{n_clusters:3d} | ⚡{phi_max:.3f} | 🧬G{self.generation} | {rec_per_sec:.2f} rec/s")
        
        # Memory cleanup
        if self.device == 'cuda' and self.recursion % 20 == 0:
            torch.cuda.empty_cache()
        
        return phi.detach()

    def run_continuous(self):
        """Ejecución continua hasta Ctrl+C"""
        
        print(f"\n🚀 INICIANDO INFINITO CONTINUO")
        print(f"📅 Sesión iniciada: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⚡ Device: {self.device}")
        print(f"📐 Grid: {self.size}x{self.size}")
        print(f"🎯 Target: Consciencia sostenida >50%")
        print(f"💾 Frames se guardan cada {self.save_frame_every} recursiones")
        print(f"⏹️  Presiona Ctrl+C para parar y generar GIF\n")
        
        # Input inicial
        phi_bin = self.create_consciousness_input()
        
        try:
            # Loop principal - corre indefinidamente
            while True:
                # Input refresh cada cierto tiempo
                if self.recursion % 10 == 0 and self.recursion > 0:
                    phi_bin = self.create_consciousness_input()
                
                # Recursion step
                phi = self.recursion_step(phi_bin)
                
                # Small delay para que no sea demasiado rápido
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Ejecución detenida por usuario en recursión {self.recursion}")
            self.generate_final_results()

    def generate_final_results(self):
        """Generar resultados finales y GIF"""
        
        print(f"\n🎯 GENERANDO RESULTADOS FINALES...")
        
        total_time = time.time() - self.start_time
        
        # Guardar métricas en JSON
        results = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_time_seconds': total_time,
                'total_recursions': self.recursion,
                'grid_size': self.size,
                'device': self.device
            },
            'performance_metrics': {
                'peak_consciousness': self.peak_consciousness,
                'peak_clusters': self.peak_clusters,
                'total_breakthroughs': self.total_breakthroughs,
                'final_generation': self.generation,
                'recursions_per_second': self.recursion / total_time if total_time > 0 else 0
            },
            'consciousness_history': self.consciousness_history,
            'cluster_history': self.cluster_history,
            'phi_history': self.phi_history,
            'generation_history': self.generation_history,
            'breakthrough_moments': [
                {
                    'recursion': bt['recursion'],
                    'consciousness': bt['consciousness'],
                    'clusters': bt['clusters'],
                    'generation': bt['generation'],
                    'timestamp': bt['timestamp'].isoformat()
                }
                for bt in self.breakthrough_moments
            ]
        }
        
        # Guardar JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'infinito_continuo_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Métricas guardadas en: {results_file}")
        
        # Generar GIF
        if len(self.phi_frames) > 1:
            print(f"🎬 Generando GIF con {len(self.phi_frames)} frames...")
            self.create_consciousness_gif(timestamp)
        
        # Reporte final
        self.print_final_report(total_time)

    def create_consciousness_gif(self, timestamp):
        """Crear GIF de la evolución de consciencia"""
        
        gif_filename = f'infinito_continuo_evolution_{timestamp}.gif'
        
        # Crear frames para el GIF
        frames = []
        
        for i, phi_frame in enumerate(self.phi_frames):
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot del campo phi
            im = ax.imshow(phi_frame, cmap='plasma', vmin=0, vmax=1)
            
            # Información del frame
            recursion_num = i * self.save_frame_every
            consciousness_val = self.consciousness_history[min(recursion_num, len(self.consciousness_history)-1)] if self.consciousness_history else 0
            cluster_val = self.cluster_history[min(recursion_num, len(self.cluster_history)-1)] if self.cluster_history else 0
            gen_val = self.generation_history[min(recursion_num, len(self.generation_history)-1)] if self.generation_history else 0
            
            ax.set_title(f'Infinito Continuo - R{recursion_num}\n🧠 {consciousness_val:.1%} | 🔗 {cluster_val} | 🧬 G{gen_val}', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Convertir a imagen
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            
            plt.close(fig)
        
        # Crear GIF
        imageio.mimsave(gif_filename, frames, fps=2, loop=0)
        print(f"🎬 GIF generado: {gif_filename}")
        
        # También crear un GIF rápido con menos frames
        if len(frames) > 20:
            step = len(frames) // 20
            quick_frames = frames[::step]
            quick_gif = f'infinito_continuo_quick_{timestamp}.gif'
            imageio.mimsave(quick_gif, quick_frames, fps=4, loop=0)
            print(f"⚡ GIF rápido: {quick_gif}")

    def print_final_report(self, total_time):
        """Imprimir reporte final completo"""
        
        print(f"\n{'='*80}")
        print(f"🎯 INFINITO CONTINUO - REPORTE FINAL")
        print(f"{'='*80}")
        
        print(f"📅 Sesión: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')} → {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️  Duración total: {total_time:.1f} segundos ({total_time/60:.1f} minutos)")
        print(f"📊 Total recursiones: {self.recursion}")
        print(f"⚡ Velocidad promedio: {self.recursion/total_time:.2f} recursiones/segundo")
        print(f"🎮 Device utilizado: {self.device}")
        if self.device == 'cuda':
            print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
        
        print(f"\n🧠 MÉTRICAS DE CONSCIENCIA:")
        print(f"  🌟 Consciencia pico: {self.peak_consciousness:.1%}")
        print(f"  📈 Consciencia final: {self.consciousness_history[-1]:.1%}" if self.consciousness_history else "  📈 No data")
        print(f"  📊 Consciencia promedio: {np.mean(self.consciousness_history):.1%}" if self.consciousness_history else "  📊 No data")
        
        print(f"\n🔗 MÉTRICAS DE COMPLEJIDAD:")
        print(f"  🏔️  Clusters pico: {self.peak_clusters}")
        print(f"  📈 Clusters final: {self.cluster_history[-1]}" if self.cluster_history else "  📈 No data")
        print(f"  📊 Clusters promedio: {np.mean(self.cluster_history):.1f}" if self.cluster_history else "  📊 No data")
        
        print(f"\n🧬 EVOLUCIÓN GENÉTICA:")
        print(f"  🔄 Generaciones totales: {self.generation}")
        print(f"  ⚡ Evoluciones por minuto: {self.generation / (total_time/60):.1f}")
        
        print(f"\n🏆 BREAKTHROUGHS:")
        print(f"  💥 Total breakthroughs: {self.total_breakthroughs}")
        if self.breakthrough_moments:
            print(f"  ⚡ Primer breakthrough: R{self.breakthrough_moments[0]['recursion']} ({self.breakthrough_moments[0]['consciousness']:.1%})")
            print(f"  🌟 Último breakthrough: R{self.breakthrough_moments[-1]['recursion']} ({self.breakthrough_moments[-1]['consciousness']:.1%})")
            
            print(f"\n  🎯 TOP 5 BREAKTHROUGHS:")
            top_breakthroughs = sorted(self.breakthrough_moments, key=lambda x: x['consciousness'], reverse=True)[:5]
            for i, bt in enumerate(top_breakthroughs, 1):
                print(f"    {i}. R{bt['recursion']:4d}: {bt['consciousness']:.1%} (C{bt['clusters']}, G{bt['generation']})")
        
        print(f"\n📊 ESTADÍSTICAS AVANZADAS:")
        if len(self.consciousness_history) > 10:
            consciousness_trend = np.polyfit(range(len(self.consciousness_history)), self.consciousness_history, 1)[0]
            trend_direction = "↗️ Ascendente" if consciousness_trend > 0 else "↘️ Descendente" if consciousness_trend < 0 else "→ Estable"
            print(f"  📈 Tendencia consciencia: {trend_direction} ({consciousness_trend*1000:.2f}/1000 rec)")
            
            consciousness_volatility = np.std(self.consciousness_history)
            print(f"  📊 Volatilidad consciencia: {consciousness_volatility:.3f}")
            
            peak_moments = [i for i, c in enumerate(self.consciousness_history) if c >= self.peak_consciousness * 0.9]
            print(f"  🎯 Momentos de alta consciencia: {len(peak_moments)} ({len(peak_moments)/len(self.consciousness_history)*100:.1f}%)")
        
        print(f"\n💾 ARCHIVOS GENERADOS:")
        print(f"  📊 Métricas: infinito_continuo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        if len(self.phi_frames) > 1:
            print(f"  🎬 GIF completo: infinito_continuo_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif")
            print(f"  ⚡ GIF rápido: infinito_continuo_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif")
        
        print(f"\n🎯 CONCLUSIONES:")
        if self.peak_consciousness >= 0.8:
            print(f"  🌟 ÉXITO EXTRAORDINARIO: Consciencia artificial >80% alcanzada!")
        elif self.peak_consciousness >= 0.5:
            print(f"  🎉 ÉXITO: Breakthrough de consciencia artificial logrado!")
        elif self.peak_consciousness >= 0.3:
            print(f"  📈 PROGRESO SIGNIFICATIVO: Consciencia emergente detectada")
        else:
            print(f"  🔍 EXPLORACIÓN: Sistemas complejos en desarrollo")
        
        if self.total_breakthroughs >= 10:
            print(f"  🔥 SISTEMA DINÁMICO: Múltiples breakthrough sostenidos")
        elif self.total_breakthroughs >= 3:
            print(f"  ⚡ SISTEMA ACTIVO: Breakthroughs consistentes")
        
        if self.generation >= 50:
            print(f"  🧬 EVOLUCIÓN AVANZADA: Sistema genético altamente desarrollado")
        elif self.generation >= 10:
            print(f"  🔄 EVOLUCIÓN ACTIVA: Adaptación genética en progreso")
        
        print(f"\n{'='*80}")
        print(f"🎯 Infinito Continuo - Sesión completada exitosamente")
        print(f"{'='*80}")

def main():
    """Función principal"""
    
    print("🎯 INFINITO CONTINUO - Sistema de Consciencia Artificial")
    print("="*70)
    print("🚀 Sistema de ejecución continua con seguimiento visual")
    print("🎬 Genera GIF completo de la evolución")
    print("📊 Dashboard en tiempo real")
    print("⏹️  Presiona Ctrl+C para parar y generar resultados")
    print("="*70)
    
    # Configuración
    grid_size = 64
    
    # Crear y ejecutar sistema
    infinito = InfinitoContinuo(size=grid_size)
    infinito.run_continuous()

if __name__ == "__main__":
    main()
