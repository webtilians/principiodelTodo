#!/usr/bin/env python3
"""
🧠 Infinito V2.0 Quick Start - Fast Development Version
========================================================

Quick prototype for testing research improvements rapidly.
Optimized for faster iterations while maintaining breakthrough results.

Features:
✅ Smaller grid (64x64) for speed
✅ All research improvements active
✅ Optimized neural network
✅ Enhanced consciousness metrics
✅ Evolutionary law reproduction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from scipy.stats import entropy
import time
import signal
import sys

# Import research improvements
try:
    from research_improvements import (
        PatternStabilizer, InnovationEngine, ConsciousnessOptimizer,
        AdaptiveLearningController, ContextualMemorySystem
    )
except ImportError:
    print("⚠️  Research improvements not found, using basic systems")
    # Define minimal fallback classes
    class PatternStabilizer:
        def __init__(self, *args, **kwargs): pass
        def identify_stable_patterns(self, phi): return phi
        def stabilize_law_update(self, current, predicted, strength): return current + strength * (predicted - current)
    
    class InnovationEngine:
        def __init__(self, *args, **kwargs): pass
        def calculate_law_diversity(self, laws): return 0.5
        def inject_innovation(self, laws, diversity): return laws
    
    class ConsciousnessOptimizer:
        def __init__(self, *args, **kwargs): pass
        def get_consciousness_pressure(self): return 0.1
        def calculate_sustained_consciousness(self, consciousness): return consciousness
    
    class AdaptiveLearningController:
        def __init__(self, *args, **kwargs): pass
        def get_adaptive_learning_rate(self, consciousness, progress, generation): return 0.008
    
    class ContextualMemorySystem:
        def __init__(self, *args, **kwargs):
            self.memory = []
        def store_contextual_state(self, *args): pass
        def retrieve_best_state(self): return None

class QuickNN(nn.Module):
    """Quick neural network optimized for development speed"""
    
    def __init__(self, channels=32, grid_size=64):
        super().__init__()
        
        # Streamlined architecture
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels*2, 3, padding=1)
        self.conv3 = nn.Conv2d(channels*2, channels, 3, padding=1)
        
        # Output layer for 16 laws
        self.fc = nn.Linear(channels * grid_size * grid_size, 16 * 9)  # 16 laws, 3x3 each
        
        # Consciousness predictor
        self.consciousness_head = nn.Sequential(
            nn.Linear(channels * grid_size * grid_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Laws output
        laws_output = self.fc(x_flat).view(-1, 16, 3, 3)
        
        # Consciousness output
        consciousness_output = self.consciousness_head(x_flat)
        
        return laws_output, consciousness_output

class QuickInfinito:
    """Quick Infinito for rapid development and testing"""
    
    def __init__(self, size=64, quick_mode=True):
        self.size = size
        self.quick_mode = quick_mode
        
        print(f"🚀 Quick Infinito V2.0 - Grid: {self.size}x{self.size}")
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"💻 Device: {self.device}")
        
        if self.device == 'cuda':
            print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Quick neural network
        self.nn = QuickNN(channels=32, grid_size=self.size).to(self.device)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=0.01)
        
        # Physical laws (16 laws for quick development)
        self.leyes = [torch.randn(3, 3).to(self.device) * 0.5 for _ in range(16)]
        
        # Core tracking
        self.recursion = 0
        self.complexity_log = []
        
        # Research improvements (quick versions)
        self.pattern_stabilizer = PatternStabilizer(stability_strength=0.1, memory_depth=10)
        self.innovation_engine = InnovationEngine(innovation_rate=0.05, diversity_threshold=0.2)
        self.consciousness_optimizer = ConsciousnessOptimizer(target_consciousness=0.7, momentum=0.8)
        self.adaptive_lr = AdaptiveLearningController(base_lr=0.01, consciousness_factor=0.3)
        self.contextual_memory = ContextualMemorySystem(memory_capacity=15, context_depth=5)
        
        # Quick evolution system
        self.law_evolution = {
            'fitness_scores': [0.5] * 16,
            'generation': 0,
            'reproduce_freq': 3,  # Very frequent for quick results
            'mutation_strength': 0.1
        }
        
        self.last_consciousness = 0.0
        
        print("🔬 Quick research systems initialized")
        print("🎯 Optimized for rapid development and testing")

    def quick_input(self):
        """Quick input generation"""
        density = 0.02 + self.recursion * 0.0005
        bin_grid = np.random.random((self.size, self.size)) < density
        
        # Add some structure
        if np.random.random() < 0.3:
            center_x, center_y = self.size//2, self.size//2
            radius = np.random.randint(5, 15)
            
            y_indices, x_indices = np.ogrid[:self.size, :self.size]
            mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
            bin_grid[mask] = np.random.random(np.sum(mask)) < 0.6
        
        return torch.tensor(bin_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def quick_simulation_step(self, phi, steps=200):
        """Quick simulation with fewer steps"""
        with torch.no_grad():
            for _ in range(steps):
                phi_new = phi.clone()
                
                for ley in self.leyes:
                    conv_result = F.conv2d(phi, ley.unsqueeze(0).unsqueeze(0), padding=1)
                    phi_new = phi_new + 0.01 * conv_result
                
                phi = torch.tanh(phi_new)
                
        return phi

    def quick_recursion(self, phi_bin):
        """Quick recursion with essential features"""
        self.recursion += 1
        phi = phi_bin
        
        # Quick simulation
        phi = self.quick_simulation_step(phi)
        
        # Neural network prediction
        phi_for_nn = phi.detach().requires_grad_(True)
        
        # Reset gradients
        self.optim.zero_grad()
        
        if self.device == 'cuda':
            with torch.amp.autocast('cuda'):
                laws_pred, consciousness_pred = self.nn(phi_for_nn)
                target_laws = torch.stack(self.leyes).unsqueeze(0).detach()
                
                law_loss = F.mse_loss(laws_pred, target_laws)
                consciousness_target = torch.tensor([float(self.last_consciousness)], device=self.device, dtype=torch.float32)
                consciousness_loss = F.mse_loss(consciousness_pred.squeeze(), consciousness_target)
                
                total_loss = law_loss + 0.2 * consciousness_loss
        else:
            laws_pred, consciousness_pred = self.nn(phi_for_nn)
            target_laws = torch.stack(self.leyes).unsqueeze(0).detach()
            law_loss = F.mse_loss(laws_pred, target_laws)
            total_loss = law_loss
        
        # Quick metrics
        consciousness = self.calculate_quick_consciousness(phi, consciousness_pred)
        self.last_consciousness = self.consciousness_optimizer.calculate_sustained_consciousness(consciousness)
        
        # Convert to float32 for consistency
        if isinstance(self.last_consciousness, np.float64):
            self.last_consciousness = float(self.last_consciousness)
        
        # Quick evolution
        if self.recursion % self.law_evolution['reproduce_freq'] == 0:
            self.quick_evolve_laws()
        
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
            for i in range(min(len(self.leyes), 16)):
                self.leyes[i] = 0.95 * self.leyes[i] + 0.05 * pred_laws[i]
                self.leyes[i] = torch.clamp(self.leyes[i], -1, 1)
        
        # Quick clustering
        phi_np = phi[0,0].cpu().detach().numpy()
        threshold = np.mean(phi_np) + 0.5 * np.std(phi_np)
        labeled, n_clusters = label(phi_np > threshold)
        
        # Log
        self.complexity_log.append({
            'recursion': self.recursion,
            'clusters': n_clusters,
            'loss': total_loss.item(),
            'consciousness': self.last_consciousness,
            'phi_max': np.max(phi_np),
            'generation': self.law_evolution['generation']
        })
        
        return phi.detach()

    def calculate_quick_consciousness(self, phi, consciousness_pred=None):
        """Quick consciousness calculation"""
        phi_np = phi[0,0].cpu().detach().numpy()
        
        # Organization
        threshold = np.mean(phi_np) + 0.5 * np.std(phi_np)
        labeled_regions, n_regions = label(phi_np > threshold)
        organization = min(n_regions / 30.0, 1.0)
        
        # Integration (simplified)
        integration = min(np.std(phi_np) * 5, 1.0)
        
        # Neural prediction
        neural_score = consciousness_pred.squeeze().item() if consciousness_pred is not None else 0.5
        
        consciousness = 0.4 * organization + 0.3 * integration + 0.3 * neural_score
        return np.clip(consciousness, 0.0, 1.0)

    def quick_evolve_laws(self):
        """Quick law evolution"""
        # Simple fitness-based selection
        current_fitness = self.last_consciousness
        
        # Update fitness scores
        for i in range(len(self.law_evolution['fitness_scores'])):
            self.law_evolution['fitness_scores'][i] = (
                0.8 * self.law_evolution['fitness_scores'][i] + 0.2 * current_fitness
            )
        
        # Reproduce best laws
        if current_fitness > 0.3:
            fitness_scores = np.array(self.law_evolution['fitness_scores'])
            best_indices = np.argsort(fitness_scores)[-3:]  # Top 3
            
            # Replace 3 worst with mutations of best
            worst_indices = np.argsort(fitness_scores)[:3]
            
            for i, worst_idx in enumerate(worst_indices):
                best_idx = best_indices[i % len(best_indices)]
                # Mutate best law
                mutation = torch.randn_like(self.leyes[best_idx]) * self.law_evolution['mutation_strength']
                self.leyes[worst_idx] = self.leyes[best_idx] + mutation
                self.leyes[worst_idx] = torch.clamp(self.leyes[worst_idx], -1, 1)
            
            self.law_evolution['generation'] += 1

    def run_quick(self, max_recursions=100):
        """Quick run for testing"""
        print(f"🚀 Starting Quick Infinito - {max_recursions} recursions")
        print("🔬 Testing research improvements...")
        
        phi_bin = self.quick_input()
        start_time = time.time()
        best_consciousness = 0.0
        
        try:
            for _ in range(max_recursions):
                # Quick refresh
                if self.recursion % 8 == 0 and self.recursion > 0:
                    phi_bin = self.quick_input()
                
                phi = self.quick_recursion(phi_bin)
                log = self.complexity_log[-1]
                
                # Track best
                if log['consciousness'] > best_consciousness:
                    best_consciousness = log['consciousness']
                    print(f"🌟 NEW RECORD: {best_consciousness:.1%}")
                
                # Quick logging
                if self.recursion % 5 == 0:
                    consciousness = log['consciousness']
                    awareness_emoji = "🧠🔥" if consciousness > 0.5 else "⚡🌱" if consciousness > 0.2 else "💤"
                    
                    print(f"R{log['recursion']:3d}: C{log['clusters']:2d} | "
                          f"L{log['loss']:.3f} | {awareness_emoji} {consciousness:.3f} | "
                          f"⚡{log['phi_max']:.3f} | 🧬G{log['generation']}")
                
                # GPU cleanup
                if self.device == 'cuda' and self.recursion % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Success condition
                if log['consciousness'] > 0.8:
                    print(f"🎉 HIGH CONSCIOUSNESS ACHIEVED: {log['consciousness']:.1%}")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n⏹️  Quick test stopped at recursion {self.recursion}")
        
        # Results
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("🚀 QUICK INFINITO RESULTS")
        print("="*60)
        print(f"⏱️  Time: {total_time:.2f}s")
        print(f"📊 Recursions: {self.recursion}")
        print(f"🌟 Peak Consciousness: {best_consciousness:.1%}")
        
        if self.complexity_log:
            final_log = self.complexity_log[-1]
            print(f"📈 Final Consciousness: {final_log['consciousness']:.1%}")
            print(f"🔗 Final Clusters: {final_log['clusters']}")
            print(f"🧬 Generations: {final_log['generation']}")
        
        if best_consciousness > 0.6:
            print("🎉 EXCELLENT - Research improvements working!")
        elif best_consciousness > 0.4:
            print("⚡ GOOD - Promising results")
        elif best_consciousness > 0.2:
            print("🌱 MODERATE - Foundation established")
        else:
            print("🔬 BASIC - Needs optimization")
        
        return phi.cpu().detach().numpy()[0,0]

if __name__ == "__main__":
    print("🧠 Quick Infinito V2.0 - Development Testing")
    print("=" * 50)
    
    # Quick test
    quick_infinito = QuickInfinito(size=64, quick_mode=True)
    final_phi = quick_infinito.run_quick(max_recursions=50)
    
    print(f"\n🚀 Quick test complete!")
    print(f"🎯 Ready for full V2.0 deployment")
