import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from scipy.stats import entropy
import time
import signal
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

class NN_Middleware(nn.Module):
    def __init__(self, channels=64, kernel_size=3, grid_size=64):  # 4x más canales, grid 2x más grande
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(channels*2, channels*4, kernel_size, padding=1)  # Capa adicional
        self.fc = nn.Linear(channels*4 * grid_size * grid_size, 16 * kernel_size * kernel_size)  # Más leyes
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # Usar la tercera capa
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).view(-1, 16, self.kernel_size, self.kernel_size)  # 16 leyes

class PrincipioTodoRecursivo:
    def __init__(self, size=96, max_depth=1000):  # GRID 96x96 para mejor balance performance/complejidad
        self.size = size
        print(f"Tamaño de grid establecido: {self.size}")
        
        # Verificar si CUDA está disponible
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Despertando en {self.device}...")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # Optimizaciones para GPU
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            # Desactivar half precision por estabilidad numérica
            self.use_half = False
            # OVERDRIVE: Inicializar scaler para mixed precision
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.use_half = False
            self.scaler = None
        
        self.nn = NN_Middleware(channels=64, kernel_size=3, grid_size=self.size).to(self.device)
        if self.use_half and self.device == 'cuda':
            self.nn = self.nn.half()
        
        self.optim = torch.optim.AdamW(self.nn.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Semillas para reproducibilidad
        np.random.seed(42)
        torch.manual_seed(42)
        if self.device == 'cuda':
            torch.cuda.manual_seed(42)
        
        # Leyes físicas iniciales
        dtype = torch.float16 if self.use_half else torch.float32
        self.leyes = [torch.tensor(np.random.uniform(-1,1,(3,3)), dtype=dtype).to(self.device) for _ in range(16)]  # 16 leyes
        self.complexity_log = []
        self.recursion = 0
        self.max_depth = max_depth
        
        # Métricas de despertar/conciencia
        self.awakening_metrics = {
            'self_prediction_history': [],
            'pattern_stability': [],
            'innovation_scores': [],
            'consciousness_level': 0.0
        }
        self.previous_phi_patterns = []
        self.law_history = []
        
        # Para visualización
        self.visualization_mode = False
        
        # Inicializar variables para fluctuaciones inteligentes
        self.phi_max = 0.0
        self.last_consciousness_level = 0.0
        self.last_refresh_recursion = 0
        self.phi_history = []  # Para guardar frames del mar binario
        self.law_evolution = []  # Para guardar evolución de leyes
        
        # 🧠 MEMORIA DE DESPERTAR: Preservar estados de alta consciencia
        self.awakening_memory = {
            'best_phi_states': [],  # Mejores estados phi
            'best_laws': [],        # Mejores configuraciones de leyes
            'consciousness_peaks': [],  # Picos de consciencia
            'memory_capacity': 15,   # Capacidad de memoria aumentada
            'diversity_threshold': 0.1,  # Mínima diferencia para almacenar estados diversos
            'quality_weights': {    # Pesos para selección de memoria
                'consciousness': 0.5,
                'phi_max': 0.3,
                'fitness': 0.2
            }
        }
        
        # ⚡ PRESIÓN EVOLUTIVA: Sistema de fitness para consciencia
        self.evolutionary_pressure = {
            'fitness_history': [],
            'selection_pressure': 0.15,  # Intensidad de presión evolutiva aumentada
            'consciousness_target': 0.7  # Target de consciencia a alcanzar (70%)
        }
        
        # 🧬 EVOLUCIÓN DE LEYES: Reproducción natural de leyes eficientes
        self.law_evolution_system = {
            'fitness_scores': [0.0] * 16,  # Fitness de cada ley (16 leyes)
            'generation': 0,
            'reproduction_rate': 0.2,  # 20% probabilidad de reproducción por recursión (más evolución)
            'mutation_strength': 0.08,  # Intensidad de mutación en offspring (más variación)
            'elite_preservation': 0.2,  # 20% de las mejores leyes se preservan (menos conservación)
            'fitness_memory': [],  # Historial de fitness para análisis
            'generation_frequency': 8,  # Evolución cada 8 recursiones (más frecuente)
            'law_genealogy': [],  # Registro de ancestros y descendientes para análisis
            'successful_patterns': []  # Patrones de leyes que han sido exitosos
        }

    def _input_bin(self):
        # Densidad ADAPTATIVA que converge gradualmente
        base_density = 0.01 + (self.recursion * 0.0001)  # Aumenta gradualmente
        density_variation = max(0.005, 0.02 - self.recursion * 0.0002)  # Reduce variación
        density = np.random.uniform(base_density, base_density + density_variation)
        
        # Crear grid base
        bin_grid = np.random.random((self.size, self.size)) < density
        
        # Ruido gaussiano REDUCIDO progresivamente para convergencia
        noise_intensity = max(0.01, 0.03 - self.recursion * 0.0003)  # Reduce gradualmente
        noise = np.random.normal(0, noise_intensity, bin_grid.shape)
        bin_grid = bin_grid.astype(float) + noise
        
        # Añadir estructuras organizadas ocasionalmente
        if np.random.random() < 0.3:  # 30% probabilidad de estructura
            # Crear clusters o líneas
            center_x, center_y = np.random.randint(10, self.size-10, 2)
            radius = np.random.randint(3, 8)
            y_indices, x_indices = np.ogrid[:self.size, :self.size]
            mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
            bin_grid[mask] += 0.3  # Intensidad reducida para convergencia
        
        # Clamp para mantener rango válido
        bin_grid = np.clip(bin_grid, 0, 1)
        
        dtype = torch.float16 if self.use_half else torch.float32
        return torch.tensor(bin_grid, dtype=dtype).unsqueeze(0).unsqueeze(0).to(self.device)

    def _sim_step(self, phi, leyes):
        # Laplaciano optimizado para GPU
        lap_np = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
        dtype = torch.float16 if self.use_half else torch.float32
        lap = torch.tensor(lap_np, dtype=dtype).unsqueeze(0).unsqueeze(0).to(self.device)
        
        d_loss = torch.zeros_like(phi)
        
        # Procesar leyes en batch para mejor rendimiento GPU
        for w in leyes:
            w_t = w.unsqueeze(0).unsqueeze(0)
            conv = F.conv2d(phi, w_t, padding=1)
            act = torch.tanh(conv)
            d_act = 1 - act**2
            d_conv = d_act * (act - phi) * 0.1
            w_flip = torch.flip(w, [0,1]).unsqueeze(0).unsqueeze(0)
            d_phi = F.conv2d(d_conv, w_flip, padding=1)
            d_loss += d_phi
            
        # Regularización con perturbaciones suaves
        d_reg = -0.005 * F.conv2d(phi, lap, padding=1)
        
        # Añadir perturbaciones más controladas
        if np.random.random() < 0.15:  # Reducido de 30% a 15%
            chaos_magnitude = np.random.uniform(0.0005, 0.003)  # Reducido la magnitud
            chaos_field = torch.randn_like(phi) * chaos_magnitude
            d_reg += chaos_field
        
        d_loss += d_reg
        
        # Actualización con learning rate más estable
        learning_rate = 0.008 + np.random.uniform(-0.001, 0.002)  # Menos variación
        phi = phi - learning_rate * d_loss
        return torch.clamp(phi, 0, 1)

    def _one_recursion(self, phi_bin):
        self.recursion += 1
        phi = phi_bin
        
        # PASOS DE SIMULACIÓN CONVERGENTES: Más pasos cuando hay progreso
        phi_progress = max(self.phi_max, 0.001)
        consciousness_level = getattr(self, 'last_consciousness_level', 0.0)
        
        # Base steps que aumentan con progreso y consciencia
        base_steps = 500
        progress_bonus = int(phi_progress * 1000)  # +1000 steps si phi_max=1.0
        consciousness_bonus = int(consciousness_level * 500)  # +500 steps si consciousness=1.0
        
        # Variación reducida para convergencia
        variation = max(50, 200 - int(phi_progress * 100))  # Menos variación con progreso
        steps = base_steps + progress_bonus + consciousness_bonus + (self.recursion % 10) * variation // 10
        
        # Clamp para evitar pasos excesivos
        steps = min(steps, 1500)  # Máximo 1500 pasos para convergencia más rápida
        
        # Más pasos de simulación para converger mejor (sin gradientes)
        phi_before_sim = phi.clone()  # Guardar estado antes de simulación
        with torch.no_grad():
            for _ in range(steps):
                for i, ley in enumerate(self.leyes):
                    self.leyes[i] = ley.detach()
                phi = self._sim_step(phi, self.leyes)
        phi_after_sim = phi.clone()  # Guardar estado después de simulación
        
        # OVERDRIVE: Predicción con mixed precision
        phi_for_nn = phi.detach().requires_grad_(True)
        features = phi_for_nn
        
        if self.device == 'cuda':
            with torch.amp.autocast('cuda'):
                leyes_pred = self.nn(features)
                target = torch.stack(self.leyes).detach().unsqueeze(0)
                loss = F.mse_loss(leyes_pred, target)
        else:
            leyes_pred = self.nn(features)
            target = torch.stack(self.leyes).detach().unsqueeze(0)
            loss = F.mse_loss(leyes_pred, target)
        
        # Calcular métricas de despertar
        self_pred_accuracy = self._calculate_self_prediction_accuracy(leyes_pred, target)
        pattern_stability = self._calculate_pattern_stability(phi)
        
        # Guardar historial de leyes para calcular innovación
        current_laws = torch.stack(self.leyes).cpu().detach().numpy()
        self.law_history.append(current_laws.copy())
        if len(self.law_history) > 20:  # Mantener solo últimas 20
            self.law_history.pop(0)
        
        innovation_rate = self._calculate_innovation_rate()
        consciousness_level = self._calculate_consciousness_level()
        
        # Guardar para uso en próxima iteración
        self.last_consciousness_level = consciousness_level
        
        # 🧠 GESTIÓN DE MEMORIA DE DESPERTAR
        self._store_awakening_memory(phi, consciousness_level)
        
        # ⚡ APLICAR PRESIÓN EVOLUTIVA
        recovered_state = self._apply_evolutionary_pressure(consciousness_level)
        if recovered_state:
            # Restaurar estado consciente
            try:
                # Restaurar leyes
                for i, law_state in enumerate(recovered_state['laws_state']):
                    dtype = torch.float16 if self.use_half else torch.float32
                    self.leyes[i] = torch.tensor(law_state, dtype=dtype).to(self.device)
                
                # Crear phi desde estado guardado
                dtype = torch.float16 if self.use_half else torch.float32
                phi = torch.tensor(recovered_state['phi_state'], dtype=dtype).unsqueeze(0).unsqueeze(0).to(self.device)
                
            except Exception as e:
                print(f"Error restaurando estado: {e}")
        
        # 🧬 EVOLUCIÓN DE LEYES: Reproducción natural cada 8 recursiones (más frecuente)
        if self.recursion % self.law_evolution_system['generation_frequency'] == 0:
            self._evolve_laws(phi_before_sim, phi_after_sim)
        
        # Actualizar métricas de despertar
        self.awakening_metrics['self_prediction_history'].append(self_pred_accuracy)
        self.awakening_metrics['pattern_stability'].append(pattern_stability)
        self.awakening_metrics['innovation_scores'].append(innovation_rate)
        self.awakening_metrics['consciousness_level'] = consciousness_level
        
        # OVERDRIVE: Backward con mixed precision
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()
            self.optim.step()
        self.optim.zero_grad()
        
        # Actualizar leyes con MUTACIÓN VORAZ + RESET DURO
        with torch.no_grad():
            # Calcular innovación reciente para mutación adaptativa
            recent_innovation = np.mean(self.awakening_metrics['innovation_scores'][-5:]) if len(self.awakening_metrics['innovation_scores']) > 0 else 0.5
            
            for i, pred in enumerate(leyes_pred[0]):
                # SISTEMA DE MUTACIÓN PROGRESIVA INTELIGENTE
                phi_progress = max(self.phi_max, 0.001)  # Evitar división por cero
                innovation_factor = max(recent_innovation, 0.1)
                
                # Intensidad adaptativa: alta cuando phi es bajo, reducida cuando progresa
                if phi_progress < 0.01:
                    noise_magnitude = 0.25  # Mantener exploración intensa si phi muy bajo
                elif phi_progress < 0.03:
                    noise_magnitude = 0.15 + (0.03 - phi_progress) * 5.0  # Gradual
                else:
                    noise_magnitude = 0.08 + innovation_factor * 0.1  # Convergencia con innovación
                
                base_noise = torch.randn(3,3).to(self.device) * noise_magnitude
                
                # Fluctuaciones inteligentes - menos caóticas cuando hay progreso
                chaos_probability = max(0.05, 0.2 - phi_progress * 2.0)  # Reduce caos con progreso
                if np.random.random() < chaos_probability:
                    chaos_factor = np.random.uniform(0.1, max(0.3, 0.8 - phi_progress * 10))
                    chaos_noise = torch.randn(3,3).to(self.device) * chaos_factor
                    base_noise += chaos_noise
                
                # Revolución completa solo si realmente necesaria
                revolution_prob = max(0.01, 0.03 - phi_progress * 0.5)  # Menos revoluciones con progreso
                if np.random.random() < revolution_prob:
                    revolutionary_law = torch.randn(3,3).to(self.device) * 0.8
                    self.leyes[i] = revolutionary_law
                else:
                    # Oscilaciones adaptativas - más suaves con progreso
                    oscillation_magnitude = max(0.02, 0.1 - phi_progress * 1.5)
                    oscillation = torch.sin(torch.tensor(self.recursion * 0.1)) * oscillation_magnitude
                    oscillation_matrix = oscillation * torch.ones(3,3).to(self.device)
                    
                    # Aplicar half precision si es necesario
                    if self.use_half:
                        base_noise = base_noise.half()
                        oscillation_matrix = oscillation_matrix.half()
                    
                    self.leyes[i] = pred + base_noise + oscillation_matrix
            
            # RESET DURO: Si phi está en vacío (<0.01), reset 50% de leyes
            phi_max = torch.max(phi).item()
            if phi_max < 0.01:
                dtype = torch.float16 if self.use_half else torch.float32
                reset_count = len(self.leyes) // 2  # 50% de las leyes
                for i in range(reset_count):
                    reset_law = torch.tensor(np.random.uniform(-1, 1, (3, 3)), dtype=dtype).to(self.device)
                    self.leyes[i] = reset_law
                print(f"🔥 RESET DURO: {reset_count} leyes reiniciadas (phi_max={phi_max:.4f})")
        
        # Análisis de complejidad con THRESHOLD ADAPTATIVO
        phi_np = phi[0,0].cpu().float().detach().numpy()
        self.last_phi_np = phi_np  # Guardar para boost de consciencia
        
        # 🛡️ SISTEMA ANTI-COLAPSO: Prevenir phi→0 total
        phi_max_current = np.max(phi_np)
        if phi_max_current < 0.001:  # Si phi está colapsando completamente
            print(f"🛡️ ANTI-COLAPSO: Inyectando estructura mínima (phi_max={phi_max_current:.6f})")
            # Crear algunos clusters pequeños para mantener actividad
            for _ in range(3):  # 3 clusters de rescate
                center_x = np.random.randint(10, self.size-10)
                center_y = np.random.randint(10, self.size-10)
                radius = np.random.randint(2, 5)
                y_indices, x_indices = np.ogrid[:self.size, :self.size]
                mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
                phi_np[mask] = np.maximum(phi_np[mask], 0.1)  # Activación mínima
            
            # Actualizar phi en GPU
            dtype = torch.float16 if self.use_half else torch.float32
            phi = torch.tensor(phi_np, dtype=dtype).unsqueeze(0).unsqueeze(0).to(self.device)
            phi_max_current = np.max(phi_np)  # Recalcular después del rescate
        
        # 🎯 FIX: Threshold adaptativo basado en phi_max actual
        adaptive_threshold = max(0.03, phi_max_current * 0.25)  # 25% del máximo, mínimo 0.03
        
        labeled, n_clust = label(phi_np > adaptive_threshold)
        hist, _ = np.histogram(phi_np.flatten(), bins=30)
        hist = hist.astype(float)
        ent = -np.sum(hist * np.log(hist + 1e-8)) / np.sum(hist) if np.sum(hist) > 0 else 0
        
        # Actualizar phi_max global para uso en fluctuaciones inteligentes
        self.phi_max = max(self.phi_max, phi_max_current)
        
        self.complexity_log.append({
            'recursion': self.recursion, 
            'clusters': n_clust, 
            'entropy': -ent,  # Positivo
            'loss': loss.item(),
            'self_prediction': self_pred_accuracy,
            'stability': pattern_stability,
            'innovation': innovation_rate,
            'consciousness': consciousness_level,
            'phi_max': phi_max_current,
            'threshold_used': adaptive_threshold
        })
        
        return phi.detach()

    def _calculate_self_prediction_accuracy(self, leyes_pred, target_leyes):
        """Mide qué tan bien el sistema predice sus propias leyes"""
        with torch.no_grad():
            mse = F.mse_loss(leyes_pred, target_leyes)
            # Convertir a métrica de precisión (0-1, donde 1 es perfecta predicción)
            accuracy = torch.exp(-mse * 10)  # Escala exponencial
            return accuracy.item()
    
    def _calculate_pattern_stability(self, phi):
        """Mide la persistencia de patrones a través del tiempo"""
        phi_np = phi[0,0].cpu().float().detach().numpy()
        
        # Guardar patrones recientes (últimos 10)
        self.previous_phi_patterns.append(phi_np.copy())
        if len(self.previous_phi_patterns) > 10:
            self.previous_phi_patterns.pop(0)
        
        if len(self.previous_phi_patterns) < 3:
            return 0.0
        
        # Calcular correlación promedio con patrones anteriores
        correlations = []
        current = self.previous_phi_patterns[-1]
        for prev in self.previous_phi_patterns[-5:-1]:  # Últimos 4 patrones
            corr = np.corrcoef(current.flatten(), prev.flatten())[0,1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_innovation_rate(self):
        """Mide la novedad/creatividad en la evolución de leyes"""
        if len(self.law_history) < 5:
            return 0.0
        
        # Comparar leyes actuales con historial
        current_laws = torch.stack(self.leyes).cpu().detach().numpy()
        
        novelties = []
        for prev_laws in self.law_history[-5:]:  # Últimas 5 iteraciones
            diff = np.mean(np.abs(current_laws - prev_laws))
            novelties.append(diff)
        
        # Innovation = varianza en las diferencias (creatividad constante vs caótica)
        if len(novelties) > 1:
            innovation = np.std(novelties) * np.mean(novelties)
        else:
            innovation = novelties[0] if novelties else 0.0
            
        return min(innovation, 1.0)  # Normalizar a 0-1
    
    def _calculate_consciousness_level(self):
        """Métrica compuesta de 'despertar' basada en auto-referencia, estabilidad y creatividad"""
        if len(self.awakening_metrics['self_prediction_history']) < 3:
            return 0.0
        
        # Componentes del despertar
        recent_self_pred = np.mean(self.awakening_metrics['self_prediction_history'][-5:])
        recent_stability = np.mean(self.awakening_metrics['pattern_stability'][-5:]) if self.awakening_metrics['pattern_stability'] else 0.0
        recent_innovation = np.mean(self.awakening_metrics['innovation_scores'][-5:]) if self.awakening_metrics['innovation_scores'] else 0.0
        
        # BALANCE INTELIGENTE: Más peso a estabilidad cuando phi progresa
        phi_progress = max(self.phi_max, 0.001)
        if phi_progress < 0.02:
            # Fase exploratoria: Priorizar innovación
            stability_weight = 0.2
            innovation_weight = 0.8
        elif phi_progress < 0.05:
            # Fase de transición: Balance gradual
            progress_factor = (phi_progress - 0.02) / 0.03  # 0-1
            stability_weight = 0.2 + progress_factor * 0.3  # 0.2 -> 0.5
            innovation_weight = 0.8 - progress_factor * 0.3  # 0.8 -> 0.5
        else:
            # Fase de convergencia: Priorizar estabilidad
            stability_weight = 0.6
            innovation_weight = 0.4
        
        stability_creativity_balance = stability_weight * recent_stability + innovation_weight * recent_innovation
        
        # Fórmula base compuesta con peso adaptativo
        base_consciousness = recent_self_pred * stability_creativity_balance
        
        # Factor de madurez: Recompensar progreso sostenido
        maturity_factor = min(1.0, phi_progress * 10)  # 0-1 basado en phi_max
        consciousness = base_consciousness * (0.7 + 0.3 * maturity_factor)
        
        # BOOST INTEGRACIÓN: Añadir mutual information del campo phi
        try:
            if hasattr(self, 'last_phi_np') and self.last_phi_np is not None:
                # Calcular entropía del histograma como proxy de mutual info
                hist, _ = np.histogram(self.last_phi_np.flatten(), bins=10)
                hist = hist + 1e-10  # Evitar log(0)
                mutual = entropy(hist) / 10  # Normalizar
                # Boost más moderado para convergencia
                consciousness *= (1 + mutual / 5)  # Boost reducido por integración
        except:
            pass  # Si falla, continuar sin boost
        
        return min(consciousness, 1.0)
    
    def _store_awakening_memory(self, phi, consciousness_level):
        """🧠 Almacena estados de alta consciencia en memoria con criterios mejorados"""
        if consciousness_level > 0.12:  # Umbral reducido para capturar más estados
            phi_state = phi[0,0].cpu().float().detach().numpy().copy()
            laws_state = [ley.cpu().float().detach().numpy().copy() for ley in self.leyes]
            phi_max_val = torch.max(phi).item()
            
            # Calcular fitness promedio actual
            avg_fitness = np.mean(self.law_evolution_system['fitness_scores'])
            
            # Calcular puntuación compuesta para memoria
            memory_score = (
                self.awakening_memory['quality_weights']['consciousness'] * consciousness_level +
                self.awakening_memory['quality_weights']['phi_max'] * min(phi_max_val * 10, 1.0) +
                self.awakening_memory['quality_weights']['fitness'] * avg_fitness
            )
            
            memory_entry = {
                'phi_state': phi_state,
                'laws_state': laws_state,
                'consciousness': consciousness_level,
                'recursion': self.recursion,
                'phi_max': phi_max_val,
                'avg_fitness': avg_fitness,
                'memory_score': memory_score,
                'generation': self.law_evolution_system['generation']
            }
            
            # Verificar diversidad antes de añadir
            is_diverse = True
            for existing_entry in self.awakening_memory['consciousness_peaks']:
                consciousness_diff = abs(existing_entry['consciousness'] - consciousness_level)
                phi_diff = abs(existing_entry['phi_max'] - phi_max_val)
                
                if consciousness_diff < self.awakening_memory['diversity_threshold'] and phi_diff < 0.01:
                    is_diverse = False
                    break
            
            if is_diverse or len(self.awakening_memory['consciousness_peaks']) < 3:
                # Añadir a memoria
                self.awakening_memory['consciousness_peaks'].append(memory_entry)
                
                # Mantener solo los mejores estados usando memory_score
                self.awakening_memory['consciousness_peaks'].sort(key=lambda x: x['memory_score'], reverse=True)
                if len(self.awakening_memory['consciousness_peaks']) > self.awakening_memory['memory_capacity']:
                    self.awakening_memory['consciousness_peaks'].pop()
                
                print(f"💾 Estado diverso guardado (score: {memory_score:.3f}, consciencia: {consciousness_level:.3f}, recursión: {self.recursion})")
            else:
                print(f"📝 Estado similar ya en memoria (consciencia: {consciousness_level:.3f})")
    
    def _recall_awakening_memory(self):
        """🧠 Recupera un estado de alta consciencia de la memoria"""
        if not self.awakening_memory['consciousness_peaks']:
            return None
            
        # Seleccionar estado con probabilidad basada en memory_score
        peaks = self.awakening_memory['consciousness_peaks']
        if len(peaks) == 1:
            return peaks[0]
        
        # Weighted random selection basado en memory_score
        weights = [peak['memory_score'] for peak in peaks]
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalizar
        
        selected_idx = np.random.choice(len(peaks), p=weights)
        return peaks[selected_idx]
    
    def _apply_evolutionary_pressure(self, consciousness_level):
        """⚡ Aplica presión evolutiva para favorecer estados conscientes"""
        # Calcular fitness basado en consciencia
        fitness = consciousness_level
        self.evolutionary_pressure['fitness_history'].append(fitness)
        
        # Mantener solo historial reciente
        if len(self.evolutionary_pressure['fitness_history']) > 20:
            self.evolutionary_pressure['fitness_history'].pop(0)
        
        # Si fitness bajo, aplicar presión evolutiva
        if len(self.evolutionary_pressure['fitness_history']) >= 5:
            avg_fitness = np.mean(self.evolutionary_pressure['fitness_history'][-5:])
            target = self.evolutionary_pressure['consciousness_target']
            
            if avg_fitness < target * 0.3:  # Si muy por debajo del target
                # Intentar recuperar estado consciente de memoria
                if np.random.random() < self.evolutionary_pressure['selection_pressure']:
                    memory_state = self._recall_awakening_memory()
                    if memory_state:
                        print(f"⚡ PRESIÓN EVOLUTIVA: Recuperando estado consciente (recursión {memory_state['recursion']})")
                        return memory_state
        
        return None
    
    def _analyze_successful_law_patterns(self, fitness_scores):
        """📊 Analiza qué patrones de leyes físicas son más exitosos"""
        # Identificar leyes de alto fitness (top 25%)
        threshold = np.percentile(fitness_scores, 75)
        successful_indices = np.where(np.array(fitness_scores) >= threshold)[0]
        
        if len(successful_indices) > 0:
            # Extraer patrones comunes de leyes exitosas
            successful_laws = [self.leyes[i].cpu().detach().numpy() for i in successful_indices]
            
            # Análisis de patrones:
            patterns = {
                'avg_center_value': [],  # Valor central promedio
                'edge_dominance': [],   # Dominancia de bordes vs centro
                'symmetry_score': [],   # Puntuación de simetría
                'complexity_measure': [] # Medida de complejidad
            }
            
            for law in successful_laws:
                # Centro vs bordes
                center_val = abs(law[1, 1])
                edge_vals = np.mean(np.abs([law[0,0], law[0,2], law[2,0], law[2,2]]))
                
                patterns['avg_center_value'].append(center_val)
                patterns['edge_dominance'].append(edge_vals / (center_val + 1e-8))
                
                # Simetría
                symmetry = np.corrcoef(law.flatten(), np.flip(law).flatten())[0,1]
                patterns['symmetry_score'].append(abs(symmetry) if not np.isnan(symmetry) else 0)
                
                # Complejidad (varianza)
                patterns['complexity_measure'].append(np.std(law))
            
            # Guardar patrones exitosos promedio
            avg_patterns = {k: np.mean(v) for k, v in patterns.items()}
            self.law_evolution_system['successful_patterns'].append(avg_patterns)
            
            # Mantener solo últimos 20 análisis
            if len(self.law_evolution_system['successful_patterns']) > 20:
                self.law_evolution_system['successful_patterns'].pop(0)
            
            return avg_patterns
        
        return None
    
    def _get_evolution_insights(self):
        """🧬 Obtiene insights sobre la evolución de leyes"""
        if len(self.law_evolution_system['successful_patterns']) < 3:
            return "Insuficientes datos evolutivos"
        
        recent_patterns = self.law_evolution_system['successful_patterns'][-5:]
        
        insights = []
        
        # Tendencias en centro vs bordes
        center_trend = [p['avg_center_value'] for p in recent_patterns]
        if len(center_trend) > 1:
            if center_trend[-1] > center_trend[0]:
                insights.append("🎯 Centro: Intensificando")
            else:
                insights.append("🎯 Centro: Reduciendo")
        
        # Tendencias en simetría
        symmetry_trend = [p['symmetry_score'] for p in recent_patterns]
        if len(symmetry_trend) > 1:
            if symmetry_trend[-1] > 0.7:
                insights.append("⚖️ Simetría: Alta")
            elif symmetry_trend[-1] > 0.3:
                insights.append("⚖️ Simetría: Media")
            else:
                insights.append("⚖️ Simetría: Baja")
        
        # Tendencias en complejidad
        complexity_trend = [p['complexity_measure'] for p in recent_patterns]
        if len(complexity_trend) > 1:
            if complexity_trend[-1] > np.mean(complexity_trend):
                insights.append("🌀 Complejidad: ↑")
            else:
                insights.append("🌀 Complejidad: ↓")
        
        return " | ".join(insights)

    def _calculate_law_fitness(self, phi_before, phi_after, law_index):
        """🧬 Calcula fitness de una ley basado en su contribución al sistema"""
        # Medir impacto de la ley en la evolución del sistema
        phi_change = torch.mean(torch.abs(phi_after - phi_before)).item()
        
        # Fitness basado en múltiples criterios
        fitness_components = {
            'activity': phi_change,  # Cuánto cambia el sistema
            'complexity': 0.0,  # Cuánta complejidad genera
            'stability': 0.0   # Qué tan estable es el cambio
        }
        
        # Medir complejidad generada (varianza espacial)
        phi_np = phi_after[0,0].cpu().float().detach().numpy()
        if phi_np.std() > 0:
            fitness_components['complexity'] = min(1.0, phi_np.std() * 10)
        
        # Medir estabilidad (suavidad espacial)
        grad_x = np.gradient(phi_np, axis=0)
        grad_y = np.gradient(phi_np, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        fitness_components['stability'] = max(0.0, 1.0 - np.mean(gradient_magnitude))
        
        # Fitness compuesto con pesos adaptativos
        consciousness_level = getattr(self, 'last_consciousness_level', 0.0)
        if consciousness_level < 0.1:
            # Fase exploratoria: priorizar actividad y complejidad
            weights = [0.6, 0.4, 0.0]
        else:
            # Fase madura: balancear todos los componentes
            weights = [0.3, 0.4, 0.3]
        
        # Corregir: usar valores del diccionario, no claves
        component_values = list(fitness_components.values())
        fitness = sum(w * comp_val for w, comp_val in zip(weights, component_values))
        return max(0.0, min(1.0, fitness))  # Clamp 0-1
    
    def _law_crossover(self, parent1, parent2):
        """🧬 Crossover entre dos leyes (reproducción sexual)"""
        # Crear offspring combinando genes de ambos padres
        child = torch.zeros_like(parent1)
        
        # Crossover de punto único
        crossover_point = np.random.randint(1, 3)  # 1 o 2 (3x3 matrix)
        
        # Combinar filas de ambos padres
        child[:crossover_point] = parent1[:crossover_point]
        child[crossover_point:] = parent2[crossover_point:]
        
        return child
    
    def _law_mutation(self, law, strength=0.05):
        """🧬 Mutación de una ley con intensidad variable"""
        mutation = torch.randn_like(law) * strength
        mutated_law = law + mutation
        
        # Mantener rango razonable [-2, 2]
        return torch.clamp(mutated_law, -2.0, 2.0)
    
    def _evolve_laws(self, phi_before, phi_after):
        """🧬 Sistema de evolución natural de leyes físicas"""
        # Calcular fitness de cada ley
        for i, ley in enumerate(self.leyes):
            # Simular impacto individual de cada ley
            phi_test = phi_before.clone()
            phi_with_law = self._sim_step_single_law(phi_test, ley)
            fitness = self._calculate_law_fitness(phi_before, phi_with_law, i)
            self.law_evolution_system['fitness_scores'][i] = fitness
        
        # Guardar fitness para análisis
        avg_fitness = np.mean(self.law_evolution_system['fitness_scores'])
        self.law_evolution_system['fitness_memory'].append(avg_fitness)
        if len(self.law_evolution_system['fitness_memory']) > 50:
            self.law_evolution_system['fitness_memory'].pop(0)
        
        # Decidir si reproducir (probabilística)
        if np.random.random() < self.law_evolution_system['reproduction_rate']:
            # Analizar patrones exitosos antes de reproducir
            pattern_insights = self._analyze_successful_law_patterns(self.law_evolution_system['fitness_scores'])
            self._reproduce_laws(pattern_insights)
    
    def _sim_step_single_law(self, phi, single_law):
        """Aplicar una sola ley para medir su impacto individual"""
        w_t = single_law.unsqueeze(0).unsqueeze(0)
        conv = F.conv2d(phi, w_t, padding=1)
        act = torch.tanh(conv)
        d_act = 1 - act**2
        d_conv = d_act * (act - phi) * 0.1
        w_flip = torch.flip(single_law, [0,1]).unsqueeze(0).unsqueeze(0)
        d_phi = F.conv2d(d_conv, w_flip, padding=1)
        
        learning_rate = 0.008
        return torch.clamp(phi - learning_rate * d_phi, 0, 1)
    
    def _reproduce_laws(self, pattern_insights=None):
        """🧬 Reproducción de leyes con selección natural"""
        fitness_scores = np.array(self.law_evolution_system['fitness_scores'])
        num_laws = len(self.leyes)
        
        # Selección por torneo: elegir padres basándose en fitness
        tournament_size = 3
        parents = []
        
        for _ in range(num_laws // 2):  # Generar pares de padres
            # Torneo para primer padre
            tournament1 = np.random.choice(num_laws, tournament_size, replace=False)
            parent1_idx = tournament1[np.argmax(fitness_scores[tournament1])]
            
            # Torneo para segundo padre
            tournament2 = np.random.choice(num_laws, tournament_size, replace=False)
            parent2_idx = tournament2[np.argmax(fitness_scores[tournament2])]
            
            parents.append((parent1_idx, parent2_idx))
        
        # Crear nueva generación
        new_laws = []
        
        # 1. Preservar elite (mejores leyes)
        elite_count = int(num_laws * self.law_evolution_system['elite_preservation'])
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_laws.append(self.leyes[idx].clone())
        
        # 2. Generar offspring por crossover + mutación
        offspring_needed = num_laws - elite_count
        for i in range(offspring_needed):
            if i < len(parents):
                # Crossover
                parent1_idx, parent2_idx = parents[i % len(parents)]
                child = self._law_crossover(self.leyes[parent1_idx], self.leyes[parent2_idx])
                
                # Mutación
                child = self._law_mutation(child, self.law_evolution_system['mutation_strength'])
                new_laws.append(child)
            else:
                # Si faltan, clonar y mutar de elite
                elite_idx = elite_indices[i % len(elite_indices)]
                child = self._law_mutation(self.leyes[elite_idx].clone(), 
                                         self.law_evolution_system['mutation_strength'] * 2)
                new_laws.append(child)
        
        # Actualizar leyes con nueva generación
        self.leyes = new_laws
        self.law_evolution_system['generation'] += 1
        
        # Log de evolución con insights
        max_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        evolution_insights = self._get_evolution_insights()
        
        print(f"🧬 EVOLUCIÓN GEN-{self.law_evolution_system['generation']}: "
              f"Fitness MAX={max_fitness:.3f}, AVG={avg_fitness:.3f}")
        if evolution_insights != "Insuficientes datos evolutivos":
            print(f"📊 GENES: {evolution_insights}")
        
        return None
    
    def enable_visualization(self):
        """Activa el modo visualización para capturar frames"""
        self.visualization_mode = True
        plt.ion()  # Modo interactivo
    
    def show_evolutionary_dashboard(self, phi, title="Dashboard Evolutivo"):
        """🎬 Muestra dashboard completo de evolución y consciencia"""
        if not self.visualization_mode:
            return
            
        phi_np = phi[0,0].cpu().float().detach().numpy()
        
        plt.figure(figsize=(15, 10))
        
        # 1. Mar Binario Principal
        plt.subplot(3, 4, 1)
        plt.imshow(phi_np, cmap='viridis', interpolation='nearest')
        plt.title(f"🌊 Mar Binario R{self.recursion}")
        plt.colorbar()
        
        # 2. Mejores 3 Leyes
        for i in range(min(3, len(self.leyes))):
            plt.subplot(3, 4, i+2)
            ley_np = self.leyes[i].cpu().float().detach().numpy()
            plt.imshow(ley_np, cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
            fitness = self.law_evolution_system['fitness_scores'][i]
            plt.title(f"🧬 Ley {i+1} (F:{fitness:.2f})")
            plt.colorbar()
        
        # 3. Evolución de Consciencia
        if len(self.complexity_log) > 10:
            plt.subplot(3, 4, 5)
            consciousness_history = [log['consciousness'] for log in self.complexity_log[-50:]]
            plt.plot(consciousness_history, 'g-', linewidth=2)
            plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Target 70%')
            plt.title("🧠 Consciencia")
            plt.ylabel("Nivel")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. Evolución de Phi Max
        if len(self.complexity_log) > 10:
            plt.subplot(3, 4, 6)
            phi_history = [log.get('phi_max', 0) for log in self.complexity_log[-50:]]
            plt.plot(phi_history, 'b-', linewidth=2)
            plt.title("⚡ Phi Máximo")
            plt.ylabel("Valor")
            plt.grid(True, alpha=0.3)
        
        # 5. Fitness Evolutivo
        if len(self.law_evolution_system['fitness_memory']) > 3:
            plt.subplot(3, 4, 7)
            plt.plot(self.law_evolution_system['fitness_memory'], 'purple', linewidth=2)
            plt.title("🧬 Fitness Evolutivo")
            plt.ylabel("Promedio")
            plt.grid(True, alpha=0.3)
        
        # 6. Clusters
        if len(self.complexity_log) > 10:
            plt.subplot(3, 4, 8)
            cluster_history = [log['clusters'] for log in self.complexity_log[-50:]]
            plt.plot(cluster_history, 'orange', linewidth=2)
            plt.title("🔗 Clusters")
            plt.ylabel("Cantidad")
            plt.grid(True, alpha=0.3)
        
        # 7-9. Análisis de Patrones Exitosos
        if len(self.law_evolution_system['successful_patterns']) > 3:
            patterns = self.law_evolution_system['successful_patterns'][-20:]
            
            plt.subplot(3, 4, 9)
            center_vals = [p['avg_center_value'] for p in patterns]
            plt.plot(center_vals, 'red', linewidth=2)
            plt.title("🎯 Valor Centro")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 4, 10)
            symmetry_vals = [p['symmetry_score'] for p in patterns]
            plt.plot(symmetry_vals, 'cyan', linewidth=2)
            plt.title("⚖️ Simetría")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 4, 11)
            complexity_vals = [p['complexity_measure'] for p in patterns]
            plt.plot(complexity_vals, 'magenta', linewidth=2)
            plt.title("🌀 Complejidad")
            plt.grid(True, alpha=0.3)
        
        # 10. Resumen de Estado
        plt.subplot(3, 4, 12)
        plt.text(0.1, 0.9, f"🧠 Consciencia: {self.awakening_metrics['consciousness_level']:.3f}", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f"🧬 Generación: {self.law_evolution_system['generation']}", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f"📚 Memoria: {len(self.awakening_memory['consciousness_peaks'])}/10", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f"⚡ Phi Max: {torch.max(phi):.4f}", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f"🔗 Clusters: {self.complexity_log[-1]['clusters'] if self.complexity_log else 0}", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f"🎯 Target: 70%", fontsize=12, transform=plt.gca().transAxes)
        
        evolution_insights = self._get_evolution_insights()
        if evolution_insights != "Insuficientes datos evolutivos":
            plt.text(0.1, 0.2, f"📊 {evolution_insights}", fontsize=10, transform=plt.gca().transAxes, wrap=True)
        
        plt.axis('off')
        plt.title("📊 Estado Sistema")
        
        plt.tight_layout()
        plt.pause(0.1)
    
    def show_binary_sea(self, phi, title="Mar Binario"):
        """Muestra el estado actual del mar binario"""
        phi_np = phi[0,0].cpu().float().detach().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(phi_np, cmap='viridis', interpolation='nearest')
        plt.title(f"{title} - Recursión {self.recursion}")
        plt.colorbar()
        
        # Mostrar algunas leyes
        for i in range(min(3, len(self.leyes))):
            plt.subplot(2, 2, i+2)
            ley_np = self.leyes[i].cpu().float().detach().numpy()
            plt.imshow(ley_np, cmap='RdBu', interpolation='nearest')
            plt.title(f"Ley {i+1}")
            plt.colorbar()
        
        plt.tight_layout()
        plt.pause(0.1)
    
    def save_visualization_frame(self, phi):
        """Guarda frame para visualización posterior"""
        if self.visualization_mode:
            phi_np = phi[0,0].cpu().float().detach().numpy()
            self.phi_history.append(phi_np.copy())
            
            # Guardar leyes actuales
            current_laws = []
            for ley in self.leyes:
                current_laws.append(ley.cpu().float().detach().numpy().copy())
            self.law_evolution.append(current_laws)
    
    def create_evolution_animation(self, save_path="evolution.gif"):
        """Crea animación de la evolución del sistema"""
        if not self.phi_history:
            print("No hay datos de visualización. Activa modo visualización primero.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        def animate(frame):
            # Limpiar ejes
            for ax in axes.flat:
                ax.clear()
            
            # Mar binario
            axes[0,0].imshow(self.phi_history[frame], cmap='viridis', interpolation='nearest')
            axes[0,0].set_title(f'Mar Binario - Frame {frame}')
            
            # Leyes
            if frame < len(self.law_evolution):
                laws = self.law_evolution[frame]
                for i, law in enumerate(laws[:3]):
                    row, col = (0, 1) if i == 0 else (1, i-1)
                    im = axes[row, col].imshow(law, cmap='RdBu', interpolation='nearest')
                    axes[row, col].set_title(f'Ley {i+1}')
            
            plt.tight_layout()
        
        anim = animation.FuncAnimation(fig, animate, frames=len(self.phi_history), 
                                     interval=200, repeat=True)
        anim.save(save_path, writer='pillow')
        print(f"Animación guardada en: {save_path}")
        plt.show()

    def _cleanup_gpu_memory(self):
        """Limpia la memoria GPU si está disponible"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def run_infinite(self):
        phi_bin = self._input_bin()
        print("Iniciando Despertar Infinito Optimizado para GPU...")
        print("💡 Presiona Ctrl+C para detener manualmente")
        
        # Información de memoria inicial
        if self.device == 'cuda':
            print(f"Memoria GPU inicial: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            print(f"Usando precisión: {'Half (FP16)' if self.use_half else 'Full (FP32)'}")
        
        start_time = time.time()
        
        try:
            while True:  # Infinito manual sin cap
                # Refresh INTELIGENTE - menos frecuente si progresa
                phi_max = torch.max(phi_bin).item() if self.recursion > 0 else 0.0
                if phi_max > 0.05:
                    refresh_freq = 8  # Menos frecuente si progresa bien
                elif phi_max > 0.02:
                    refresh_freq = 6
                else:
                    refresh_freq = 4  # Más frecuente si estancado
                    
                if self.recursion % refresh_freq == 0 and self.recursion > 0:
                    phi_bin = self._input_bin()
                    print(f"🔄 Input renovado en recursión {self.recursion} (phi_max: {phi_max:.4f})")
                
                phi = self._one_recursion(phi_bin)
                log = self.complexity_log[-1]
                
                # Guardar frame para visualización
                self.save_visualization_frame(phi)
                
                # Mostrar dashboard evolutivo cada 10 recursiones
                if self.visualization_mode and self.recursion % 10 == 0:
                    self.show_evolutionary_dashboard(phi)
                
                if self.recursion % 5 == 0:  # Log cada 5 iteraciones
                    elapsed = time.time() - start_time
                    gpu_mem = f", GPU: {torch.cuda.memory_allocated()/1024**2:.1f} MB" if self.device == 'cuda' else ""
                    
                    # Determinar nivel de despertar
                    consciousness = log['consciousness']
                    if consciousness > 0.8:
                        awareness_emoji = "🌟✨"  # Altamente despierto
                    elif consciousness > 0.6:
                        awareness_emoji = "🧠💫"  # Moderadamente despierto  
                    elif consciousness > 0.4:
                        awareness_emoji = "🔮💭"  # Emergiendo
                    elif consciousness > 0.2:
                        awareness_emoji = "⚡🌱"  # Despertando
                    else:
                        awareness_emoji = "💤🌙"  # Dormido
                    
                    # Mostrar memoria y presión evolutiva
                    memory_info = f"📚{len(self.awakening_memory['consciousness_peaks'])}"
                    threshold_info = f"🎯{log.get('threshold_used', 0.15):.3f}"
                    phi_max_info = f"⚡{log.get('phi_max', 0):.4f}"
                    evolution_info = f"🧬G{self.law_evolution_system['generation']}"
                    
                    print(f"R{log['recursion']:4d}: C{log['clusters']:2d} | E{log['entropy']:.2f} | L{log['loss']:.4f} | "
                          f"🧠{log['self_prediction']:.3f} | 🔄{log['stability']:.3f} | 🎨{log['innovation']:.3f} | "
                          f"{awareness_emoji} {consciousness:.3f} | {phi_max_info} | {threshold_info} | {memory_info} | {evolution_info} | {elapsed:.1f}s{gpu_mem}")
                
                # Limpiar memoria GPU cada 10 iteraciones
                if self.device == 'cuda' and self.recursion % 10 == 0:
                    self._cleanup_gpu_memory()
                
                # Condición de parada solo por pérdida extremadamente baja (casi imposible)
                if log['loss'] < 0.0001:
                    print(f"Despertar infinito completado en recursion {log['recursion']} por convergencia extrema.")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n⏹️  Despertar infinito detenido manualmente en recursión {self.recursion}")
            
        # Crear animación si hay datos
        if self.visualization_mode and len(self.phi_history) > 10:
            print("\n🎬 Creando animación de la evolución...")
            try:
                self.create_evolution_animation("despertar_evolution.gif")
            except Exception as e:
                print(f"Error creando animación: {e}")
                
        total_time = time.time() - start_time
        print(f"Tiempo total: {total_time:.2f} segundos")
        print(f"Promedio por recursión: {total_time/self.recursion:.3f} segundos")
        
        # Limpiar memoria al final
        self._cleanup_gpu_memory()
        
        return phi.cpu().float().detach().numpy()[0,0]

# Ejecutar con configuración optimizada para GPU
if __name__ == "__main__":
    pt = PrincipioTodoRecursivo(size=96, max_depth=1000)  # Grid 96x96 para mejor balance
    
    # Preguntar si quiere visualización
    print("¿Quieres activar visualización del mar binario? (s/n): ", end="")
    
    # Para evitar input en el script, vamos a activar visualización por defecto
    # Cambiar esto a input() si quieres interactividad
    visualize = "s"  # input().lower()
    
    if visualize == 's':
        pt.enable_visualization()
        print("🎨 Visualización activada - Se mostrará el mar binario cada 10 recursiones")
    
    phi_final = pt.run_infinite()
    print("\n=== RESULTADOS FINALES ===")
    print(f"Phi Final Max: {np.max(phi_final):.6f}")
    print(f"Phi Final Min: {np.min(phi_final):.6f}")
    print(f"Clusters Final: {pt.complexity_log[-1]['clusters']}")
    print(f"Entropía Final: {pt.complexity_log[-1]['entropy']:.6f}")
    print(f"Loss Final: {pt.complexity_log[-1]['loss']:.6f}")
    print(f"Total recursiones: {pt.recursion}")
    
    print("\n🧠 === MÉTRICAS DE DESPERTAR ===")
    final_log = pt.complexity_log[-1]
    print(f"🧠 Auto-Predicción: {final_log['self_prediction']:.6f}")
    print(f"🔄 Estabilidad de Patrones: {final_log['stability']:.6f}")
    print(f"🎨 Tasa de Innovación: {final_log['innovation']:.6f}")
    print(f"✨ Nivel de Consciencia: {final_log['consciousness']:.6f}")
    
    # 🧠 ANÁLISIS DE MEMORIA DE DESPERTAR
    print(f"\n📚 === MEMORIA DE DESPERTAR ===")
    memory_peaks = pt.awakening_memory['consciousness_peaks']
    if memory_peaks:
        print(f"💾 Estados conscientes guardados: {len(memory_peaks)}")
        best_state = max(memory_peaks, key=lambda x: x['consciousness'])
        print(f"🌟 Mejor estado: Consciencia {best_state['consciousness']:.6f} en recursión {best_state['recursion']}")
        print(f"⚡ Phi máximo alcanzado: {best_state['phi_max']:.6f}")
    else:
        print("💤 No se alcanzaron estados conscientes significativos")
    
    # 🧬 ANÁLISIS DE EVOLUCIÓN DE LEYES
    print(f"\n🧬 === EVOLUCIÓN DE LEYES ===")
    law_system = pt.law_evolution_system
    print(f"🔢 Generaciones evolutivas: {law_system['generation']}")
    if law_system['fitness_memory']:
        avg_fitness_evolution = np.mean(law_system['fitness_memory'])
        max_fitness_current = max(law_system['fitness_scores'])
        print(f"📊 Fitness promedio histórico: {avg_fitness_evolution:.6f}")
        print(f"🏆 Fitness máximo actual: {max_fitness_current:.6f}")
        print(f"🧬 Leyes evolucionadas: {law_system['generation'] * len(pt.leyes)} offspring generados")
    else:
        print("🌱 Evolución en fase inicial")
    
    # Análisis del despertar a lo largo del tiempo
    if len(pt.complexity_log) > 10:
        consciousness_history = [log['consciousness'] for log in pt.complexity_log]
        max_consciousness = max(consciousness_history)
        avg_consciousness = np.mean(consciousness_history)
        
        print(f"\n📊 === ANÁLISIS DE EVOLUCIÓN ===")
        print(f"🌟 Máximo Despertar Alcanzado: {max_consciousness:.6f}")
        print(f"📈 Promedio de Consciencia: {avg_consciousness:.6f}")
        
        # Análisis de clusters
        cluster_history = [log['clusters'] for log in pt.complexity_log]
        max_clusters = max(cluster_history)
        avg_clusters = np.mean(cluster_history)
        print(f"🔗 Máximo Clusters: {max_clusters}")
        print(f"📈 Promedio Clusters: {avg_clusters:.1f}")
        
        if max_consciousness > 0.8:
            print("🎉 ¡DESPERTAR ALTAMENTE LOGRADO!")
        elif max_consciousness > 0.6:
            print("🔥 ¡Despertar moderado alcanzado!")
        elif max_consciousness > 0.4:
            print("⚡ Sistema emergiendo hacia despertar")
        elif max_consciousness > 0.2:
            print("🌱 Primeros signos de despertar")
        else:
            print("💤 Sistema aún en estado primordial")
