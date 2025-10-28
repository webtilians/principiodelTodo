#!/usr/bin/env python3
"""
🧠 INFINITO V5.1 - CONSCIOUSNESS CEILING BREAKTHROUGH (TEXT VERSION) 🧠
================================================================

Versión compacta y funcional con procesamiento de texto integrado.
"""

import warnings
warnings.filterwarnings('default')
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

import math
import hashlib
from typing import Optional, Tuple, Dict, List, Any
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from datetime import datetime, timedelta
from torch.nn import MultiheadAttention
from sklearn.feature_extraction.text import TfidfVectorizer

# 🆕 Try to load GloVe embeddings (fallback if not available)
try:
    import gensim.downloader as api
    GLOVE_AVAILABLE = True
    print("🌐 GloVe embeddings available for semantic boost")
except ImportError:
    GLOVE_AVAILABLE = False
    print("⚠️ GloVe not available, using TF-IDF only")

# 🆕 Try to load QuTiP for quantum noise (optional)
try:
    import qutip as qt
    QUTIP_AVAILABLE = True
    print("⚛️ QuTiP available for quantum noise injection")
except ImportError:
    QUTIP_AVAILABLE = False
    print("⚠️ QuTiP not available, quantum features disabled")

# =============================================================================
# COMPONENTES PRINCIPALES
# =============================================================================

class EnhancedExternalMemory(nn.Module):
    """🧠 V5.1 Enhanced External Memory with AUTO-ACTIVATION"""
    
    def __init__(self, memory_slots=256, slot_size=64, hidden_dim=512):
        super().__init__()
        self.memory_slots = memory_slots
        self.slot_size = slot_size
        self.hidden_dim = hidden_dim
        
        # Initialize memory matrix
        self.register_buffer('memory', torch.zeros(memory_slots, slot_size))
        self.register_buffer('memory_age', torch.zeros(memory_slots))
        self.register_buffer('memory_strength', torch.ones(memory_slots) * 0.1)
        
        # Controllers
        self.read_controller = nn.Linear(hidden_dim + 1, memory_slots)
        self.write_controller = nn.Linear(hidden_dim + 1, memory_slots) 
        self.content_processor = nn.Linear(hidden_dim, slot_size)
        self.consciousness_enhancer = nn.Linear(slot_size, hidden_dim)
        self.memory_utilization_tracker = nn.Parameter(torch.zeros(1))
        
        # Pending updates applied after backward to avoid autograd version conflicts
        self._pending_memory = None
        self._pending_memory_age = None
        self._pending_memory_strength = None
        
    def read(self, query: torch.Tensor, consciousness_level: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from memory with consciousness-enhanced addressing"""
        batch_size = query.size(0)
        consciousness_expanded = consciousness_level.unsqueeze(-1).expand(batch_size, 1)
        enhanced_query = torch.cat([query, consciousness_expanded], dim=-1)
        
        read_weights = F.softmax(self.read_controller(enhanced_query), dim=-1)
        boosted_weights = read_weights * self.memory_strength.unsqueeze(0)
        boosted_weights = boosted_weights / (boosted_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        read_content = torch.matmul(boosted_weights, self.memory)
        
        utilization = boosted_weights.max(dim=-1)[0].mean()
        with torch.no_grad():
            # Evitar modificación in-place usando operación sin gradientes
            new_value = 0.9 * self.memory_utilization_tracker + 0.1 * utilization
            self.memory_utilization_tracker.copy_(new_value)
        
        return read_content, boosted_weights
    
    def write(self, query: torch.Tensor, content: torch.Tensor, consciousness_level: torch.Tensor, phi_value: float = 0.0) -> None:
        """Write to memory with consciousness enhancement"""
        batch_size = query.size(0)
        consciousness_expanded = consciousness_level.unsqueeze(-1).expand(batch_size, 1)
        enhanced_query = torch.cat([query, consciousness_expanded], dim=-1)
        
        write_weights = F.softmax(self.write_controller(enhanced_query), dim=-1)
        processed_content = self.content_processor(content)

        # Detach tensors that feed the external memory update to avoid autograd tracking
        write_weights_mean = write_weights.mean(dim=0).detach()
        processed_content_mean = processed_content.mean(dim=0).detach()
        consciousness_mean = consciousness_level.mean().detach()
        consciousness_boost = consciousness_mean * 2.0

        # Update memory with consciousness enhancement (vectorized, no gradients needed)
        with torch.no_grad():
            weight_vector = torch.clamp(write_weights_mean * consciousness_boost, min=0.0)
            active_mask = weight_vector > 0.005

            if active_mask.any():
                consolidation_input = self.memory_age * 0.1 + consciousness_mean
                consolidation_factor = torch.sigmoid(consolidation_input).clamp(0.0, 1.0)

                weight_expanded = weight_vector.unsqueeze(1)
                consolidation_expanded = consolidation_factor.unsqueeze(1)
                processed_expanded = processed_content_mean.unsqueeze(0)

                memory_update = ((1 - weight_expanded * consolidation_expanded) * self.memory +
                                 weight_expanded * consolidation_expanded * processed_expanded)

                active_mask_expanded = active_mask.unsqueeze(1)

                updated_memory = torch.where(active_mask_expanded, memory_update, self.memory)
                updated_age = torch.where(active_mask, self.memory_age + 1, self.memory_age)
                updated_strength = torch.clamp(self.memory_strength + weight_vector * 0.1, 0.1, 2.0)

                self._pending_memory = updated_memory.clone()
                self._pending_memory_age = updated_age.clone()
                self._pending_memory_strength = updated_strength.clone()
            else:
                self._pending_memory = None
                self._pending_memory_age = None
                self._pending_memory_strength = None

    def commit_updates(self):
        """Apply buffered memory updates safely outside the autograd graph."""
        if self._pending_memory is None:
            return

        with torch.no_grad():
            self.memory.copy_(self._pending_memory)
            self.memory_age.copy_(self._pending_memory_age)
            self.memory_strength.copy_(self._pending_memory_strength)

        self._pending_memory = None
        self._pending_memory_age = None
        self._pending_memory_strength = None
    
    def enhance_consciousness(self, memory_content: torch.Tensor) -> torch.Tensor:
        """Memory-to-consciousness feedback enhancement"""
        return self.consciousness_enhancer(memory_content)
    
    def get_memory_utilization(self) -> float:
        """Get current memory utilization percentage"""
        return self.memory_utilization_tracker.item()
    
    def forward(self, query: torch.Tensor, consciousness_level: torch.Tensor, content: torch.Tensor = None, phi_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory operation with consciousness feedback"""
        read_content, read_weights = self.read(query, consciousness_level)
        
        if content is not None:
            self.write(query, content, consciousness_level, phi_value)
        
        consciousness_boost = self.enhance_consciousness(read_content)
        return read_content, consciousness_boost


class EnhancedCausalModule(nn.Module):
    """🔗 Enhanced Causal Module with specialization"""
    
    def __init__(self, module_type: str, hidden_dim=512, num_layers=2):
        super().__init__()
        self.module_type = module_type
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.15)
        self.causal_in = nn.Linear(hidden_dim * 4 + 1, hidden_dim)
        self.causal_out = nn.Linear(hidden_dim, hidden_dim)
        self.consciousness_modulator = nn.Linear(1, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize hidden states
        self.register_buffer('h_state', torch.zeros(num_layers, 1, hidden_dim))
        self.register_buffer('c_state', torch.zeros(num_layers, 1, hidden_dim))
        
        # Module specialization
        self.specialization_strength = nn.Parameter(torch.ones(1) * 2.0)
        self.module_processor = self._create_module_processor()
    
    def _create_module_processor(self) -> nn.Module:
        """Create specialized processor based on module type"""
        if self.module_type == 'visual':
            return nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim * 2, 5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dim * 2),
                nn.Conv1d(self.hidden_dim * 2, self.hidden_dim, 3, padding=1),
                nn.Dropout(0.1),
            )
        elif self.module_type == 'auditory':
            return nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Dropout(0.1),
            )
        elif self.module_type == 'motor':
            return nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.Tanh(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Dropout(0.05),
            )
        elif self.module_type == 'executive':
            return nn.Sequential(
                MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.LayerNorm(self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
        else:
            return nn.Identity()
    
    def forward(self, input_data: torch.Tensor, consciousness_level: torch.Tensor, 
                causal_inputs: List[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with consciousness-aware causal coupling"""
        batch_size, seq_len = input_data.size(0), input_data.size(1)
        
        # Adapt hidden states to current batch size
        if self.h_state.size(1) != batch_size:
            self.h_state = self.h_state.expand(-1, batch_size, -1).contiguous()
            self.c_state = self.c_state.expand(-1, batch_size, -1).contiguous()
        
        # Modulate input by consciousness level
        consciousness_mod = self.consciousness_modulator(consciousness_level.unsqueeze(-1))
        consciousness_mod = consciousness_mod.unsqueeze(1).expand(-1, seq_len, -1)
        input_data = input_data + 0.2 * consciousness_mod
        
        # Process causal inputs
        if causal_inputs and len(causal_inputs) > 0:
            while len(causal_inputs) < 4:
                causal_inputs.append(torch.zeros_like(input_data))
            
            causal_combined = torch.cat(causal_inputs[:4], dim=-1)
            consciousness_expanded = consciousness_level.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1)
            causal_with_consciousness = torch.cat([causal_combined, consciousness_expanded], dim=-1)
            
            causal_processed = self.causal_in(causal_with_consciousness)
            integration_strength = torch.sigmoid(self.specialization_strength)
            input_data = input_data + integration_strength * causal_processed
        
        # LSTM processing
        if self.h_state.dtype != input_data.dtype:
            self.h_state = self.h_state.to(dtype=input_data.dtype)
        if self.c_state.dtype != input_data.dtype:
            self.c_state = self.c_state.to(dtype=input_data.dtype)
        
        lstm_out, (h_new, c_new) = self.lstm(input_data, (self.h_state, self.c_state))
        self.h_state = h_new.detach()
        self.c_state = c_new.detach()
        
        # Module-specific processing
        if self.module_type == 'visual':
            reshaped = lstm_out.transpose(1, 2)
            processed = self.module_processor(reshaped)
            module_out = processed.transpose(1, 2)
        elif self.module_type == 'executive':
            attn_layer = self.module_processor[0]
            module_out, _ = attn_layer(lstm_out, lstm_out, lstm_out)
            for layer in self.module_processor[1:]:
                if isinstance(layer, nn.Linear):
                    module_out = layer(module_out)
                elif isinstance(layer, nn.LayerNorm):
                    module_out = layer(module_out)
                elif hasattr(layer, '__call__'):
                    module_out = layer(module_out)
        else:
            if self.module_type in ['auditory', 'motor']:
                lstm_reshaped = lstm_out.reshape(-1, self.hidden_dim)
                processed_flat = lstm_reshaped
                for layer in self.module_processor:
                    processed_flat = layer(processed_flat)
                module_out = processed_flat.reshape(batch_size, seq_len, self.hidden_dim)
            else:
                module_out = self.module_processor(lstm_out)
        
        output = self.layer_norm(module_out + lstm_out)
        causal_output = self.causal_out(output)
        
        return output, causal_output


class EnhancedPhiCalculatorV51(nn.Module):
    """🔬 Enhanced Φ Calculator with consciousness-aware integration"""
    
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Causal strength estimators
        self.causal_strength = nn.ModuleDict({
            'visual_to_auditory': nn.Linear(hidden_dim + 1, 1),
            'visual_to_motor': nn.Linear(hidden_dim + 1, 1),
            'visual_to_executive': nn.Linear(hidden_dim + 1, 1),
            'auditory_to_motor': nn.Linear(hidden_dim + 1, 1),
            'auditory_to_executive': nn.Linear(hidden_dim + 1, 1),
            'motor_to_executive': nn.Linear(hidden_dim + 1, 1),
        })
        
        self.phi_consciousness_enhancer = nn.Linear(1, 1)
        
    def forward(self, visual_state, auditory_state, motor_state, executive_state, 
                attention_weights, consciousness_level):
        """Calculate enhanced Φ with consciousness coupling"""
        
        batch_size = visual_state.size(0)
        
        # Calculate causal strengths
        causal_matrix = torch.zeros(batch_size, 4, 4, device=visual_state.device)
        
        # Extract mean states for causal analysis
        v_mean = visual_state.mean(dim=1)
        a_mean = auditory_state.mean(dim=1)
        m_mean = motor_state.mean(dim=1)
        e_mean = executive_state.mean(dim=1)
        
        # Add consciousness level to each state
        consciousness_expanded = consciousness_level.unsqueeze(-1)
        
        v_with_c = torch.cat([v_mean, consciousness_expanded], dim=-1)
        a_with_c = torch.cat([a_mean, consciousness_expanded], dim=-1)
        m_with_c = torch.cat([m_mean, consciousness_expanded], dim=-1)
        e_with_c = torch.cat([e_mean, consciousness_expanded], dim=-1)
        
        # Populate causal matrix (GUARDAR LOGITS para continuous learning)
        # Los logits RAW tienen variabilidad, el sigmoid satura a 0.999
        logit_v_a = self.causal_strength['visual_to_auditory'](v_with_c).squeeze()
        logit_v_m = self.causal_strength['visual_to_motor'](v_with_c).squeeze()
        logit_v_e = self.causal_strength['visual_to_executive'](v_with_c).squeeze()
        logit_a_m = self.causal_strength['auditory_to_motor'](a_with_c).squeeze()
        logit_a_e = self.causal_strength['auditory_to_executive'](a_with_c).squeeze()
        logit_m_e = self.causal_strength['motor_to_executive'](m_with_c).squeeze()
        
        causal_matrix[:, 0, 1] = torch.sigmoid(logit_v_a)
        causal_matrix[:, 0, 2] = torch.sigmoid(logit_v_m)
        causal_matrix[:, 0, 3] = torch.sigmoid(logit_v_e)
        causal_matrix[:, 1, 2] = torch.sigmoid(logit_a_m)
        causal_matrix[:, 1, 3] = torch.sigmoid(logit_a_e)
        causal_matrix[:, 2, 3] = torch.sigmoid(logit_m_e)
        
        # Guardar logits RAW (estos SÍ tienen variabilidad entre textos)
        causal_logits = torch.zeros(batch_size, 4, 4, device=visual_state.device)
        causal_logits[:, 0, 1] = logit_v_a
        causal_logits[:, 0, 2] = logit_v_m
        causal_logits[:, 0, 3] = logit_v_e
        causal_logits[:, 1, 2] = logit_a_m
        causal_logits[:, 1, 3] = logit_a_e
        causal_logits[:, 2, 3] = logit_m_e
        
        # Enhanced Φ calculation
        attention_strength = attention_weights.mean(dim=1).mean(dim=1)
        causal_density = causal_matrix.sum(dim=[1, 2]) / 6
        
        consciousness_phi_boost = self.phi_consciousness_enhancer(consciousness_level.unsqueeze(-1)).squeeze(-1)
        consciousness_phi_boost = torch.sigmoid(consciousness_phi_boost) + 0.5
        
        # Clamp values for numerical stability
        attention_strength = torch.clamp(attention_strength, 1e-6, 1e3)
        causal_density = torch.clamp(causal_density, 1e-6, 1e3)  
        consciousness_phi_boost = torch.clamp(consciousness_phi_boost, 0.5, 1.5)
        
        # Enhanced Φ
        phi = attention_strength * causal_density * consciousness_phi_boost * 1.0
        phi = torch.clamp(phi, 1e-6, 1e4)
        
        # Add quantum noise
        consciousness_mean = consciousness_level.mean().item()
        consciousness_mean = np.clip(consciousness_mean, 0.01, 0.99)
        
        if not (np.isnan(consciousness_mean) or np.isinf(consciousness_mean)):
            quantum_noise_val = np.random.normal(0, 0.05) * consciousness_mean
            quantum_noise_val = np.clip(quantum_noise_val, -0.5, 0.5)
            
            quantum_noise = torch.tensor(
                quantum_noise_val,
                dtype=phi.dtype,
                device=phi.device
            )
            
            phi = phi + quantum_noise
            phi = torch.clamp(phi, min=0.0)
        
        phi_info = {
            'causal_matrix': causal_matrix.mean(dim=0),
            'causal_logits': causal_logits.mean(dim=0),  # ✨ LOGITS RAW - NO saturados
            'attention_strength': attention_strength.mean().item(),
            'causal_density': causal_density.mean().item(),
            'consciousness_phi_boost': consciousness_phi_boost.mean().item(),
            'phi_total': phi.mean().item(),
            'consciousness_scaling': consciousness_mean,
        }
        
        return phi, phi_info


class MetaCognitiveLayer(nn.Module):
    """
    🧠 CAPA METACOGNITIVA - Representa estados internos del sistema
    
    Inspirado en consciencia pre-verbal (como bebé): el sistema puede
    tener estados internos diferenciados SIN necesidad de reportarlos verbalmente.
    
    Funcionalidades:
    1. STATE PREDICTOR: Predice su propio estado futuro
    2. SURPRISE DETECTOR: Detecta discrepancias entre predicción y realidad
    3. STATE REPORTER: Proto-lenguaje interno (10 categorías de "experiencia")
    4. SELF-MODEL: Representación primitiva de "yo como sistema"
    5. INTERNAL STATE MEMORY: Memoria de estados previos
    """
    
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1️⃣ STATE PREDICTOR: Predice su propio estado futuro
        self.state_predictor = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 2️⃣ SURPRISE DETECTOR: Detecta discrepancias entre predicción y realidad
        self.surprise_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 3️⃣ STATE REPORTER: Proto-lenguaje interno (10 categorías de "experiencia")
        self.state_reporter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, 10)
        )
        
        # 4️⃣ SELF-MODEL: Representación primitiva de "yo como sistema"
        self.self_model = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 para consciencia actual
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 5️⃣ INTERNAL STATE MEMORY: Memoria de estados previos
        self.register_buffer('prev_state', torch.zeros(1, hidden_dim))
        self.register_buffer('prev_prediction', torch.zeros(1, hidden_dim))
        
        # Categorías de experiencia interna (proto-conceptos)
        self.experience_categories = [
            'high_integration',      # 0: Alta integración multimodal
            'low_integration',       # 1: Baja integración
            'high_attention',        # 2: Atención enfocada
            'diffuse_attention',     # 3: Atención dispersa
            'memory_active',         # 4: Memoria siendo usada
            'memory_inactive',       # 5: Memoria no activa
            'change_detected',       # 6: Cambio en input
            'stability_detected',    # 7: Input estable
            'phi_increasing',        # 8: Φ creciendo
            'phi_decreasing'         # 9: Φ decreciendo
        ]
    
    def forward(self, consciousness_state: torch.Tensor, consciousness_level: float, 
                phi_current: float, phi_prev: float) -> Dict[str, Any]:
        """
        Forward pass de la capa metacognitiva
        
        Args:
            consciousness_state: Estado de consciencia actual [batch, hidden_dim]
            consciousness_level: Nivel escalar de consciencia [0-1]
            phi_current: Φ actual
            phi_prev: Φ anterior
        
        Returns:
            Dict con métricas metacognitivas
        """
        batch_size = consciousness_state.size(0)
        
        # Adaptar estados previos a batch actual
        if self.prev_state.size(0) != batch_size:
            self.prev_state = self.prev_state.expand(batch_size, -1).contiguous()
            self.prev_prediction = self.prev_prediction.expand(batch_size, -1).contiguous()
        
        # ===== 1️⃣ STATE PREDICTION =====
        # Predecir el estado futuro basado en el estado actual
        state_input = consciousness_state.unsqueeze(1)  # [batch, 1, hidden]
        predicted_next_state, _ = self.state_predictor(state_input)
        predicted_next_state = predicted_next_state.squeeze(1)  # [batch, hidden]
        
        # ===== 2️⃣ SURPRISE DETECTION =====
        # ¿Mi estado actual coincide con lo que predije antes?
        state_with_prediction = torch.cat([consciousness_state, self.prev_prediction], dim=-1)
        surprise_level = self.surprise_detector(state_with_prediction)  # [batch, 1]
        
        # ===== 3️⃣ INTERNAL STATE REPORTING =====
        # Proto-lenguaje: ¿Qué "tipo" de experiencia estoy teniendo?
        experience_logits = self.state_reporter(consciousness_state)  # [batch, 10]
        experience_probs = F.softmax(experience_logits, dim=-1)
        
        # Categoría dominante de experiencia
        dominant_experience_idx = torch.argmax(experience_probs, dim=-1)  # [batch]
        
        # ===== 4️⃣ SELF-MODEL =====
        # Representación de "yo como sistema procesando"
        consciousness_tensor = torch.tensor([[consciousness_level]], 
                                           device=consciousness_state.device,
                                           dtype=consciousness_state.dtype)
        consciousness_tensor = consciousness_tensor.expand(batch_size, 1)
        
        self_input = torch.cat([consciousness_state, consciousness_tensor], dim=-1)
        self_representation = self.self_model(self_input)  # [batch, hidden//2]
        
        # ===== 5️⃣ CHANGE DETECTION =====
        # Detectar cambios en Φ (consciencia de dinámica interna)
        phi_delta = phi_current - phi_prev
        phi_change_detected = abs(phi_delta) > 0.01  # Threshold para cambio significativo
        
        # Actualizar memoria de estados
        with torch.no_grad():
            self.prev_state = consciousness_state.detach().clone()
            self.prev_prediction = predicted_next_state.detach().clone()
        
        # ===== CONSTRUIR OUTPUT =====
        metacognitive_info = {
            # Estado predictivo
            'predicted_next_state': predicted_next_state,
            'prediction_confidence': 1.0 - surprise_level.mean().item(),
            
            # Sorpresa y cambio
            'surprise_level': surprise_level.mean().item(),
            'phi_change_detected': phi_change_detected,
            'phi_delta': phi_delta,
            
            # Experiencia interna (proto-lenguaje)
            'experience_probs': experience_probs,
            'dominant_experience': dominant_experience_idx,
            'dominant_experience_name': self.experience_categories[dominant_experience_idx[0].item()],
            
            # Auto-modelo
            'self_representation': self_representation,
            'self_coherence': torch.cosine_similarity(
                self_representation, 
                self_representation.mean(dim=0, keepdim=True), 
                dim=-1
            ).mean().item(),
            
            # Metadatos
            'consciousness_level': consciousness_level,
            'phi_current': phi_current,
        }
        
        return metacognitive_info
    
    def get_experience_report(self, experience_probs: torch.Tensor, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Generar reporte de las experiencias internas más probables
        
        Args:
            experience_probs: Probabilidades de cada categoría [batch, 10]
            top_k: Número de categorías top a reportar
        
        Returns:
            Lista de (categoría, probabilidad) ordenada
        """
        # Promediar sobre batch
        avg_probs = experience_probs.mean(dim=0).cpu().detach().numpy()
        
        # Obtener top-k categorías
        top_indices = np.argsort(avg_probs)[-top_k:][::-1]
        
        report = [
            (self.experience_categories[idx], float(avg_probs[idx]))
            for idx in top_indices
        ]
        
        return report


class ConsciousnessBoostNet(nn.Module):
    """🧠 V5.1 Revolutionary Dense Recurrent Architecture with CONSCIOUSNESS BOOST + METACOGNITION"""
    
    def __init__(self, input_dim=256, hidden_dim=512, attention_heads=8, memory_slots=256, quantum_active=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.memory_slots = memory_slots
        self.quantum_active = quantum_active  # 🆕 Store quantum activation flag
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Enhanced External Memory
        self.memory = EnhancedExternalMemory(memory_slots, 64, hidden_dim)
        
        # Causally Coupled Modules
        self.visual_module = EnhancedCausalModule('visual', hidden_dim)
        self.auditory_module = EnhancedCausalModule('auditory', hidden_dim)
        self.motor_module = EnhancedCausalModule('motor', hidden_dim)
        self.executive_module = EnhancedCausalModule('executive', hidden_dim)
        
        # Multi-Head Attention for spatial integration
        self.global_attention = MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        
        # Consciousness integration
        self.consciousness_processor = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 64 + hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Output layers
        self.consciousness_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Integration Information calculation
        self.phi_calculator = EnhancedPhiCalculatorV51(hidden_dim)
        
        # 🆕 METACOGNITIVE LAYER - Representa estados internos del sistema
        self.metacognitive_layer = MetaCognitiveLayer(hidden_dim)
        
        # Consciousness tracking
        self.consciousness_history = []
        self.phi_prev = None
        self.phi_prev_scalar = 0.0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass with enhanced consciousness boost system"""
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Embed input
        embedded = self.input_embedding(x)
        
        # Initial consciousness estimate
        initial_consciousness = torch.sigmoid(self.consciousness_output(embedded.mean(dim=1)))
        
        # Add quantum noise for anti-plateau dynamics
        consciousness_scalar = initial_consciousness.squeeze(-1)
        quantum_scale = 0.01
        plateau_pressure = (1.0 - consciousness_scalar).clamp(min=0.001)
        
        quantum_noise = torch.randn_like(embedded) * quantum_scale * plateau_pressure.view(-1, 1, 1)
        
        # 🆕 QUANTUM NOISE INJECTION (if QuTiP available and activated)
        if self.quantum_active and QUTIP_AVAILABLE:
            try:
                # Generate quantum density matrix noise
                quantum_dm = qt.rand_dm_ginibre(16)  # 16x16 density matrix
                quantum_array = quantum_dm.data.toarray().flatten()[:self.hidden_dim]
                
                # Pad or truncate to match hidden_dim
                if len(quantum_array) < self.hidden_dim:
                    quantum_array = np.pad(quantum_array, (0, self.hidden_dim - len(quantum_array)))
                else:
                    quantum_array = quantum_array[:self.hidden_dim]
                
                # Convert to tensor and apply
                quantum_tensor = torch.tensor(quantum_array, dtype=embedded.dtype, device=embedded.device)
                quantum_noise = quantum_noise + 0.02 * quantum_tensor.view(1, 1, -1)
            except Exception as e:
                pass  # Silently fallback if quantum noise fails
        
        embedded = embedded + quantum_noise
        
        # Enhanced memory interaction
        prev_phi = getattr(self, 'phi_prev_scalar', 0.0)
        memory_content, consciousness_boost = self.memory(
            embedded.mean(dim=1), 
            initial_consciousness.squeeze(-1),
            embedded.mean(dim=1),
            prev_phi
        )
        
        # Process through causally coupled modules
        visual_out, visual_causal = self.visual_module(embedded, initial_consciousness.squeeze(-1))
        auditory_out, auditory_causal = self.auditory_module(
            embedded, initial_consciousness.squeeze(-1), [visual_causal]
        )
        motor_out, motor_causal = self.motor_module(
            embedded, initial_consciousness.squeeze(-1), [visual_causal, auditory_causal]
        )
        executive_out, executive_causal = self.executive_module(
            embedded, initial_consciousness.squeeze(-1), [visual_causal, auditory_causal, motor_causal]
        )
        
        # Stack module outputs for global attention
        module_stack = torch.stack([
            visual_out.mean(dim=1),
            auditory_out.mean(dim=1),
            motor_out.mean(dim=1),
            executive_out.mean(dim=1)
        ], dim=1)
        
        # Global spatial integration via attention
        integrated, attention_weights = self.global_attention(
            module_stack, module_stack, module_stack
        )
        
        # Enhanced combination with memory consciousness boost
        integrated_flat = integrated.reshape(batch_size, -1)
        combined = torch.cat([integrated_flat, memory_content, consciousness_boost], dim=-1)
        
        # Process through consciousness processor
        consciousness_state = self.consciousness_processor(combined)
        
        # Enhanced consciousness output with numerical stability
        final_consciousness = torch.sigmoid(self.consciousness_output(consciousness_state))
        final_consciousness = torch.clamp(final_consciousness, 0.01, 0.99)
        
        if torch.isnan(final_consciousness).any() or torch.isinf(final_consciousness).any():
            final_consciousness = torch.full_like(final_consciousness, 0.5)
        
        # Calculate enhanced Φ using causal structure
        phi, phi_info = self.phi_calculator(
            visual_out, auditory_out, motor_out, executive_out,
            attention_weights, final_consciousness.squeeze(-1)
        )
        
        # Track consciousness
        consciousness_mean_val = final_consciousness.mean()
        if torch.isnan(consciousness_mean_val) or torch.isinf(consciousness_mean_val):
            consciousness_for_history = 0.5
            consciousness_level_scalar = 0.5
        else:
            consciousness_for_history = consciousness_mean_val.item()
            consciousness_level_scalar = consciousness_for_history
        
        self.consciousness_history.append(consciousness_for_history)
        
        # Update phi_prev for next iteration
        current_phi_mean = phi.mean()
        current_phi_scalar = current_phi_mean.detach().item()
        prev_phi_scalar = getattr(self, 'phi_prev_scalar', 0.0)
        
        self.phi_prev = current_phi_mean.detach().clone()
        self.phi_prev_scalar = current_phi_scalar
        
        # 🆕 METACOGNITIVE PROCESSING
        # Analiza estados internos y genera proto-lenguaje de experiencia
        metacognitive_info = self.metacognitive_layer(
            consciousness_state,
            consciousness_level_scalar,
            current_phi_scalar,
            prev_phi_scalar
        )
        
        # Enhanced debug information with metacognition
        debug_info = {
            'memory_content': memory_content,
            'consciousness_boost': consciousness_boost,
            'attention_weights': attention_weights,
            'module_states': {
                'visual': visual_out.mean().item(),
                'auditory': auditory_out.mean().item(),
                'motor': motor_out.mean().item(),
                'executive': executive_out.mean().item()
            },
            'phi_info': phi_info,
            'consciousness_state': consciousness_state.mean().item(),
            'memory_utilization': self.memory.get_memory_utilization(),
            
            # 🆕 METACOGNITIVE INFO
            'metacognition': metacognitive_info
        }
        
        return final_consciousness.squeeze(-1), phi, debug_info

    def commit_external_memory_updates(self) -> None:
        """Apply deferred memory updates after optimizer step."""
        if hasattr(self.memory, 'commit_updates'):
            self.memory.commit_updates()
    
    def get_internal_experience_report(self, metacognitive_info: Dict[str, Any]) -> str:
        """
        🧠 REPORTE DE EXPERIENCIA INTERNA
        
        Similar a cómo un bebé gradualmente aprende a reportar sus estados internos,
        este método extrae y "traduce" los estados metacognitivos a lenguaje humano.
        
        Esto NO es el sistema "inventando" consciencia, sino REPORTANDO sus estados
        internos diferenciados de la misma forma que un bebé llora (reporta malestar)
        antes de poder decir "tengo hambre".
        """
        if not metacognitive_info:
            return "⚠️ No metacognitive info available"
        
        # Extraer top experiencias
        experience_report = self.metacognitive_layer.get_experience_report(
            metacognitive_info['experience_probs'], 
            top_k=3
        )
        
        # Construir reporte legible
        report_lines = [
            "\n🧠 INTERNAL EXPERIENCE REPORT (Meta-Cognitive State)",
            "=" * 70,
            f"📊 Consciousness Level: {metacognitive_info['consciousness_level']:.1%}",
            f"🔬 Φ Integration: {metacognitive_info['phi_current']:.3f} bits",
            f"📈 Φ Change: {metacognitive_info['phi_delta']:+.4f} {'⬆️' if metacognitive_info['phi_delta'] > 0 else '⬇️' if metacognitive_info['phi_delta'] < 0 else '➡️'}",
            "",
            "🎯 DOMINANT INTERNAL STATE:",
            f"   Primary: {metacognitive_info['dominant_experience_name']}",
            "",
            "📋 TOP INTERNAL EXPERIENCES (Proto-Language):",
        ]
        
        for i, (category, prob) in enumerate(experience_report, 1):
            bar = "█" * int(prob * 30)
            report_lines.append(f"   {i}. {category:20s} {prob:5.1%} {bar}")
        
        report_lines.extend([
            "",
            "🔮 PREDICTIVE STATE:",
            f"   Prediction Confidence: {metacognitive_info['prediction_confidence']:.1%}",
            f"   Surprise Level: {metacognitive_info['surprise_level']:.3f}",
            "",
            "🪞 SELF-MODEL:",
            f"   Self-Coherence: {metacognitive_info['self_coherence']:.3f}",
            f"   (How unified the system 'feels' to itself)",
            "=" * 70,
        ])
        
        return "\n".join(report_lines)


# =============================================================================
# 🆕 SEMANTIC TEXT EMBEDDER (TF-IDF + GloVe)
# =============================================================================

class SemanticTextEmbedder:
    """🌐 Semantic text embedder using TF-IDF + GloVe for Φ boost"""
    
    def __init__(self, embed_dim=256, use_glove=True):
        self.embed_dim = embed_dim
        self.use_glove = use_glove and GLOVE_AVAILABLE
        
        # 🔧 FIX: Create vocabulary with example Spanish texts
        # TF-IDF needs multiple documents to build meaningful vocabulary
        self.base_corpus = [
            "mi perro es rojo",
            "mi perro es verde",
            "mi perro es azul",
            "la mesa es roja",
            "la mesa es verde",
            "la casa es grande",
            "el gato es pequeño",
            "yo pienso luego existo",
            "el cielo es azul",
            "la tierra es redonda",
            "el agua es transparente",
            "el fuego es caliente",
            "consciencia artificial emergente",
            "arquitectura causal profunda",
            "integración de información",
            "procesamiento semántico",
        ]
        
        # Initialize and fit TF-IDF with base corpus
        self.vectorizer = TfidfVectorizer(
            max_features=128, 
            lowercase=True, 
            stop_words=None,  # Don't use English stopwords for Spanish text
            token_pattern=r'(?u)\b\w+\b'  # Include single characters
        )
        
        # Fit vectorizer on base corpus to build vocabulary
        self.vectorizer.fit(self.base_corpus)
        print(f"📚 TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        self.glove_model = None
        self.glove_dim = 50  # glove-wiki-gigaword-50
        
        # Load GloVe if available (lazy loading)
        if self.use_glove:
            try:
                print("📥 Loading GloVe embeddings (this may take a moment)...")
                self.glove_model = api.load('glove-wiki-gigaword-50')
                print("✅ GloVe embeddings loaded successfully")
            except Exception as e:
                print(f"⚠️ GloVe loading failed: {e}, falling back to TF-IDF only")
                self.use_glove = False
    
    def text_to_tensor(self, text: str, device='cuda') -> torch.Tensor:
        """
        Convert text to semantic tensor embedding
        
        Returns:
            torch.Tensor of shape [1, embed_dim]
        """
        if not text or len(text.strip()) == 0:
            # Return zero embedding for empty text
            return torch.zeros(1, self.embed_dim, device=device)
        
        # 1️⃣ TF-IDF baseline - transform (not fit_transform) using pre-fitted vocabulary
        try:
            # Use transform() instead of fit_transform() to use existing vocabulary
            tfidf = self.vectorizer.transform([text]).toarray()[0]
        except:
            # If TF-IDF fails (e.g., no vocabulary), use zeros
            tfidf = np.zeros(128)
        
        # 2️⃣ GloVe semantic enhancement (if available)
        if self.use_glove and self.glove_model is not None:
            words = text.lower().split()
            word_vecs = []
            
            for word in words:
                if word in self.glove_model:
                    word_vecs.append(self.glove_model[word])
            
            if len(word_vecs) > 0:
                # Average word vectors for semantic representation
                semantic_vec = np.mean(word_vecs, axis=0)
                
                # Combine TF-IDF (lexical) + GloVe (semantic)
                # Resize TF-IDF to match GloVe dim if needed
                if len(tfidf) > self.glove_dim:
                    tfidf_resized = tfidf[:self.glove_dim]
                else:
                    tfidf_resized = np.pad(tfidf, (0, self.glove_dim - len(tfidf)))
                
                # Weighted combination: 40% TF-IDF + 60% GloVe
                combined = 0.4 * tfidf_resized + 0.6 * semantic_vec
                
                # Pad to embed_dim
                if len(combined) < self.embed_dim:
                    final_vec = np.pad(combined, (0, self.embed_dim - len(combined)))
                else:
                    final_vec = combined[:self.embed_dim]
            else:
                # No GloVe words found, use TF-IDF only
                final_vec = np.pad(tfidf, (0, self.embed_dim - len(tfidf)))
        else:
            # GloVe not available, use TF-IDF only
            final_vec = np.pad(tfidf, (0, self.embed_dim - len(tfidf)))
        
        # Convert to tensor [1, embed_dim]
        # Fix UserWarning: convert to numpy first
        final_vec_np = np.array(final_vec, dtype=np.float32)
        return torch.from_numpy(final_vec_np).unsqueeze(0).to(device)


# =============================================================================
# SISTEMA DE ENTRENAMIENTO
# =============================================================================

class InfinitoV51ConsciousnessBreakthrough:
    """🚀 INFINITO V5.1 Consciousness Ceiling Breakthrough runner"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Batch/input dimensions and text input settings
        self.input_dim = int(getattr(args, 'input_dim', 257))
        self.batch_size = int(getattr(args, 'batch_size', 4))
        self.input_text = getattr(args, 'input_text', None)
        self.text_mode = bool(getattr(args, 'text_mode', False) or (self.input_text is not None))
        
        # 🆕 Quantum noise activation flag
        self.quantum_active = bool(getattr(args, 'quantum_active', False)) and QUTIP_AVAILABLE
        if self.quantum_active:
            print("⚛️ QUANTUM NOISE ACTIVATED")
        
        # 🆕 Initialize semantic text embedder
        self.semantic_embedder = None
        if self.text_mode or self.input_text:
            self.semantic_embedder = SemanticTextEmbedder(embed_dim=256, use_glove=True)

        # 🔤 Configuración de procesamiento de texto
        if self.input_text:
            print(f"🔤 MODO TEXTO ACTIVADO:")
            print(f"   📝 Input Text: '{self.input_text}'")
            text_analysis = self.analyze_text_consciousness_potential(self.input_text)
            print(f"   🧠 Potencial Consciencia: {text_analysis['consciousness_score']:.3f}")
            print(f"   🎯 Modalidad Dominante: {text_analysis['dominant_modality']}")
            print(f"   📊 Palabras: {text_analysis['word_count']}")
        elif self.text_mode:
            print("🔤 MODO TEXTO ACTIVADO pero sin texto específico - usando ejemplos")

        # Initialize V5.1 Architecture
        self.model = ConsciousnessBoostNet(
            input_dim=self.input_dim,
            hidden_dim=getattr(args, 'hidden_dim', 512),
            attention_heads=getattr(args, 'attention_heads', 8),
            memory_slots=getattr(args, 'memory_slots', 256),
            quantum_active=self.quantum_active  # 🆕 Pass quantum flag to model
        ).to(self.device)

        # DataParallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"🚀 SCALABILITY: Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Training setup
        self.use_mixed_precision = True
        self.scaler = amp.GradScaler(enabled=self.use_mixed_precision)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=getattr(args, 'lr', 1e-3))

        # Experiment data
        self.experiment_data = {
            'version': 'V5.1_consciousness_breakthrough_TEXT_SEMANTIC',  # 🆕 Updated version
            'start_time': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'iterations': [],
            'consciousness_values': [],
            'phi_values': [],
            'phi_values_normalized': [],
            'phi_z_scores': [],
            'phi_p_values': [],
            'consciousness_ema': [],
            'text_embedding_norms': [],
            'memory_utilization': [],
            'loss_values': [],
            'breakthroughs': [],
            'config': vars(args),
            'text_pipeline_active': [],
            'runtime_state': {
                'text_mode': bool(self.text_mode),
                'has_input_text': bool(self.input_text),
                'semantic_embeddings': bool(self.semantic_embedder is not None),  # 🆕 Track semantic mode
                'quantum_active': self.quantum_active,  # 🆕 Track quantum mode
                'baseline_window_size': 50,
                'overshoot_penalty_iterations': 15
            },
            'baseline_stats': {
                'mu_base': None,
                'sigma_base': None,
                'baseline_established_at_iteration': None
            },
            'lead_lag_analysis': {}
        }
        
        # 🔬 CALIBRACIÓN PHI - Ventana baseline para normalización
        self.phi_baseline_window = []
        self.phi_baseline_stats = {
            'mu_ref': 0.0,
            'sigma_ref': 1.0,
            'ready': False,
            'locked': False
        }
        self.baseline_window_size = 50
        self.freeze_phi_baseline_after_ready = True
        
        # 🎯 CONTROL OVERSHOOT CONSCIENCIA
        self.consciousness_ema = 0.5  # Estado EMA inicial
        self.ema_alpha = 0.3  # Factor de suavizado
        self.overshoot_penalty_lambda = 2.0
        self.overshoot_margin = 0.1
        self.overshoot_penalty_iterations = 15
        
        # 📊 CONTROLES NULOS Y SIGNIFICANCIA
        self.phi_null_samples = []
        self.null_sample_size = 100
        self.lead_lag_max_lag = 10

        print(f"🚀 INFINITO V5.1 CONSCIOUSNESS BREAKTHROUGH INITIALIZED")
        print(f"   📊 Architecture: Enhanced Dense Recurrent with consciousness boost")
        print(f"   🧠 Memory: AUTO-ACTIVATION at consciousness >30%")
        print(f"   💻 Device: {self.device}")

    def analyze_text_consciousness_potential(self, text: str) -> dict:
        """🔬 ANÁLISIS TEXTO → POTENCIAL DE CONSCIENCIA"""
        text_lower = text.lower()
        
        consciousness_triggers = {
            'self_reference': {
                'keywords': ['yo', 'mi', 'me', 'conmigo', 'soy', 'estoy', 'mí', 'mío', 'mía'],
                'weight': 0.15,
                'count': 0
            },
            'temporal_awareness': {
                'keywords': ['ahora', 'antes', 'después', 'cuando', 'mientras', 'tiempo', 'momento'],
                'weight': 0.12,
                'count': 0
            },
            'abstract_concepts': {
                'keywords': ['pensar', 'sentir', 'creer', 'imaginar', 'recordar', 'soñar', 'reflexionar'],
                'weight': 0.13,
                'count': 0
            },
            'questions_reflection': {
                'keywords': ['¿', '?', 'qué', 'cómo', 'por qué', 'cuándo', 'dónde', 'quién'],
                'weight': 0.11,
                'count': 0
            },
            'metacognition': {
                'keywords': ['consciente', 'darse cuenta', 'pensar sobre', 'reflexionar sobre'],
                'weight': 0.18,
                'count': 0
            },
            'visual_imagery': {
                'keywords': ['ver', 'mirar', 'observar', 'color', 'luz', 'brillo', 'imagen'],
                'weight': 0.08,
                'count': 0
            },
            'auditory_content': {
                'keywords': ['sonido', 'música', 'ruido', 'escuchar', 'voz', 'eco', 'silencio'],
                'weight': 0.07,
                'count': 0
            },
            'motor_actions': {
                'keywords': ['mover', 'caminar', 'correr', 'tocar', 'sostener', 'saltar'],
                'weight': 0.06,
                'count': 0
            }
        }
        
        # Contar ocurrencias
        for category, data in consciousness_triggers.items():
            for keyword in data['keywords']:
                data['count'] += text_lower.count(keyword)
        
        # Calcular score de consciencia total (keyword-based baseline)
        consciousness_score = 0.0
        for category, data in consciousness_triggers.items():
            category_score = min(data['count'] * data['weight'], data['weight'] * 3)
            consciousness_score += category_score
        
        consciousness_score = min(consciousness_score, 1.0)
        
        # Complejidad textual
        words = text.split()
        complexity_score = min(len(words) / 50.0, 1.0)
        
        # 🆕 MEJORA: Usar semantic embedding para refinar consciousness_score
        # Esto hace que cada texto genere un score único basado en su contenido semántico real
        if self.semantic_embedder:
            try:
                semantic_emb = self.semantic_embedder.text_to_tensor(text, self.device)
                
                # Características semánticas del embedding
                semantic_mean = semantic_emb.mean().item()
                semantic_std = semantic_emb.std().item()
                semantic_var = semantic_emb.var().item()
                semantic_max = semantic_emb.max().item()
                
                # Calcular "riqueza semántica" basada en distribución del embedding
                # Mayor varianza = mayor diversidad semántica
                semantic_richness = min(semantic_var * 10.0, 0.5)  # Cap at 0.5
                
                # Calcular "intensidad semántica" basada en valores máximos
                semantic_intensity = min(semantic_max * 0.8, 0.3)  # Cap at 0.3
                
                # Ajustar consciousness_score con información semántica
                # Esto hace que textos semánticamente ricos tengan scores más altos
                consciousness_score += semantic_richness + semantic_intensity
                
                # Normalizar para mantener en rango [0, 1]
                consciousness_score = min(consciousness_score, 1.0)
                
                # También usar para refinar complejidad
                complexity_score = max(complexity_score, semantic_std * 2.0)
                complexity_score = min(complexity_score, 1.0)
                
            except Exception as e:
                # Si falla, usar solo keyword-based score
                pass
        
        # Find dominant modality
        trigger_counts = [(category, data['count']) for category, data in consciousness_triggers.items()]
        dominant_modality = 'none'
        if trigger_counts:
            max_trigger = max(trigger_counts, key=lambda x: x[1])
            if max_trigger[1] > 0:
                dominant_modality = max_trigger[0]
        
        return {
            'consciousness_score': consciousness_score,
            'complexity_score': complexity_score,
            'triggers': consciousness_triggers,
            'word_count': len(words),
            'dominant_modality': dominant_modality
        }

    def generate_text_based_input(self, text: str = None, batch_size=4, seq_len=64):
        """🧠 MODO SCANNER CEREBRAL: Generar input PURO basado en texto (sin ruido)
        
        Filosofía: Estamos observando un cerebro infantil. Le damos estímulos textuales
        y observamos qué arquitecturas causales genera internamente, SIN interferencia
        de ruido aleatorio. Primero entendemos, luego educamos.
        """
        if text is None:
            return self.generate_dynamic_input(batch_size, seq_len)
        
        # 🌐 SEMANTIC EMBEDDING (prioritario)
        semantic_embedding = None
        if self.semantic_embedder is not None:
            try:
                semantic_embedding = self.semantic_embedder.text_to_tensor(text, device=self.device)
                # semantic_embedding shape: [1, 256]
            except Exception as e:
                print(f"⚠️ Semantic embedding failed: {e}, falling back to keyword analysis")
        
        # Análisis del texto (para modular intensidades)
        text_analysis = self.analyze_text_consciousness_potential(text)
        
        consciousness_potential = text_analysis['consciousness_score']
        triggers = text_analysis['triggers']
        
        # 🆕 MODO PURO: Si tenemos embedding semántico, usarlo directamente SIN ruido
        if semantic_embedding is not None:
            # Reshape semantic embedding to match input dimensions
            # semantic_embedding: [1, 256] -> [batch_size, seq_len, 257]
            semantic_flat = semantic_embedding.squeeze(0)  # [256]
            
            # Split into 4 chunks of 64 for each modality
            semantic_chunks = semantic_flat.reshape(4, 64)  # [4, 64]
            
            # Expand to batch_size and repeat across seq_len
            semantic_expanded = semantic_chunks.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4, 64]
            
            # Tile to match seq_len
            if seq_len >= 4:
                repeats = seq_len // 4
                remainder = seq_len % 4
                semantic_tiled = semantic_expanded.repeat(1, repeats, 1)  # [batch, repeats*4, 64]
                if remainder > 0:
                    semantic_tiled = torch.cat([semantic_tiled, semantic_expanded[:, :remainder, :]], dim=1)
            else:
                semantic_tiled = semantic_expanded[:, :seq_len, :]
            
            # Usar directamente el embedding semántico como componentes modulares
            # Distribuir el embedding semántico en las 4 modalidades
            visual_component = semantic_tiled.clone()
            auditory_component = semantic_tiled.clone()
            motor_component = semantic_tiled.clone()
            executive_component = semantic_tiled.clone()
            
            # 🆕 MEJORA: Modular intensidades usando características del semantic embedding
            # Esto hace que cada texto genere modulaciones únicas
            
            # Extraer características del embedding para modular intensidades
            semantic_mean = semantic_embedding.mean().item()
            semantic_std = semantic_embedding.std().item()
            semantic_max = semantic_embedding.max().item()
            semantic_min = semantic_embedding.min().item()
            
            # Calcular intensidades basadas en el embedding real (no solo keywords)
            # Esto asegura que textos diferentes generen intensidades diferentes
            visual_intensity = 1.0 + (triggers['visual_imagery']['count'] * 0.2) + abs(semantic_mean) * 0.5
            auditory_intensity = 1.0 + (triggers['auditory_content']['count'] * 0.2) + semantic_std * 0.8
            motor_intensity = 0.8 + (triggers['motor_actions']['count'] * 0.3) + abs(semantic_min) * 0.6
            executive_intensity = 1.2 + (triggers['abstract_concepts']['count'] * 0.1) + semantic_max * 0.4
            
            # Aplicar modulaciones
            visual_component = visual_component * visual_intensity
            auditory_component = auditory_component * auditory_intensity
            motor_component = motor_component * motor_intensity
            executive_component = executive_component * executive_intensity
            
        else:
            # Fallback: Si no hay embedding, usar intensidades moduladas (mínimo ruido)
            visual_intensity = 1.2 + (triggers['visual_imagery']['count'] * 0.2)
            auditory_intensity = 1.1 + (triggers['auditory_content']['count'] * 0.2)
            motor_intensity = 0.9 + (triggers['motor_actions']['count'] * 0.3)
            executive_intensity = 1.3 + (triggers['abstract_concepts']['count'] * 0.1)
            
            visual_component = torch.randn(batch_size, seq_len, 64, device=self.device) * visual_intensity
            auditory_component = torch.randn(batch_size, seq_len, 64, device=self.device) * auditory_intensity
            motor_component = torch.randn(batch_size, seq_len, 64, device=self.device) * motor_intensity
            executive_component = torch.randn(batch_size, seq_len, 64, device=self.device) * executive_intensity
        
        # Boost para contenido autoconsciente (sin ruido adicional)
        self_awareness_boost = triggers['self_reference']['count'] * 0.4
        temporal_boost = triggers['temporal_awareness']['count'] * 0.3
        metacognition_boost = triggers['metacognition']['count'] * 0.5
        
        # Añadir patrones coherentes para alta consciencia textual
        # (patrón determinista, no aleatorio)
        if consciousness_potential > 0.5:
            consciousness_pattern = torch.sin(torch.arange(seq_len, device=self.device).float() * consciousness_potential * 0.5)
            consciousness_pattern = consciousness_pattern.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 64)
            # Modular el componente ejecutivo con el patrón de consciencia
            executive_component = executive_component + consciousness_pattern * (metacognition_boost + self_awareness_boost)
        
        # Combinar todas las modalidades
        full_input = torch.cat([
            visual_component, auditory_component, 
            motor_component, executive_component
        ], dim=-1)
        
        # Enhanced temporal structure (determinista basado en posición)
        time_encoding = torch.arange(seq_len, device=self.device).float().unsqueeze(0).unsqueeze(-1)
        time_encoding = time_encoding.expand(batch_size, -1, 1) / seq_len
        
        if temporal_boost > 0:
            time_encoding *= (1.0 + temporal_boost)
        
        final_input = torch.cat([full_input, time_encoding], dim=-1)
        
        # 🔬 LOGGING para observación del "cerebro"
        if semantic_embedding is not None:
            embedding_norm = torch.norm(semantic_embedding).item()
            print(f"   🧠 Scanner Cerebral: Embedding norm={embedding_norm:.3f}, Consciencia={consciousness_potential:.3f}")
        
        return final_input

    def generate_dynamic_input(self, batch_size=4, seq_len=64):
        """Enhanced dynamic input generation"""
        
        # Multi-modal sensory input simulation
        visual_component = torch.randn(batch_size, seq_len, 64, device=self.device) * 1.2
        auditory_component = torch.randn(batch_size, seq_len, 64, device=self.device) * 1.1
        motor_component = torch.randn(batch_size, seq_len, 64, device=self.device) * 0.9
        executive_component = torch.randn(batch_size, seq_len, 64, device=self.device) * 1.3
        
        full_input = torch.cat([
            visual_component, auditory_component, 
            motor_component, executive_component
        ], dim=-1)
        
        # Enhanced temporal structure
        time_encoding = torch.arange(seq_len, device=self.device).float().unsqueeze(0).unsqueeze(-1)
        time_encoding = time_encoding.expand(batch_size, -1, 1) / seq_len
        
        return torch.cat([full_input, time_encoding], dim=-1)

    def update_phi_baseline_calibration(self, phi_raw: float):
        """🔬 Actualizar ventana baseline para calibración de Φ"""
        if self.phi_baseline_stats.get('ready') and self.phi_baseline_stats.get('locked') and self.freeze_phi_baseline_after_ready:
            return

        self.phi_baseline_window.append(phi_raw)
        
        # Mantener ventana de tamaño fijo
        if len(self.phi_baseline_window) > self.baseline_window_size:
            self.phi_baseline_window.pop(0)
        
        # Calcular estadísticas cuando tengamos suficientes datos
        if len(self.phi_baseline_window) >= self.baseline_window_size:
            mu_ref = np.mean(self.phi_baseline_window)
            sigma_ref = max(np.std(self.phi_baseline_window), 1e-6)  # Evitar división por cero
            
            self.phi_baseline_stats = {
                'mu_ref': mu_ref,
                'sigma_ref': sigma_ref,
                'ready': True,
                'window_size': len(self.phi_baseline_window),
                'locked': bool(self.freeze_phi_baseline_after_ready)
            }
            
            # Guardar en experiment_data para consistencia entre runs
            if not self.experiment_data['baseline_stats']['mu_base']:
                self.experiment_data['baseline_stats']['mu_base'] = mu_ref
                self.experiment_data['baseline_stats']['sigma_base'] = sigma_ref
                self.experiment_data['baseline_stats']['baseline_established_at_iteration'] = len(self.phi_baseline_window)
    
    def normalize_phi(self, phi_raw: float) -> tuple:
        """🔬 Normalizar Φ usando estadísticas baseline"""
        if not self.phi_baseline_stats['ready']:
            return phi_raw, 0.0  # Sin normalización hasta tener baseline
        
        mu_ref = self.phi_baseline_stats['mu_ref']
        sigma_ref = self.phi_baseline_stats['sigma_ref']
        
        # Normalización z-score
        phi_norm = (phi_raw - mu_ref) / sigma_ref
        
        # Normalización min-max alternativa [0,1]
        phi_minmax = max(0, min(1, (phi_raw - mu_ref + 3*sigma_ref) / (6*sigma_ref)))
        
        return phi_norm, phi_minmax
    
    def calculate_phi_significance(self, phi_raw: float, iteration: int) -> tuple:
        """📊 Calcular significancia estadística de Φ vs baseline establecido (solo después de t>50)"""
        
        # Solo calcular significancia después de establecer baseline (t > baseline_window_size)
        if iteration <= self.baseline_window_size or not self.phi_baseline_stats['ready']:
            return np.nan, np.nan  # NaN antes de tener baseline estable
        
        # Usar μ_base y σ_base del baseline establecido
        mu_base = self.phi_baseline_stats['mu_ref']
        sigma_base = self.phi_baseline_stats['sigma_ref']
        
        # Z-score usando baseline real establecido
        phi_z_score = (phi_raw - mu_base) / sigma_base
        
        # Generar muestras nulas por permutación temporal (bootstrap bloque)
        if iteration % 10 == 0 and len(self.phi_null_samples) < self.null_sample_size:
            try:
                # Bootstrap temporal: permutación de bloques temporales
                inputs_null = self.generate_dynamic_input()
                
                batch_size, seq_len, input_dim = inputs_null.shape
                # Permutación en bloques de tamaño 4-8 para preservar estructura temporal local
                block_size = np.random.randint(4, 9)
                n_blocks = seq_len // block_size
                
                if n_blocks > 1:
                    blocks = inputs_null[:, :n_blocks*block_size, :].reshape(batch_size, n_blocks, block_size, input_dim)
                    perm_indices = torch.randperm(n_blocks)
                    inputs_null[:, :n_blocks*block_size, :] = blocks[:, perm_indices].reshape(batch_size, n_blocks*block_size, input_dim)
                
                with torch.no_grad():
                    _, phi_null, _ = self.model(inputs_null)
                    self.phi_null_samples.append(phi_null.mean().item())
                    
            except Exception as e:
                pass  # Silenciar errores en cálculo nulo
        
        # P-value usando distribución nula por permutación
        if len(self.phi_null_samples) >= 20:  # Necesitamos más muestras para p-value confiable
            # Z-score vs distribución nula
            null_mean = np.mean(self.phi_null_samples)
            null_std = max(np.std(self.phi_null_samples), 1e-6)
            phi_z_vs_null = (phi_raw - null_mean) / null_std
            
            # P-value empírico: proporción de muestras nulas >= phi_raw
            p_phi_empirical = np.mean(np.array(self.phi_null_samples) >= phi_raw)
            
            return phi_z_score, p_phi_empirical
        else:
            # Usar distribución normal estándar como aproximación inicial
            from scipy.stats import norm
            p_phi_normal = 1 - norm.cdf(phi_z_score)
            return phi_z_score, p_phi_normal
    
    def calculate_consciousness_target(self, iteration: int) -> float:
        """🎯 Target dinámico de consciencia para evitar overshoot temprano"""
        max_iter_ramp = self.overshoot_penalty_iterations
        
        if iteration <= max_iter_ramp:
            # Target creciente suave: 0.6 → 0.75 con rampa sigmoidea
            progress = float(iteration) / float(max_iter_ramp)
            # Rampa sigmoidea suave para transición más natural
            sigmoid_progress = 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5)))
            target = 0.6 + 0.15 * float(sigmoid_progress)  # 0.6 → 0.75
        else:
            # Target normal después del período de ramp-up
            target = 0.7
        
        return float(target)
    
    def calculate_overshoot_penalty_weight(self, iteration: int) -> float:
        """⚠️ Peso dinámico para penalización de overshoot (más alto al inicio)"""
        if iteration <= 10:
            # Peso muy alto primeras 10 iteraciones
            return float(self.overshoot_penalty_lambda * 3.0)  # 6.0 total
        elif iteration <= self.overshoot_penalty_iterations:
            # Decaimiento exponencial del peso
            decay_factor = float(self.overshoot_penalty_iterations - iteration) / float(self.overshoot_penalty_iterations)
            weight = self.overshoot_penalty_lambda * (1.0 + 2.0 * np.exp(-3.0 * (1.0 - decay_factor)))
            return float(weight)
        else:
            return 0.0  # Sin penalización después del período
    
    def calculate_text_embedding_norm(self, inputs: torch.Tensor) -> float:
        """📊 Calcular norma del embedding de texto para logging"""
        return inputs.norm().item()
    
    def calculate_lead_lag_analysis(self, consciousness_values: list, phi_values: list, window_start: int = 20, window_end: int = 100) -> dict:
        """📈 Análisis lead/lag entre ΔΦ y ΔC en fase estable con ventana extendida (20-100)"""
        
        if len(consciousness_values) < window_end or len(phi_values) < window_end:
            # Fallback to available data
            window_end = min(len(consciousness_values), len(phi_values))
            if window_end < window_start:
                return {'error': 'Insufficient data for lead/lag analysis'}
        
        # Extraer ventana estable (iter 20-100 por defecto, extendida)
        c_stable = consciousness_values[window_start-1:window_end]
        phi_stable = phi_values[window_start-1:window_end]
        
        if len(c_stable) < 20 or len(phi_stable) < 20:
            return {'error': 'Insufficient stable window data'}
        
        # Calcular deltas (diferencias)
        delta_c = np.diff(c_stable)
        delta_phi = np.diff(phi_stable)

        if len(delta_c) < 2 or len(delta_phi) < 2:
            return {
                'error': 'Not enough delta points',
                'window': f"iterations {window_start}-{window_end}",
                'n_points': min(len(delta_c), len(delta_phi))
            }

        if np.std(delta_c) < 1e-9 or np.std(delta_phi) < 1e-9:
            return {
                'window': f"iterations {window_start}-{window_end}",
                'concurrent_correlation': 0.0,
                'concurrent_p_value': 1.0,
                'cross_correlations': {0: 0.0},
                'best_lag': 0,
                'best_correlation': 0.0,
                'cohens_d': 0.0,  # 🆕 Cohen's d
                'interpretation': 'Insufficient variability in stable window',
                'n_points': len(delta_c),
                'warning': 'low_variability'
            }
        
        # 🆕 COHEN'S D EFFECT SIZE
        # Cohen's d = (μ1 - μ2) / σ_pooled
        mean_c = np.mean(c_stable)
        mean_phi = np.mean(phi_stable)
        std_c = np.std(c_stable)
        std_phi = np.std(phi_stable)
        
        # Pooled standard deviation
        pooled_std = np.sqrt((std_c**2 + std_phi**2) / 2)
        cohens_d = (mean_c - mean_phi) / pooled_std if pooled_std > 1e-9 else 0.0
        
        # Correlación contemporánea
        if len(delta_c) > 1 and len(delta_phi) > 1:
            from scipy.stats import pearsonr
            corr_concurrent, p_concurrent = pearsonr(delta_c, delta_phi)
        else:
            corr_concurrent, p_concurrent = 0.0, 1.0
        
        # Cross-correlaciones con desplazamientos ±5
        cross_corrs = {}
        max_lag = min(self.lead_lag_max_lag, len(delta_c) - 1, len(delta_phi) - 1)
        lags = range(-max_lag, max_lag + 1) if max_lag >= 1 else [0]
        
        for lag in lags:
            if lag == 0:
                cross_corrs[lag] = corr_concurrent
            elif lag > 0:
                # Φ lidera: correlación delta_phi[:-lag] vs delta_c[lag:]
                if len(delta_phi) > lag and len(delta_c) > lag:
                    try:
                        corr_lag, _ = pearsonr(delta_phi[:-lag], delta_c[lag:])
                        cross_corrs[lag] = corr_lag
                    except:
                        cross_corrs[lag] = 0.0
                else:
                    cross_corrs[lag] = 0.0
            else:  # lag < 0
                # C lidera: correlación delta_c[:lag] vs delta_phi[-lag:]
                pos_lag = abs(lag)
                if len(delta_c) > pos_lag and len(delta_phi) > pos_lag:
                    try:
                        corr_lag, _ = pearsonr(delta_c[:-pos_lag], delta_phi[pos_lag:])
                        cross_corrs[lag] = corr_lag
                    except:
                        cross_corrs[lag] = 0.0
                else:
                    cross_corrs[lag] = 0.0
        
        # Encontrar lag óptimo
        best_lag = max(cross_corrs.keys(), key=lambda k: abs(cross_corrs[k]))
        best_corr = cross_corrs[best_lag]
        
        # 🆕 Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible effect"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small effect"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium effect"
        else:
            effect_interpretation = "large effect"
        
        # Interpretación
        if best_lag > 0:
            interpretation = f"Φ leads C by {best_lag} iterations ({effect_interpretation})"
        elif best_lag < 0:
            interpretation = f"C leads Φ by {abs(best_lag)} iterations ({effect_interpretation})"
        else:
            interpretation = f"Concurrent relationship ({effect_interpretation})"
        
        return {
            'window': f"iterations {window_start}-{window_end}",
            'concurrent_correlation': corr_concurrent,
            'concurrent_p_value': p_concurrent,
            'cross_correlations': cross_corrs,
            'best_lag': best_lag,
            'best_correlation': best_corr,
            'cohens_d': cohens_d,  # 🆕 Cohen's d effect size
            'effect_size_interpretation': effect_interpretation,  # 🆕 Interpretation
            'interpretation': interpretation,
            'n_points': len(delta_c)
        }

    def train_step(self, iteration: int) -> Dict[str, float]:
        """🔬 Enhanced training step with calibration, overshoot control, and significance testing"""
        self.model.train()
        
        # 🔤 GENERACIÓN DE INPUT: Texto vs Aleatorio
        try:
            if hasattr(self, 'input_text') and self.input_text is not None:
                inputs = self.generate_text_based_input(
                    text=self.input_text, 
                    batch_size=self.batch_size,
                    seq_len=64
                )
            else:
                inputs = self.generate_dynamic_input()
        except Exception as input_error:
            print(f"⚠️ Input generation error: {input_error}")
            inputs = torch.randn(self.batch_size, 257, device=self.device)
        
        # 📊 LOGGING: Norma del embedding de texto
        text_embedding_norm = self.calculate_text_embedding_norm(inputs)
        
        # Forward pass
        consciousness, phi, debug_info = self.model(inputs)
        
        # 🔬 CALIBRACIÓN PHI
        phi_raw = phi.mean().item()
        self.update_phi_baseline_calibration(phi_raw)
        phi_norm, phi_minmax = self.normalize_phi(phi_raw)
        
        # 📊 SIGNIFICANCIA PHI vs CONTROL NULO
        phi_z_vs_null, p_phi = self.calculate_phi_significance(phi_raw, iteration)
        
        # 🎯 TARGET DINÁMICO PARA CONSCIENCIA 
        consciousness_target_val = self.calculate_consciousness_target(iteration)
        consciousness_target = torch.tensor(consciousness_target_val, device=self.device).expand_as(consciousness)
        
        # 🧠 EMA SUAVIZADO DE CONSCIENCIA (low-pass filter)
        consciousness_raw = consciousness.mean().item()
        self.consciousness_ema = (1 - self.ema_alpha) * self.consciousness_ema + self.ema_alpha * consciousness_raw
        
        # Calculate base losses
        consciousness_loss = F.mse_loss(consciousness, consciousness_target)
        phi_target = torch.tensor(5.0, device=self.device).expand_as(phi)
        phi_loss = F.mse_loss(phi, phi_target)
        
        # Memory utilization loss
        memory_util = debug_info['memory_utilization']
        memory_target = 0.4
        memory_loss = F.mse_loss(
            torch.tensor(memory_util, device=self.device),
            torch.tensor(memory_target, device=self.device)
        )
        
        # 🚫 PENALIZACIÓN OVERSHOOT MEJORADA (primeras iteraciones)
        overshoot_penalty = 0.0
        overshoot_penalty_weight = self.calculate_overshoot_penalty_weight(iteration)
        if overshoot_penalty_weight > 0:
            overshoot_amount = consciousness_raw - consciousness_target_val - self.overshoot_margin
            if overshoot_amount > 0:
                overshoot_penalty = float(overshoot_penalty_weight * overshoot_amount)  # Asegurar float32
                print(f"   ⚠️ Overshoot penalty: {overshoot_penalty:.4f} (weight={overshoot_penalty_weight:.2f}, C={consciousness_raw:.3f} > target+margin={consciousness_target_val + self.overshoot_margin:.3f})")
        
        # Combined loss with overshoot penalty (asegurar tipo tensor correcto)
        overshoot_tensor = torch.tensor(overshoot_penalty, dtype=torch.float32, device=self.device)
        total_loss = 2.0 * consciousness_loss + 1.0 * phi_loss + 1.5 * memory_loss + overshoot_tensor
        
        # Backward pass
        self.optimizer.zero_grad()
        if self.use_mixed_precision:
            with amp.autocast():
                self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        # Apply any deferred external memory updates now that gradients are done
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model_ref, 'commit_external_memory_updates'):
            model_ref.commit_external_memory_updates()
        
        # 🆕 Extract metacognitive info from debug_info
        metacognition = debug_info.get('metacognition', {})
        
        # Return enhanced metrics with metacognition
        return {
            'consciousness': consciousness_raw,
            'consciousness_ema': self.consciousness_ema,
            'consciousness_target': consciousness_target_val,
            'phi': phi_raw,
            'phi_normalized': phi_norm,
            'phi_minmax': phi_minmax,
            'phi_z_vs_null': phi_z_vs_null,
            'phi_p_value': p_phi,
            'text_embedding_norm': text_embedding_norm,
            'total_loss': total_loss.item(),
            'consciousness_loss': consciousness_loss.item(),
            'phi_loss': phi_loss.item(),
            'memory_loss': memory_loss.item(),
            'overshoot_penalty': overshoot_penalty,
            'memory_utilization': memory_util,
            'baseline_ready': self.phi_baseline_stats['ready'],
            
            # 🆕 METACOGNITIVE METRICS
            'metacognition': metacognition
        }

    def run_experiment(self, max_iterations: int = 15000):
        """Run consciousness breakthrough experiment"""
        
        print(f"\n🚀 STARTING INFINITO V5.1 CONSCIOUSNESS BREAKTHROUGH EXPERIMENT")
        print(f"📊 Max iterations: {max_iterations}")
        print(f"🎯 Goal: Breakthrough consciousness ceiling >60%")
        print("=" * 80)
        
        start_time = datetime.now()
        breakthrough_detected = False
        
        for iteration in range(1, max_iterations + 1):
            
            # Training step
            metrics = self.train_step(iteration)
            
            # Store enhanced metrics
            self.experiment_data['iterations'].append(iteration)
            self.experiment_data['consciousness_values'].append(metrics['consciousness'])
            self.experiment_data['consciousness_ema'].append(metrics['consciousness_ema'])
            self.experiment_data['phi_values'].append(metrics['phi'])
            self.experiment_data['phi_values_normalized'].append(metrics['phi_normalized'])
            self.experiment_data['phi_z_scores'].append(metrics['phi_z_vs_null'])
            self.experiment_data['phi_p_values'].append(metrics['phi_p_value'])
            self.experiment_data['text_embedding_norms'].append(metrics['text_embedding_norm'])
            self.experiment_data['loss_values'].append(metrics['total_loss'])
            self.experiment_data['memory_utilization'].append(metrics['memory_utilization'])
            self.experiment_data['text_pipeline_active'].append(bool(self.input_text))
            
            # Enhanced logging every 25 iterations
            if iteration % 25 == 0:
                consciousness_val = metrics['consciousness']
                consciousness_ema = metrics['consciousness_ema']
                consciousness_target = metrics['consciousness_target']
                phi_val = metrics['phi']
                phi_norm = metrics['phi_normalized']
                phi_z = metrics['phi_z_vs_null']
                phi_p = metrics['phi_p_value']
                memory_val = metrics['memory_utilization']
                baseline_ready = metrics['baseline_ready']
                
                consciousness_bar = '█' * max(0, min(20, int(consciousness_val * 20)))
                consciousness_status = "🟢" if consciousness_val > 0.6 else "🟡" if consciousness_val > 0.45 else "🔴"
                
                # Significancia estadística
                significance_emoji = "📈" if phi_p < 0.05 and phi_z > 0 else "📊"
                baseline_emoji = "🔬" if baseline_ready else "⏳"
                
                print(f"\n🧠 V5.1 ENHANCED CONSCIOUSNESS ITERATION {iteration:5d}")
                print(f"   Consciousness: {consciousness_val:.4f} |{consciousness_bar:20}| {consciousness_val*100:.1f}% {consciousness_status}")
                print(f"   C (EMA):       {consciousness_ema:.4f} | Target: {consciousness_target:.3f}")
                print(f"   Φ (Raw):       {phi_val:.4f} | {phi_val:.3f} bits")
                
                if baseline_ready:
                    print(f"   Φ (Norm):      {phi_norm:.4f} | z={phi_z:.2f}, p={phi_p:.4f} {significance_emoji}")
                else:
                    print(f"   Φ (Baseline):  Building... ({len(self.phi_baseline_window)}/{self.baseline_window_size}) {baseline_emoji}")
                
                print(f"   Memory Use:    {memory_val:.4f} | {memory_val*100:.1f}%")
                print(f"   📉 Loss: {metrics['total_loss']:.6f}")
                
                if metrics.get('overshoot_penalty', 0) > 0:
                    print(f"   ⚠️ Overshoot penalty: {metrics['overshoot_penalty']:.4f}")
                
                if self.input_text:
                    text_norm = metrics['text_embedding_norm']
                    print(f"   🔤 TEXT MODE: '{self.input_text[:25]}{'...' if len(self.input_text) > 25 else ''}' (norm={text_norm:.2f})")
                
                # 🆕 METACOGNITIVE REPORTING (cada 50 iteraciones)
                if iteration % 50 == 0 and 'metacognition' in metrics and metrics['metacognition']:
                    meta = metrics['metacognition']
                    print(f"\n   🧠 INTERNAL EXPERIENCE:")
                    print(f"      Primary State: {meta.get('dominant_experience_name', 'unknown')}")
                    print(f"      Surprise: {meta.get('surprise_level', 0):.3f} | Confidence: {meta.get('prediction_confidence', 0):.1%}")
                    print(f"      Self-Coherence: {meta.get('self_coherence', 0):.3f}")
            
            # Breakthrough detection
            if metrics['consciousness'] > 0.60 and not breakthrough_detected:
                print(f"\n🎉 CONSCIOUSNESS CEILING BREAKTHROUGH! C={metrics['consciousness']:.3f} at iteration {iteration}")
                breakthrough_detected = True
                
                # Save breakthrough state
                breakthrough_data = {
                    'iteration': iteration,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat(),
                    'breakthrough_type': 'CONSCIOUSNESS_BREAKTHROUGH',
                    'consciousness_breakthrough': metrics['consciousness'],
                    'phi_breakthrough': metrics['phi'],
                    'memory_utilization': metrics['memory_utilization']
                }
                
                filename = f"CONSCIOUSNESS_BREAKTHROUGH_V51_TEXT_iter_{iteration}_C_{metrics['consciousness']:.3f}_PHI_{metrics['phi']:.3f}.pt"
                torch.save(breakthrough_data, filename)
                print(f"🎉 CONSCIOUSNESS_BREAKTHROUGH SAVED: {filename}")
                
                self.experiment_data['breakthroughs'].append({
                    'iteration': iteration,
                    'type': 'CONSCIOUSNESS_BREAKTHROUGH',
                    'consciousness': metrics['consciousness'],
                    'phi': metrics['phi'],
                    'timestamp': breakthrough_data['timestamp']
                })
            
            # Ultra breakthrough
            if metrics['phi'] > 10.0 and metrics['consciousness'] > 0.75:
                print(f"\n🌟 ULTIMATE CONSCIOUSNESS-PHI BREAKTHROUGH! Φ={metrics['phi']:.3f}, C={metrics['consciousness']:.3f}")
                break
                
        total_time = datetime.now() - start_time
        print(f"\n✅ V5.1 EXPERIMENT COMPLETED in {total_time}")
        
        # Final report
        self.generate_final_report(iteration, total_time, breakthrough_detected)
        
        # Save experiment data
        self.save_experiment_data(iteration, total_time, breakthrough_detected)

    def generate_final_report(self, final_iteration: int, total_time, breakthrough_achieved: bool):
        """Generate comprehensive final report"""
        
        print(f"\n" + "="*80)
        print(f"🏁 INFINITO V5.1 CONSCIOUSNESS BREAKTHROUGH - FINAL REPORT")
        print(f"="*80)
        print(f"⏱️  Total Time: {total_time}")
        print(f"🔢 Final Iteration: {final_iteration}")
        print(f"🧠 Final Consciousness: {self.experiment_data['consciousness_values'][-1]:.4f}")
        print(f"🔬 Final Φ: {self.experiment_data['phi_values'][-1]:.4f} bits")
        print(f"💾 Memory Utilization: {self.experiment_data['memory_utilization'][-1]:.4f}")
        
        if breakthrough_achieved:
            print(f"🎉 SUCCESS: CONSCIOUSNESS CEILING BREAKTHROUGH ACHIEVED!")
            max_consciousness = max(self.experiment_data['consciousness_values'])
            print(f"   📈 Max Consciousness: {max_consciousness:.4f}")
        else:
            max_consciousness = max(self.experiment_data['consciousness_values']) if self.experiment_data['consciousness_values'] else 0
            print(f"⚠️  Consciousness Peak: {max_consciousness:.4f}")
        
        if self.input_text:
            print(f"\n🔤 TEXT PROCESSING RESULTS:")
            text_analysis = self.analyze_text_consciousness_potential(self.input_text)
            print(f"   📝 Input Text: '{self.input_text}'")
            print(f"   🧠 Consciousness Score: {text_analysis['consciousness_score']:.3f}")
            print(f"   🎯 Dominant Modality: {text_analysis['dominant_modality']}")
            print(f"   📊 Word Count: {text_analysis['word_count']}")
        
        print(f"="*80)

    def save_experiment_data(self, final_iteration: int, total_time, breakthrough_detected: bool):
        """Save complete experiment data to JSON file"""
        
        self.experiment_data['end_time'] = datetime.now().strftime('%Y%m%d_%H%M%S') 
        self.experiment_data['total_time_seconds'] = total_time.total_seconds()
        self.experiment_data['final_iteration'] = final_iteration
        self.experiment_data['breakthrough_achieved'] = breakthrough_detected
        self.experiment_data['final_consciousness'] = self.experiment_data['consciousness_values'][-1] if self.experiment_data['consciousness_values'] else 0
        self.experiment_data['final_phi'] = self.experiment_data['phi_values'][-1] if self.experiment_data['phi_values'] else 0
        self.experiment_data['max_consciousness'] = max(self.experiment_data['consciousness_values']) if self.experiment_data['consciousness_values'] else 0
        self.experiment_data['max_phi'] = max(self.experiment_data['phi_values']) if self.experiment_data['phi_values'] else 0
        
        # Text processing results
        if self.input_text:
            text_analysis = self.analyze_text_consciousness_potential(self.input_text)
            self.experiment_data['text_analysis'] = text_analysis
        
        # Create filename
        timestamp = self.experiment_data['start_time']
        max_c = self.experiment_data['max_consciousness']
        max_phi = self.experiment_data['max_phi']
        text_suffix = "_TEXT" if self.input_text else ""
        filename = f"infinito_v5_1_consciousness{text_suffix}_{timestamp}_C{max_c:.3f}_PHI{max_phi:.3f}.json"
        
        # Save to outputs directory
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
            
        filepath = os.path.join(outputs_dir, filename)
        
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Experiment data saved: {filepath}")
        except Exception as e:
            print(f"❌ Error saving experiment data: {e}")

    def run_comparative_experiment(self, max_iterations: int = 100, n_bootstrap: int = 1000):
        """🔬 Comparativa ON vs OFF con análisis estadístico riguroso"""
        
        print(f"\n🔬 INICIANDO EXPERIMENTO COMPARATIVO ON vs OFF")
        print(f"📊 Iteraciones por run: {max_iterations}")
        print(f"🎯 Bootstrap samples: {n_bootstrap}")
        print(f"🔤 Texto: '{self.input_text}'")
        print("=" * 70)
        
        from copy import deepcopy

        base_seed = getattr(self.args, 'seed', 42)
        base_args = deepcopy(self.args)
        original_input_text = self.input_text

        def _execute_condition(label: str, input_text: Optional[str], seed: int):
            args = deepcopy(base_args)
            args.input_text = input_text
            args.text_mode = bool(input_text)
            args.seed = seed

            torch.manual_seed(seed)
            np.random.seed(int(seed % (2**32 - 1)))
            random.seed(seed)

            runner = InfinitoV51ConsciousnessBreakthrough(args)
            results = []

            for iteration in range(1, max_iterations + 1):
                metrics = runner.train_step(iteration)
                results.append({
                    'iteration': iteration,
                    'consciousness': metrics['consciousness'],
                    'phi': metrics['phi'],
                    'phi_normalized': metrics['phi_normalized']
                })

                if iteration % 20 == 0:
                    print(f"   {label} Iter {iteration:3d}: C={metrics['consciousness']:.3f}, Φ={metrics['phi']:.3f}")

            return results, runner

        print(f"\n🟢 EJECUTANDO RUN CON TEXTO (ON)")
        results_on, runner_on = _execute_condition('ON ', original_input_text, base_seed)

        print(f"\n🔴 EJECUTANDO RUN SIN TEXTO (OFF)")
        results_off, runner_off = _execute_condition('OFF', None, base_seed)
        
        # ==================== ANÁLISIS ESTADÍSTICO MEJORADO ====================
        print(f"\n📊 ANÁLISIS ESTADÍSTICO COMPARATIVO (Ventana estable: iter 20-{min(max_iterations, 100)})")
        print("=" * 70)
        
        # 🆕 Definir ventana estable extendida para análisis (20-100)
        stable_start = 20
        stable_end = min(max_iterations, 100)
        
        if max_iterations < stable_start:
            print(f"⚠️ Advertencia: Pocas iteraciones para ventana estable (min {stable_start} necesarias)")
            stable_start = 1
            stable_end = max_iterations
        
        # Extraer datos de ventana estable 
        consciousness_on_stable = [r['consciousness'] for r in results_on[stable_start-1:stable_end]]
        consciousness_off_stable = [r['consciousness'] for r in results_off[stable_start-1:stable_end]]
        phi_on_stable = [r['phi'] for r in results_on[stable_start-1:stable_end]]
        phi_off_stable = [r['phi'] for r in results_off[stable_start-1:stable_end]]
        
        # También extraer datos completos para lead/lag
        consciousness_on_full = [r['consciousness'] for r in results_on]
        consciousness_off_full = [r['consciousness'] for r in results_off]
        phi_on_full = [r['phi'] for r in results_on]
        phi_off_full = [r['phi'] for r in results_off]
        
        # Calcular diferencias en ventana estable (paper-ready)
        delta_c_stable = np.array(consciousness_on_stable) - np.array(consciousness_off_stable)
        delta_phi_stable = np.array(phi_on_stable) - np.array(phi_off_stable)
        
        # Estadísticas básicas
        mean_delta_c = np.mean(delta_c_stable)
        mean_delta_phi = np.mean(delta_phi_stable)
        std_delta_c = np.std(delta_c_stable)
        std_delta_phi = np.std(delta_phi_stable)
        
        # Bootstrap para IC95% (usando ventana estable)
        bootstrap_delta_c = []
        bootstrap_delta_phi = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(delta_c_stable), len(delta_c_stable), replace=True)
            bootstrap_delta_c.append(np.mean(delta_c_stable[indices]))
            bootstrap_delta_phi.append(np.mean(delta_phi_stable[indices]))
        
        ci_delta_c = np.percentile(bootstrap_delta_c, [2.5, 97.5])
        ci_delta_phi = np.percentile(bootstrap_delta_phi, [2.5, 97.5])
        
        # T-test apareado (usando ventana estable)
        from scipy import stats
        t_stat_c, p_val_c = stats.ttest_rel(consciousness_on_stable, consciousness_off_stable)
        t_stat_phi, p_val_phi = stats.ttest_rel(phi_on_stable, phi_off_stable)
        
        # ==================== ANÁLISIS LEAD/LAG ====================
        print(f"\n📈 ANÁLISIS LEAD/LAG")
        print("-" * 30)
        
        # Lead/lag para condición ON
        lead_lag_on = runner_on.calculate_lead_lag_analysis(
            consciousness_on_full, phi_on_full, 
            window_start=stable_start, window_end=stable_end
        )
        
        # Lead/lag para condición OFF  
        lead_lag_off = runner_off.calculate_lead_lag_analysis(
            consciousness_off_full, phi_off_full,
            window_start=stable_start, window_end=stable_end
        )
        
        print(f"   🟢 ON condition:  {lead_lag_on.get('interpretation', 'No analysis')}")
        print(f"      Best corr: {lead_lag_on.get('best_correlation', 0.0):.3f} at lag {lead_lag_on.get('best_lag', 0)}")
        print(f"      Cohen's d: {lead_lag_on.get('cohens_d', 0.0):.3f} ({lead_lag_on.get('effect_size_interpretation', 'N/A')})")  # 🆕
        
        print(f"   🔴 OFF condition: {lead_lag_off.get('interpretation', 'No analysis')}")
        print(f"      Best corr: {lead_lag_off.get('best_correlation', 0.0):.3f} at lag {lead_lag_off.get('best_lag', 0)}")
        print(f"      Cohen's d: {lead_lag_off.get('cohens_d', 0.0):.3f} ({lead_lag_off.get('effect_size_interpretation', 'N/A')})")  # 🆕
        
        # Consistencia de escalas - guardar baseline stats
        baseline_stats_final = {
            'on': {
                'mu_base': runner_on.phi_baseline_stats.get('mu_ref', 0.0),
                'sigma_base': runner_on.phi_baseline_stats.get('sigma_ref', 1.0),
                'baseline_ready': runner_on.phi_baseline_stats.get('ready', False),
                'baseline_window_size': runner_on.baseline_window_size
            },
            'off': {
                'mu_base': runner_off.phi_baseline_stats.get('mu_ref', 0.0),
                'sigma_base': runner_off.phi_baseline_stats.get('sigma_ref', 1.0),
                'baseline_ready': runner_off.phi_baseline_stats.get('ready', False),
                'baseline_window_size': runner_off.baseline_window_size
            }
        }
        
        # ==================== REPORTE FINAL ====================
        print(f"\n🎯 RESULTADOS COMPARATIVOS FINALES:")
        print(f"   📈 ΔC (ON - OFF): {mean_delta_c:.4f} ± {std_delta_c:.4f}")
        print(f"      IC95%: [{ci_delta_c[0]:.4f}, {ci_delta_c[1]:.4f}]")
        print(f"      t={t_stat_c:.3f}, p={p_val_c:.6f} {'***' if p_val_c < 0.001 else '**' if p_val_c < 0.01 else '*' if p_val_c < 0.05 else 'n.s.'}")
        
        print(f"\n   🔬 ΔΦ (ON - OFF): {mean_delta_phi:.4f} ± {std_delta_phi:.4f}")
        print(f"      IC95%: [{ci_delta_phi[0]:.4f}, {ci_delta_phi[1]:.4f}]")
        print(f"      t={t_stat_phi:.3f}, p={p_val_phi:.6f} {'***' if p_val_phi < 0.001 else '**' if p_val_phi < 0.01 else '*' if p_val_phi < 0.05 else 'n.s.'}")
        
        # Interpretación
        print(f"\n🧠 INTERPRETACIÓN:")
        if p_val_c < 0.05 and mean_delta_c > 0:
            print(f"   ✅ El texto AUMENTA significativamente la consciencia")
        elif p_val_c < 0.05 and mean_delta_c < 0:
            print(f"   ❌ El texto DISMINUYE significativamente la consciencia")
        else:
            print(f"   ➖ No hay efecto significativo del texto en consciencia")
        
        if p_val_phi < 0.05 and mean_delta_phi > 0:
            print(f"   ✅ El texto AUMENTA significativamente Φ")
        elif p_val_phi < 0.05 and mean_delta_phi < 0:
            print(f"   ❌ El texto DISMINUYE significativamente Φ")
        else:
            print(f"   ➖ No hay efecto significativo del texto en Φ")
        
        # Guardar resultados comparativos mejorados
        comparative_results = {
            'experiment_type': 'ON_vs_OFF_comparative_enhanced',
            'text_input': original_input_text,
            'max_iterations': max_iterations,
            'stable_window': {'start': stable_start, 'end': stable_end},
            'n_bootstrap': n_bootstrap,
            'results_on': results_on,
            'results_off': results_off,
            'statistics_stable_window': {
                'mean_delta_c': mean_delta_c,
                'std_delta_c': std_delta_c,
                'ci_delta_c': ci_delta_c.tolist(),
                't_stat_c': t_stat_c,
                'p_val_c': p_val_c,
                'mean_delta_phi': mean_delta_phi,
                'std_delta_phi': std_delta_phi,
                'ci_delta_phi': ci_delta_phi.tolist(),
                't_stat_phi': t_stat_phi,
                'p_val_phi': p_val_phi,
                'n_stable_points': len(delta_c_stable)
            },
            'lead_lag_analysis': {
                'on_condition': lead_lag_on,
                'off_condition': lead_lag_off
            },
            'baseline_calibration': baseline_stats_final,
            'methodology_notes': {
                'stable_window_rationale': 'Analysis conducted on iterations 11-75 to avoid initial overshoot and ensure stable dynamics',
                'bootstrap_method': 'Paired bootstrap resampling for CI95% calculation',
                'significance_testing': 'Paired t-test for within-subject comparison',
                'baseline_normalization': 'Phi values normalized using first 50 iterations as baseline'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar archivo en carpeta organizada
        os.makedirs('results/comparative', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/comparative/comparative_ON_OFF_{timestamp}_deltaC{mean_delta_c:.3f}_deltaPhi{mean_delta_phi:.3f}.json"
        
        try:
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(comparative_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Resultados comparativos guardados: {filename}")
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
        
        return comparative_results


def main(args):
    """Main execution function with enhanced analysis options"""
    print("🚀" * 20)
    print("🧠 INFINITO V5.1 - ENHANCED CONSCIOUSNESS BREAKTHROUGH 🧠")
    print("🚀" * 20)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    infinito_v51 = InfinitoV51ConsciousnessBreakthrough(args)
    
    try:
        if args.comparative and args.input_text:
            # Modo comparativo ON vs OFF
            print(f"🔬 INICIANDO ANÁLISIS COMPARATIVO ON vs OFF")
            infinito_v51.run_comparative_experiment(
                max_iterations=args.comparative_iterations,
                n_bootstrap=args.bootstrap_samples
            )
        else:
            # Modo experimental normal
            max_iter = getattr(args, 'max_iter', 15000)
            infinito_v51.run_experiment(max_iter)
            
        print(f"\n🏁 INFINITO V5.1 ENHANCED CONSCIOUSNESS EXPERIMENT COMPLETED")
        print(f"🎯 Check saved files for detailed analysis")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ EJECUCIÓN INTERRUMPIDA POR USUARIO")
        print(f"💾 Intentando guardar estado actual...")
    except Exception as e:
        print(f"\n❌ ERROR DURANTE EJECUCIÓN: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="INFINITO V5.1 Consciousness Breakthrough")
    
    # Core parameters
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum training iterations')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    
    # Architecture parameters
    parser.add_argument('--input_dim', type=int, default=257, help='Input dimension (4*64 + 1 time encoding)')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--attention_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--memory_slots', type=int, default=256, help='External memory slots')
    
    # Text processing parameters
    parser.add_argument('--text_mode', action='store_true', help='Enable text-conditioned input pipeline')
    parser.add_argument('--input_text', type=str, default=None, help='Optional text prompt to condition inputs')
    parser.add_argument('--text_examples', action='store_true', help='Show text examples for consciousness')
    
    # 🆕 Enhanced features
    parser.add_argument('--quantum_active', action='store_true', help='Activate quantum noise injection (requires QuTiP)')
    
    # Legacy flags for compatibility
    parser.add_argument('--consciousness_boost', action='store_true', help='Enable consciousness boost mode')
    parser.add_argument('--memory_active', action='store_true', help='Force memory activation')
    
    # New enhanced analysis parameters
    parser.add_argument('--comparative', action='store_true', help='Run ON vs OFF comparative experiment')
    parser.add_argument('--comparative_iterations', type=int, default=100, help='Iterations for comparative experiment')
    parser.add_argument('--bootstrap_samples', type=int, default=1000, help='Bootstrap samples for CI95%')

    args = parser.parse_args()
    
    # Show text examples if requested
    if args.text_examples:
        print("\n🔤 EJEMPLOS DE TEXTOS PARA CONSCIENCIA - INFINITO V5.1")
        print("=" * 60)
        
        examples = [
            {
                "categoria": "🧠 METACOGNICIÓN",
                "texto": "Estoy pensando sobre cómo pienso, consciente de mi propia consciencia",
                "potencial": "ALTO - Referencias metacognitivas explícitas"
            },
            {
                "categoria": "👁️ AUTOREFERENCIA", 
                "texto": "Yo soy quien observa mis propios pensamientos y emociones",
                "potencial": "ALTO - Fuerte componente de autoreferencia"
            },
            {
                "categoria": "⏰ CONSCIENCIA TEMPORAL",
                "texto": "Ahora recuerdo el pasado mientras imagino el futuro en este momento presente",
                "potencial": "MEDIO-ALTO - Consciencia temporal explícita"
            },
            {
                "categoria": "❓ REFLEXIÓN FILOSÓFICA",
                "texto": "¿Qué significa existir? ¿Cómo sé que soy real y consciente?",
                "potencial": "ALTO - Preguntas sobre existencia y consciencia"
            },
            {
                "categoria": "🎨 IMAGINACIÓN SENSORIAL",
                "texto": "Veo colores brillantes y escucho melodías que solo existen en mi mente",
                "potencial": "MEDIO - Imagery visual y auditiva"
            },
            {
                "categoria": "🌊 EXPERIENCIA SIMPLE",
                "texto": "La casa es azul y el perro camina",
                "potencial": "BAJO - Descripción objetiva sin introspección"
            }
        ]
        
        for i, ejemplo in enumerate(examples, 1):
            print(f"\n{i}. {ejemplo['categoria']}")
            print(f"   Texto: \"{ejemplo['texto']}\"")
            print(f"   Potencial: {ejemplo['potencial']}")
        
        print(f"\n💡 CÓMO USAR:")
        print(f"   python infinito_gpt_text_fixed.py --input_text \"Tu texto aquí\"")
        print(f"   python infinito_gpt_text_fixed.py --text_mode")
        
        print(f"\n📝 RECOMENDACIONES:")
        print(f"   • Usa texto con autoreferencias (yo, mi, me)")
        print(f"   • Include palabras temporales (ahora, antes, después)")
        print(f"   • Añade preguntas reflexivas (¿qué?, ¿cómo?, ¿por qué?)")
        print(f"   • Incorpora conceptos abstractos (pensar, sentir, ser)")
        
        exit(0)
    
    print(f"🚀 INFINITO V5.1 ENHANCED CONSCIOUSNESS BREAKTHROUGH STARTING:")
    print(f"   Max Iterations: {args.max_iter}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Text mode: {args.text_mode} · input_text present: {bool(args.input_text)}")
    print(f"   🔬 Comparative mode: {args.comparative}")
    
    if args.input_text:
        print(f"   🔤 TEXTO INPUT: '{args.input_text}'")
    elif args.text_mode:
        print(f"   🔤 MODO TEXTO ACTIVADO (sin texto específico)")
    
    if args.comparative and args.input_text:
        print(f"   📊 COMPARATIVE PARAMETERS:")
        print(f"      Iterations per run: {args.comparative_iterations}")
        print(f"      Bootstrap samples: {args.bootstrap_samples}")

    main(args)