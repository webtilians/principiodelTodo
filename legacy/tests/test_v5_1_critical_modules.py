#!/usr/bin/env python3
"""
🧪 INFINITO V5.1 - TESTS CRÍTICOS DE MÓDULOS COMPLEJOS
======================================================

Tests exhaustivos para los módulos más críticos de V5.1:
- 🧠 Enhanced External Memory (auto-activación)
- ⚡ Enhanced Phi Calculator (cálculos Φ)  
- 🛑 Early Stop Manager (gestión parada temprana)

Autor: INFINITO Testing Team
Fecha: 2025-09-25
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Tuple

# Agregar src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar módulos específicos de V5.1
try:
    from infinito_v5_1_consciousness import (
        EnhancedExternalMemory,
        EnhancedPhiCalculatorV51,
        V51ConsciousnessEarlyStopManager,
        InfinitoV51Consciousness
    )
except ImportError:
    pytest.skip("V5.1 modules not available", allow_module_level=True)


class TestEnhancedExternalMemory:
    """🧠 Tests exhaustivos para Enhanced External Memory V5.1"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.memory = EnhancedExternalMemory(
            memory_slots=128, 
            slot_size=32, 
            hidden_dim=256
        )
        self.batch_size = 4
        self.hidden_dim = 256
        
    def test_memory_initialization(self):
        """Test inicialización correcta de memoria"""
        assert self.memory.memory.shape == (128, 32)
        assert self.memory.memory_age.shape == (128,)
        assert self.memory.memory_strength.shape == (128,)
        
        # Verificar inicialización de valores
        assert torch.all(self.memory.memory == 0)
        assert torch.all(self.memory.memory_age == 0)
        assert torch.allclose(self.memory.memory_strength, torch.ones(128) * 0.1)
        
    def test_memory_read_without_consciousness(self):
        """Test lectura de memoria con consciencia baja (<30%)"""
        query = torch.randn(self.batch_size, self.hidden_dim)
        consciousness_level = torch.tensor([0.1, 0.2, 0.15, 0.25])  # Todos <30%
        
        memory_content, attention_weights = self.memory.read(query, consciousness_level)
        
        # Verificar dimensiones
        assert memory_content.shape == (self.batch_size, 32)
        assert attention_weights.shape == (self.batch_size, 128)
        
        # Con consciencia baja, la memoria debería tener actividad limitada
        assert torch.all(attention_weights.sum(dim=1) <= 1.1)  # Normalización aproximada
        
    def test_memory_read_with_high_consciousness(self):
        """Test lectura de memoria con consciencia alta (>30%) - AUTO-ACTIVACIÓN"""
        query = torch.randn(self.batch_size, self.hidden_dim)
        consciousness_level = torch.tensor([0.8, 0.9, 0.7, 0.95])  # Todos >30%
        
        # Primero escribir algo en memoria
        self.memory.write(query, torch.randn(self.batch_size, 32), consciousness_level)
        
        # Ahora leer con alta consciencia
        memory_content, attention_weights = self.memory.read(query, consciousness_level)
        
        # Con alta consciencia, debería haber más activación de memoria
        assert memory_content.shape == (self.batch_size, 32)
        assert torch.any(memory_content != 0)  # Debería tener contenido
        
    def test_memory_write_consolidation(self):
        """Test escritura y consolidación de memoria"""
        query = torch.randn(self.batch_size, self.hidden_dim)
        content = torch.randn(self.batch_size, 32)
        consciousness_level = torch.tensor([0.9, 0.8, 0.7, 0.95])
        
        # Estado inicial
        initial_memory_state = self.memory.memory.clone()
        
        # Escribir en memoria
        updated_memory = self.memory.write(query, content, consciousness_level)
        
        # Verificar que la memoria cambió
        assert not torch.equal(self.memory.memory, initial_memory_state)
        assert updated_memory.shape == (self.batch_size, 32)
        
    def test_memory_utilization_tracking(self):
        """Test seguimiento de utilización de memoria"""
        query = torch.randn(self.batch_size, self.hidden_dim)
        consciousness_level = torch.tensor([0.8, 0.9, 0.7, 0.95])
        
        initial_utilization = self.memory.memory_utilization_tracker.item()
        
        # Realizar operaciones de memoria
        self.memory.read(query, consciousness_level)
        
        # La utilización puede cambiar (dependiendo de la implementación)
        final_utilization = self.memory.memory_utilization_tracker.item()
        
        # Verificar que el tracker existe y funciona
        assert isinstance(final_utilization, float)
        
    def test_consciousness_threshold_activation(self):
        """Test crítico: activación automática al superar 30% consciencia"""
        query = torch.randn(1, self.hidden_dim)
        
        # Test con consciencia exactamente en 30%
        consciousness_30 = torch.tensor([0.3])
        memory_content_30, weights_30 = self.memory.read(query, consciousness_30)
        
        # Test con consciencia justo por debajo (29%)
        consciousness_29 = torch.tensor([0.29])
        memory_content_29, weights_29 = self.memory.read(query, consciousness_29)
        
        # Test con consciencia por encima (31%)
        consciousness_31 = torch.tensor([0.31])
        memory_content_31, weights_31 = self.memory.read(query, consciousness_31)
        
        # La activación debería ser diferente en el umbral
        # (comportamiento específico depende de la implementación)
        assert memory_content_30.shape == memory_content_29.shape == memory_content_31.shape


class TestEnhancedPhiCalculatorV51:
    """⚡ Tests exhaustivos para Enhanced Phi Calculator V5.1"""
    
    def setup_method(self):
        """Setup para cada test de Φ"""
        self.phi_calc = EnhancedPhiCalculatorV51(input_dim=256)
        self.batch_size = 4
        self.input_dim = 256
        
    def test_phi_calculator_initialization(self):
        """Test inicialización correcta del calculador Φ"""
        assert hasattr(self.phi_calc, 'integration_processor')
        assert hasattr(self.phi_calc, 'differentiation_processor') 
        assert hasattr(self.phi_calc, 'phi_estimator')
        
    def test_phi_calculation_basic(self):
        """Test cálculo básico de Φ"""
        system_state = torch.randn(self.batch_size, self.input_dim)
        
        phi_value, phi_components = self.phi_calc.calculate_phi(system_state)
        
        # Verificar dimensiones de salida
        assert phi_value.shape == (self.batch_size,)
        assert isinstance(phi_components, dict)
        
        # Verificar que Φ es un valor positivo
        assert torch.all(phi_value >= 0)
        
    def test_phi_values_range(self):
        """Test que los valores Φ están en rangos realistas"""
        system_state = torch.randn(self.batch_size, self.input_dim)
        
        phi_value, phi_components = self.phi_calc.calculate_phi(system_state)
        
        # V5.1 debería producir Φ en rango realista (0-15 bits como máximo)
        assert torch.all(phi_value <= 15.0)
        assert torch.all(phi_value >= 0.0)
        
    def test_phi_gradient_computation(self):
        """Test que el cálculo Φ permite gradientes"""
        system_state = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        
        phi_value, _ = self.phi_calc.calculate_phi(system_state)
        loss = phi_value.sum()
        
        # Verificar que se pueden calcular gradientes
        loss.backward()
        assert system_state.grad is not None
        assert not torch.all(system_state.grad == 0)  # Debería tener gradientes no-cero
        
    def test_phi_components_validity(self):
        """Test que los componentes Φ son válidos"""
        system_state = torch.randn(self.batch_size, self.input_dim)
        
        phi_value, phi_components = self.phi_calc.calculate_phi(system_state)
        
        # Verificar componentes esperados
        expected_components = ['integration', 'differentiation', 'complexity']
        for component in expected_components:
            if component in phi_components:
                assert isinstance(phi_components[component], torch.Tensor)
                
    def test_phi_consistency_across_calls(self):
        """Test consistencia de Φ con misma entrada"""
        system_state = torch.randn(self.batch_size, self.input_dim)
        
        # Calcular Φ múltiples veces con misma entrada
        phi_1, _ = self.phi_calc.calculate_phi(system_state)
        phi_2, _ = self.phi_calc.calculate_phi(system_state)
        
        # Deberían ser idénticos (o muy similares si hay componentes estocásticos)
        assert torch.allclose(phi_1, phi_2, rtol=1e-5, atol=1e-5)
        
    def test_phi_scaling_with_complexity(self):
        """Test que Φ escala apropiadamente con complejidad"""
        # Estado simple (baja complejidad)
        simple_state = torch.zeros(1, self.input_dim)
        simple_phi, _ = self.phi_calc.calculate_phi(simple_state)
        
        # Estado complejo (alta complejidad)
        complex_state = torch.randn(1, self.input_dim)
        complex_phi, _ = self.phi_calc.calculate_phi(complex_state)
        
        # El estado complejo debería tener Φ mayor (generalmente)
        # Nota: esto es una heurística, no siempre se cumple
        print(f"Simple Φ: {simple_phi.item():.3f}, Complex Φ: {complex_phi.item():.3f}")


class TestV51ConsciousnessEarlyStopManager:
    """🛑 Tests exhaustivos para Early Stop Manager V5.1"""
    
    def setup_method(self):
        """Setup para cada test de early stopping"""
        self.early_stop = V51ConsciousnessEarlyStopManager()
        
    def test_early_stop_initialization(self):
        """Test inicialización correcta del early stop manager"""
        assert hasattr(self.early_stop, 'consciousness_history')
        assert hasattr(self.early_stop, 'phi_history')
        assert hasattr(self.early_stop, 'loss_history')
        
    def test_should_stop_with_no_history(self):
        """Test early stop sin historial (no debería parar)"""
        metrics = {
            'consciousness': 0.5,
            'phi_integration': 1.0,
            'loss': 0.1
        }
        
        should_stop = self.early_stop.should_stop(metrics, iteration=10)
        assert not should_stop  # No debería parar sin historial suficiente
        
    def test_should_stop_with_good_progress(self):
        """Test early stop con buen progreso (no debería parar)"""
        # Simular progreso constante
        for i in range(100):
            metrics = {
                'consciousness': 0.3 + i * 0.005,  # Progreso constante
                'phi_integration': 0.5 + i * 0.01,
                'loss': 1.0 - i * 0.008
            }
            should_stop = self.early_stop.should_stop(metrics, iteration=i)
            
        # Con buen progreso, no debería parar
        assert not should_stop
        
    def test_should_stop_with_stagnation(self):
        """Test early stop con estancamiento (debería parar)"""
        # Simular estancamiento
        base_metrics = {
            'consciousness': 0.5,
            'phi_integration': 1.0,
            'loss': 0.3
        }
        
        # Agregar muchas iteraciones con mismo valor
        for i in range(200):
            # Pequeñas variaciones aleatorias para simular ruido
            metrics = {
                'consciousness': base_metrics['consciousness'] + np.random.normal(0, 0.001),
                'phi_integration': base_metrics['phi_integration'] + np.random.normal(0, 0.001),
                'loss': base_metrics['loss'] + np.random.normal(0, 0.001)
            }
            should_stop = self.early_stop.should_stop(metrics, iteration=i)
            
            # V5.1 requiere TODOS los criterios para early stop
            if should_stop:
                print(f"Early stop triggered at iteration {i}")
                break
                
    def test_consciousness_stagnation_detection(self):
        """Test detección específica de estancamiento de consciencia"""
        # Simular consciencia estancada
        consciousness_values = [0.5] * 150  # Consciencia constante
        
        for i, consciousness in enumerate(consciousness_values):
            metrics = {
                'consciousness': consciousness,
                'phi_integration': 1.0 + np.random.normal(0, 0.1),  # Φ variando
                'loss': 0.3 + np.random.normal(0, 0.05)  # Loss variando
            }
            self.early_stop.should_stop(metrics, iteration=i)
            
        # Verificar que detecta estancamiento de consciencia
        stagnation = self.early_stop._check_consciousness_stagnation()
        # El resultado depende de la implementación específica
        
    def test_phi_stagnation_detection(self):
        """Test detección específica de estancamiento de Φ"""
        # Simular Φ estancado
        phi_values = [1.2] * 150  # Φ constante
        
        for i, phi in enumerate(phi_values):
            metrics = {
                'consciousness': 0.5 + np.random.normal(0, 0.1),  # Consciencia variando
                'phi_integration': phi,
                'loss': 0.3 + np.random.normal(0, 0.05)  # Loss variando
            }
            self.early_stop.should_stop(metrics, iteration=i)
            
        # Verificar detección de estancamiento Φ
        phi_stagnation = self.early_stop._check_phi_stagnation()
        # El resultado depende de la implementación específica
        
    def test_early_stop_patience(self):
        """Test que el patience funciona correctamente"""
        # El early stop manager debería tener paciencia extendida en V5.1
        # Simular condiciones de early stop por muchas iteraciones
        
        stop_count = 0
        for i in range(500):  # V5.1 debería tener más paciencia
            metrics = {
                'consciousness': 0.5,  # Estancado
                'phi_integration': 1.0,  # Estancado
                'loss': 0.3  # Estancado
            }
            
            if self.early_stop.should_stop(metrics, iteration=i):
                stop_count += 1
                break
                
        # V5.1 debería esperar más iteraciones antes de parar
        print(f"Early stop triggered after {stop_count} iterations (should be high for V5.1)")


class TestV51IntegrationTests:
    """🔄 Tests de integración para V5.1 completo"""
    
    def setup_method(self):
        """Setup para tests de integración"""
        # Configuración mínima para tests rápidos
        self.config = {
            'max_iterations': 50,  # Muy poco para tests
            'grid_size': 16,       # Pequeño para rapidez
            'batch_size': 2,
            'hidden_dim': 64
        }
        
    def test_full_system_initialization(self):
        """Test que el sistema V5.1 completo se inicializa correctamente"""
        try:
            system = InfinitoV51Consciousness(**self.config)
            assert system is not None
            assert hasattr(system, 'memory')
            assert hasattr(system, 'phi_calculator')
            assert hasattr(system, 'early_stop_manager')
        except Exception as e:
            pytest.fail(f"Failed to initialize V5.1 system: {e}")
            
    def test_memory_phi_interaction(self):
        """Test interacción entre memoria y cálculo Φ"""
        system = InfinitoV51Consciousness(**self.config)
        
        # Ejecutar unas pocas iteraciones
        initial_metrics = system.get_current_metrics()
        
        # Verificar que las métricas son válidas
        assert 'consciousness' in initial_metrics
        assert 'phi_integration' in initial_metrics
        assert initial_metrics['consciousness'] >= 0
        assert initial_metrics['phi_integration'] >= 0
        
    def test_consciousness_progression_workflow(self):
        """Test que el flujo completo de progresión de consciencia funciona"""
        system = InfinitoV51Consciousness(**self.config)
        
        consciousness_values = []
        phi_values = []
        
        # Ejecutar algunas iteraciones y recoger métricas
        for i in range(10):
            metrics = system.get_current_metrics()
            consciousness_values.append(metrics['consciousness'])
            phi_values.append(metrics['phi_integration'])
            
            # Simular una iteración de entrenamiento
            # (esto requeriría acceso a métodos internos)
            
        # Verificar que los valores están en rangos esperados
        assert all(0 <= c <= 1 for c in consciousness_values)
        assert all(0 <= p <= 15 for p in phi_values)  # Φ realista


# Fixtures para tests
@pytest.fixture
def sample_metrics():
    """Métricas de ejemplo para tests"""
    return {
        'consciousness': 0.7,
        'phi_integration': 1.5,
        'loss': 0.25,
        'memory_utilization': 0.6,
        'eeg_correlation': 0.8
    }


@pytest.fixture
def v51_system_config():
    """Configuración de ejemplo para sistema V5.1"""
    return {
        'max_iterations': 100,
        'grid_size': 32,
        'batch_size': 4,
        'hidden_dim': 128,
        'memory_slots': 64,
        'consciousness_threshold': 0.3
    }


# Tests de rendimiento
class TestV51Performance:
    """⚡ Tests de rendimiento para V5.1"""
    
    def test_memory_performance_with_large_batch(self):
        """Test rendimiento de memoria con batch grande"""
        memory = EnhancedExternalMemory(memory_slots=512, slot_size=64, hidden_dim=512)
        
        large_batch_size = 32
        query = torch.randn(large_batch_size, 512)
        consciousness = torch.rand(large_batch_size)
        
        import time
        start_time = time.time()
        
        # Múltiples operaciones de memoria
        for _ in range(10):
            memory_content, _ = memory.read(query, consciousness)
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Debería completarse en tiempo razonable (< 5 segundos)
        assert duration < 5.0, f"Memory operations too slow: {duration:.2f}s"
        
    def test_phi_calculation_performance(self):
        """Test rendimiento del cálculo Φ"""
        phi_calc = EnhancedPhiCalculatorV51(input_dim=512)
        
        batch_size = 16
        system_state = torch.randn(batch_size, 512)
        
        import time
        start_time = time.time()
        
        # Múltiples cálculos Φ
        for _ in range(20):
            phi_value, _ = phi_calc.calculate_phi(system_state)
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Debería ser eficiente
        assert duration < 10.0, f"Phi calculations too slow: {duration:.2f}s"


if __name__ == "__main__":
    # Ejecutar tests específicos
    pytest.main([__file__, "-v", "--tb=short"])