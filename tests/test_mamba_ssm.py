"""
Vector 6: Test Mamba-SSM Core Components
Coverage Target: >80% para componentes críticos
"""
import pytest
import torch
import numpy as np
import sys
import os

# Agregar src al path para imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

try:
    from infinito_v3_stable import MambaSSM, device
    MAMBA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MambaSSM components: {e}")
    MAMBA_AVAILABLE = False

@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="MambaSSM components not available")

class TestMambaSSM:
    """Test suite para MambaSSM con gating entropy"""
    
    @pytest.fixture
    def mamba_model(self):
        """Fixture para modelo Mamba-SSM"""
        return MambaSSM(d_model=128, d_state=16, d_conv=4).to(device)
    
    def test_mamba_initialization(self, mamba_model):
        """Test: Inicialización correcta del modelo"""
        assert mamba_model.d_model == 128
        assert mamba_model.d_state == 16
        assert mamba_model.A.shape == (128, 16)
        assert mamba_model.D.shape == (128,)
        assert hasattr(mamba_model, 'gating_layer')
        assert hasattr(mamba_model, 'entropy_projector')
        assert hasattr(mamba_model, 'last_autocorr')
    
    def test_mamba_forward_pass(self, mamba_model):
        """Test: Forward pass genera outputs válidos"""
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        
        output, gates = mamba_model(x)
        
        # Verificar shapes
        assert output.shape == (batch_size, seq_len, d_model)
        assert gates.shape == (batch_size, seq_len, 16)  # d_state
        
        # Verificar que outputs son finitos
        assert torch.isfinite(output).all()
        assert torch.isfinite(gates).all()
        
        # Verificar que gates están en rango [0,1] (post sigmoid)
        assert (gates >= 0).all() and (gates <= 1).all()
    
    def test_gating_entropy_loss(self, mamba_model):
        """Test: Gating entropy loss es calculable"""
        batch_size, seq_len, d_state = 2, 10, 16
        gates = torch.rand(batch_size, seq_len, d_state).to(device)
        
        entropy_loss = mamba_model.get_gating_entropy_loss(gates)
        
        assert isinstance(entropy_loss, torch.Tensor)
        assert entropy_loss.dim() == 0  # scalar
        assert entropy_loss.item() >= 0  # KL divergence is non-negative
        assert torch.isfinite(entropy_loss)
    
    def test_autocorr_calculation(self, mamba_model):
        """Test: Autocorrelación lag-1 es calculable"""
        # Simular history con valores conocidos
        test_states = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        mamba_model.state_history.extend(test_states)
        
        autocorr = mamba_model.get_autocorr_lag1()
        
        # Para secuencia creciente, autocorr debería ser alta
        assert isinstance(autocorr, float)
        assert autocorr > 0.8  # Secuencia muy correlacionada
        assert mamba_model.last_autocorr == autocorr  # Vector 5 metric updated
    
    def test_gate_strength_std(self, mamba_model):
        """Test: Gate strength std es calculable"""
        # Simular gate strengths variados
        test_strengths = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5]
        mamba_model.gate_history.extend(test_strengths)
        
        std_strength = mamba_model.get_gate_strength_std()
        
        assert isinstance(std_strength, float)
        assert std_strength > 0.15  # KPI target >0.15
        assert std_strength < 1.0   # Reasonable upper bound
    
    def test_state_diversity_loss(self, mamba_model):
        """Test: State diversity loss promueve diversidad"""
        # Estados idénticos = baja diversidad = alto loss
        identical_states = [0.5] * 10
        mamba_model.state_history.extend(identical_states)
        
        loss_identical = mamba_model.get_state_diversity_loss()
        
        # Estados diversos = alta diversidad = bajo loss
        mamba_model.state_history.clear()
        diverse_states = [i * 0.1 for i in range(10)]
        mamba_model.state_history.extend(diverse_states)
        
        loss_diverse = mamba_model.get_state_diversity_loss()
        
        # Loss debería ser mayor para estados idénticos
        assert loss_identical > loss_diverse
        assert torch.isfinite(loss_identical)
        assert torch.isfinite(loss_diverse)
    
    def test_mamba_gradient_flow(self, mamba_model):
        """Test: Gradientes fluyen correctamente"""
        batch_size, seq_len, d_model = 1, 5, 128
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True).to(device)
        
        output, gates = mamba_model(x)
        loss = output.sum() + gates.sum()
        loss.backward()
        
        # Verificar que parámetros tienen gradientes
        for name, param in mamba_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"
    
    @pytest.mark.parametrize("d_model,d_state", [(64, 8), (256, 32), (512, 64)])
    def test_mamba_different_sizes(self, d_model, d_state):
        """Test: Mamba funciona con diferentes tamaños"""
        model = MambaSSM(d_model=d_model, d_state=d_state).to(device)
        x = torch.randn(1, 10, d_model).to(device)
        
        output, gates = model(x)
        
        assert output.shape == (1, 10, d_model)
        assert gates.shape == (1, 10, d_state)
        assert torch.isfinite(output).all()
        assert torch.isfinite(gates).all()
    
    def test_mamba_state_persistence(self, mamba_model):
        """Test: Estado persiste entre forward passes"""
        x1 = torch.randn(1, 5, 128).to(device)
        x2 = torch.randn(1, 5, 128).to(device)
        
        # Primer forward
        output1, _ = mamba_model(x1)
        state_after_1 = mamba_model.state.clone()
        
        # Segundo forward
        output2, _ = mamba_model(x2)
        state_after_2 = mamba_model.state.clone()
        
        # Estado debería cambiar entre forwards
        assert not torch.equal(state_after_1, state_after_2)
        
        # History debería crecer
        assert len(mamba_model.state_history) >= 2
    
    def test_mamba_finite_assertions(self, mamba_model):
        """Test: Vector 6 - Finite model assertions"""
        x = torch.randn(2, 10, 128).to(device)
        
        output, gates = mamba_model(x)
        
        # Assertions para modelo finito
        assert torch.isfinite(output).all(), "Output contiene valores no finitos"
        assert torch.isfinite(gates).all(), "Gates contienen valores no finitos"
        assert not torch.isnan(output).any(), "Output contiene NaN"
        assert not torch.isnan(gates).any(), "Gates contienen NaN"
        assert not torch.isinf(output).any(), "Output contiene Inf"
        assert not torch.isinf(gates).any(), "Gates contienen Inf"
        
        # Verificar rangos razonables
        assert output.abs().max() < 100, "Output excede rango razonable"
        assert gates.min() >= 0 and gates.max() <= 1, "Gates fuera de rango [0,1]"
