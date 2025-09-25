"""
Vector 6: Conftest - Configuración pytest compartida
"""
import sys
import os
import pytest

# Agregar src al path para todos los tests
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configuración global para tests
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup global para todos los tests"""
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reducir verbosidad en tests
    
    # Import verificación
    try:
        import infinito_v3_stable
        print(f"✅ infinito_v3_stable imported successfully from: {infinito_v3_stable.__file__}")
    except ImportError as e:
        print(f"❌ Could not import infinito_v3_stable: {e}")

@pytest.fixture
def sample_tensor():
    """Fixture para tensor de prueba"""
    import torch
    return torch.randn(4, 32)

@pytest.fixture  
def device():
    """Fixture para device"""
    try:
        from infinito_v3_stable import device
        return device
    except ImportError:
        import torch
        return torch.device('cpu')
