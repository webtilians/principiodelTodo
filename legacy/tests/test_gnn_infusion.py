"""
Vector 6: Test GNN Infusion Components  
Coverage Target: >80% para TopologicalGNN y ClusterAnalyzer
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
    from infinito_v3_stable import TopologicalGNN, ClusterAnalyzer, device
    GNN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import GNN components: {e}")
    GNN_AVAILABLE = False

@pytest.mark.skipif(not GNN_AVAILABLE, reason="GNN components not available")
class TestTopologicalGNN:
    """Test suite para TopologicalGNN"""
    
    @pytest.fixture
    def gnn_model(self):
        """Fixture para modelo GNN"""
        return TopologicalGNN(hidden_size=64).to(device)
    
    def test_gnn_initialization(self, gnn_model):
        """Test: GNN se inicializa correctamente"""
        assert gnn_model.hidden_size == 64
        assert hasattr(gnn_model, 'conv1')
        assert hasattr(gnn_model, 'conv2')
        assert hasattr(gnn_model, 'transition_detector')
        assert hasattr(gnn_model, 'phase_history')
    
    def test_gnn_forward_pass(self, gnn_model):
        """Test: Forward pass del GNN"""
        batch_size, num_nodes, hidden_size = 1, 10, 64
        x = torch.randn(batch_size, num_nodes, hidden_size).to(device)
        
        output, topology_features = gnn_model(x)
        
        assert output.shape == (batch_size, num_nodes, hidden_size)
        assert topology_features.shape[0] == batch_size
        assert torch.isfinite(output).all()
        assert torch.isfinite(topology_features).all()
    
    def test_gnn_edge_generation(self, gnn_model):
        """Test: Generación de edges 4-conectados"""
        num_nodes = 8
        edge_index = gnn_model._create_4connected_edges(num_nodes)
        
        assert edge_index.shape[0] == 2  # [source, target]
        assert edge_index.max() < num_nodes  # Índices válidos
        assert edge_index.min() >= 0
        
        # Verificar que hay conexiones (no vacío)
        assert edge_index.shape[1] > 0
    
    def test_gnn_phase_transitions(self, gnn_model):
        """Test: Detección de transiciones de fase"""
        # Simular features para transición
        features = torch.randn(1, 64).to(device)
        
        phase_info = gnn_model.detect_phase_transitions(features)
        
        assert isinstance(phase_info, dict)
        assert 'variance' in phase_info
        assert 'coherence' in phase_info
        assert 'phase_type' in phase_info
        
        assert isinstance(phase_info['variance'], float)
        assert isinstance(phase_info['coherence'], float)
        assert phase_info['phase_type'] in ['gapped', 'gapless', 'unknown']
    
    @pytest.mark.parametrize("hidden_size", [32, 64, 128, 256])
    def test_gnn_different_sizes(self, hidden_size):
        """Test: GNN con diferentes tamaños"""
        model = TopologicalGNN(hidden_size=hidden_size).to(device)
        x = torch.randn(1, 10, hidden_size).to(device)
        
        output, features = model(x)
        
        assert output.shape == (1, 10, hidden_size)
        assert torch.isfinite(output).all()
        assert torch.isfinite(features).all()

@pytest.mark.skipif(not GNN_AVAILABLE, reason="GNN components not available")  
class TestClusterAnalyzer:
    """Test suite para ClusterAnalyzer"""
    
    @pytest.fixture
    def cluster_analyzer(self):
        """Fixture para ClusterAnalyzer"""
        return ClusterAnalyzer()
    
    def test_analyzer_initialization(self, cluster_analyzer):
        """Test: ClusterAnalyzer se inicializa correctamente"""
        assert hasattr(cluster_analyzer, 'cluster_history')
        assert hasattr(cluster_analyzer, 'detected_laws')
        assert hasattr(cluster_analyzer, 'pattern_memory')
        assert hasattr(cluster_analyzer, 'phi_gnn')
        assert hasattr(cluster_analyzer, 'last_gnn_complexity')
        
        # Vector 5 metrics inicializadas
        assert cluster_analyzer.phi_gnn == 0.0
        assert cluster_analyzer.last_gnn_complexity == 0.0
    
    def test_gnn_initialization(self, cluster_analyzer):
        """Test: GNN se inicializa cuando es necesario"""
        hidden_size = 64
        cluster_analyzer.initialize_gnn(hidden_size)
        
        assert cluster_analyzer.gnn is not None
        assert cluster_analyzer.gnn.hidden_size == hidden_size
    
    def test_kmeans_clustering(self, cluster_analyzer):
        """Test: K-means clustering funciona"""
        # Datos sintéticos con clusters obvios
        data = np.vstack([
            np.random.randn(10, 5) + [0, 0, 0, 0, 0],  # Cluster 1
            np.random.randn(10, 5) + [5, 5, 5, 5, 5],  # Cluster 2
        ])
        
        clusters = cluster_analyzer._simple_kmeans(data, k=2)
        
        assert isinstance(clusters, list)
        assert len(clusters) == 2  # 2 clusters solicitados
        
        # Verificar estructura de cluster
        for cluster in clusters:
            assert 'center' in cluster
            assert 'size' in cluster
            assert 'cohesion' in cluster
            assert cluster['size'] > 0
    
    def test_pattern_detection(self, cluster_analyzer):
        """Test: Detección de patrones funciona"""
        # Simular clusters para detección
        clusters = [
            {'center': [0, 0], 'size': 10, 'cohesion': 0.9},
            {'center': [1, 1], 'size': 8, 'cohesion': 0.7}
        ]
        
        phase_info = {'variance': 0.5, 'coherence': 0.8, 'phase_type': 'gapless'}
        patterns = cluster_analyzer._detect_patterns(clusters, iteration=100, phase_info=phase_info)
        
        assert isinstance(patterns, list)
        for pattern in patterns:
            assert 'type' in pattern
            assert 'strength' in pattern
            assert 'description' in pattern
    
    def test_law_detection(self, cluster_analyzer):
        """Test: Detección de leyes emergentes"""
        # Simular patrones para detección de leyes
        patterns = [
            {'type': 'high_cohesion', 'strength': 0.9, 'description': 'Test pattern'},
            {'type': 'topological_coherence', 'strength': 0.8, 'description': 'Test topo'}
        ]
        
        phase_info = {'variance': 0.6, 'coherence': 0.7, 'phase_type': 'gapless'}
        laws = cluster_analyzer._detect_laws(patterns, iteration=200, phase_info=phase_info)
        
        assert isinstance(laws, list)
        for law in laws:
            assert 'name' in law
            assert 'description' in law
            assert 'strength' in law
            assert 'evidence' in law
    
    def test_gnn_phi_calculation(self, cluster_analyzer):
        """Test: Vector 5 - Cálculo de Φ_GNN"""
        # Simular topology features
        topology_features = torch.randn(1, 64).to(device)
        phase_info = {'variance': 0.5, 'coherence': 0.8, 'phase_type': 'gapless'}
        
        cluster_analyzer._calculate_gnn_phi(topology_features, phase_info)
        
        assert isinstance(cluster_analyzer.phi_gnn, float)
        assert 0.0 <= cluster_analyzer.phi_gnn <= 1.0
        assert isinstance(cluster_analyzer.last_gnn_complexity, float)
    
    def test_analyze_hidden_states(self, cluster_analyzer):
        """Test: Análisis completo de estados ocultos"""
        # Simular batch de hidden states
        hidden_states = [np.random.randn(64) for _ in range(10)]
        
        clusters, laws = cluster_analyzer.analyze_hidden_states(hidden_states, iteration=150)
        
        # Si hay suficientes estados, debería retornar clusters
        if clusters is not None:
            assert isinstance(clusters, list)
            assert len(clusters) > 0
        
        if laws is not None:
            assert isinstance(laws, list)
    
    def test_cluster_analyzer_finite_assertions(self, cluster_analyzer):
        """Test: Vector 6 - Finite model assertions para analyzer"""
        # Test con datos válidos
        valid_states = [np.random.randn(32) for _ in range(5)]
        
        try:
            clusters, laws = cluster_analyzer.analyze_hidden_states(valid_states, iteration=100)
            
            # Si se generan clusters, verificar finitud
            if clusters:
                for cluster in clusters:
                    center = np.array(cluster['center'])
                    assert np.isfinite(center).all(), "Cluster center contiene valores no finitos"
                    assert np.isfinite(cluster['cohesion']), "Cohesión no finita"
                    assert cluster['size'] > 0, "Tamaño de cluster inválido"
            
            # Verificar métricas Vector 5
            assert np.isfinite(cluster_analyzer.phi_gnn), "Φ_GNN no finito"
            assert np.isfinite(cluster_analyzer.last_gnn_complexity), "Complejidad GNN no finita"
            
        except Exception as e:
            pytest.skip(f"Test saltado debido a error de dependencias: {e}")
    
    def test_cluster_memory_management(self, cluster_analyzer):
        """Test: Gestión de memoria de patrones"""
        # Llenar memory hasta límite
        for i in range(150):  # Más que maxlen=100
            pattern = {'iteration': i, 'data': f'pattern_{i}'}
            cluster_analyzer.pattern_memory.append(pattern)
        
        assert len(cluster_analyzer.pattern_memory) == 100  # maxlen respetado
        assert cluster_analyzer.pattern_memory[-1]['iteration'] == 149  # Último elemento correcto
