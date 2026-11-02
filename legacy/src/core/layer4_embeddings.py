"""
Layer 4 Conscious Embeddings - INFINITO
========================================
Extrae embeddings del transformer con métricas PHI.

Análisis empírico mostró:
- Capa 4: Máximo PHI (8.91) - punto de máxima integración de información
- Capa 11: Mejor discriminación semántica (para búsqueda vectorial)

Por defecto usa Capa 11 para embeddings (óptimo para búsqueda) pero
calcula PHI desde el PHI Observer (que analiza todas las capas).

Características:
- Attention-mask aware: Solo promedia tokens reales (ignora padding)
- Normalización L2: Embeddings unitarios para mejor cosine similarity
- Dimensión dinámica: Detecta n_embd del modelo (768 para GPT-2, otros para custom)

Uso:
    extractor = Layer4Extractor(model)  # Usa capa 11 por defecto
    embedding = extractor.extract("La consciencia es un fenómeno")
    # embedding.shape = (n_embd,) - vector normalizado para ChromaDB
    
    # O con capa 4 (máximo PHI pero peor búsqueda):
    extractor = Layer4Extractor(model, use_search_layer=False)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
import numpy as np


class Layer4Extractor:
    """
    Extractor de embeddings con métricas PHI.
    
    Análisis empírico muestra:
    - Capa 4: Máximo PHI (8.91) pero baja discriminación semántica
    - Capa 11: Mejor discriminación semántica (para búsqueda)
    
    Por defecto usa Capa 11 para embeddings (mejor búsqueda) pero
    calcula PHI desde Capa 4 para métricas.
    """
    
    PHI_LAYER = 4      # Capa con máximo PHI (para métricas)
    SEARCH_LAYER = 11  # Capa con mejor discriminación (para búsqueda)
    
    def __init__(self, model, device: str = None, use_search_layer: bool = True):
        """
        Args:
            model: InfinitoGPT2WithObserver o similar con .gpt2 y .tokenizer
            device: 'cuda' o 'cpu' (auto-detectado si None)
            use_search_layer: Si True, usa capa 11 para embeddings (mejor búsqueda)
                              Si False, usa capa 4 (máximo PHI pero peor búsqueda)
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = model.tokenizer
        self.use_search_layer = use_search_layer
        
        # Determinar capa objetivo
        self.TARGET_LAYER = self.SEARCH_LAYER if use_search_layer else self.PHI_LAYER
        
        # Hook storage
        # Ya no usamos hooks - usamos output_hidden_states directamente
        # (los hooks capturan salidas diferentes a hidden_states)
        
    def remove_hook(self):
        """Método de compatibilidad (ya no usa hooks)."""
        pass
            
    def _get_hidden_states(self, input_ids):
        """Obtiene hidden states de la capa objetivo."""
        with torch.no_grad():
            outputs = self.model.gpt2(
                input_ids=input_ids, 
                output_hidden_states=True
            )
        # hidden_states[0] = embeddings, hidden_states[i+1] = capa i
        return outputs.hidden_states[self.TARGET_LAYER + 1]
            
    def extract(self, text: str, pooling: str = 'mean', normalize: bool = True) -> np.ndarray:
        """
        Extrae embedding para un texto.
        
        Args:
            text: Texto de entrada
            pooling: Método de pooling ('mean', 'max', 'cls', 'last')
                - 'mean': Promedio ponderado de tokens reales (ignora padding)
                - 'max': Máximo por dimensión
                - 'cls': Primer token (estilo BERT)
                - 'last': Último token real (no padding)
            normalize: Si True, normaliza a L2 (mejor para cosine similarity)
        
        Returns:
            np.ndarray de shape (n_embd,) - embedding denso
        """
        # Tokenizar
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)  # Máscara de tokens reales
        
        # Obtener hidden states de la capa objetivo
        hidden = self._get_hidden_states(input_ids)  # [batch, seq_len, n_embd]
        
        # Aplicar pooling CON attention_mask para ignorar padding
        if pooling == 'mean':
            # Expandir máscara para que coincida con dimensiones (B, L, D)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            # Sumar hidden states * mascara
            sum_embeddings = torch.sum(hidden * input_mask_expanded, dim=1)
            # Sumar cantidad de tokens reales (evitar división por cero)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            # Promedio real (solo tokens útiles)
            embedding = sum_embeddings / sum_mask
        elif pooling == 'max':
            # Aplicar máscara: poner -inf en posiciones de padding antes de max
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            hidden_masked = hidden.clone()
            hidden_masked[input_mask_expanded == 0] = float('-inf')
            embedding = hidden_masked.max(dim=1).values
        elif pooling == 'cls':
            embedding = hidden[:, 0, :]  # Primer token
        elif pooling == 'last':
            # Último token REAL (no padding)
            seq_lengths = attention_mask.sum(dim=1) - 1  # Índice del último token
            batch_size = hidden.size(0)
            embedding = hidden[torch.arange(batch_size, device=self.device), seq_lengths]
        else:
            raise ValueError(f"Pooling '{pooling}' no soportado. Usa: mean, max, cls, last")
        
        embedding = embedding.squeeze().cpu().numpy()
        
        # Normalización L2 para mejor cosine similarity en búsquedas
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def extract_with_phi(self, text: str, pooling: str = 'mean', normalize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Extrae embedding + métricas PHI completas.
        
        Args:
            text: Texto de entrada
            pooling: Método de pooling
            normalize: Si True, normaliza a L2
        
        Returns:
            Tuple de (embedding, phi_metrics)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs, phi_metrics = self.model(input_ids, return_phi=True, use_memory=False)
            # También obtener hidden states para embedding
            gpt2_outputs = self.model.gpt2(input_ids=input_ids, output_hidden_states=True)
        
        # Obtener embedding de la capa objetivo
        hidden = gpt2_outputs.hidden_states[self.TARGET_LAYER + 1]
        
        # Aplicar pooling CON attention_mask
        if pooling == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_embeddings = torch.sum(hidden * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        elif pooling == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            hidden_masked = hidden.clone()
            hidden_masked[input_mask_expanded == 0] = float('-inf')
            embedding = hidden_masked.max(dim=1).values
        elif pooling == 'cls':
            embedding = hidden[:, 0, :]
        elif pooling == 'last':
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden.size(0)
            embedding = hidden[torch.arange(batch_size, device=self.device), seq_lengths]
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_embeddings = torch.sum(hidden * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        
        embedding = embedding.squeeze().cpu().numpy()
        
        # Normalización L2
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        # Convertir métricas PHI a dict serializable
        phi_dict = {
            'phi': phi_metrics['phi'].mean().item(),
            'temporal': phi_metrics['raw_components']['temporal'],
            'integration': phi_metrics['raw_components']['integration'],
            'complexity': phi_metrics['raw_components']['complexity'],
            'attention': phi_metrics['raw_components']['attention']
        }
        
        return embedding, phi_dict
    
    def extract_batch(self, texts: List[str], pooling: str = 'mean', normalize: bool = True) -> np.ndarray:
        """
        Extrae embeddings para múltiples textos.
        
        Args:
            texts: Lista de textos
            pooling: Método de pooling
            normalize: Si True, normaliza cada embedding a L2
            
        Returns:
            np.ndarray de shape (n_texts, n_embd)
        """
        embeddings = []
        for text in texts:
            emb = self.extract(text, pooling=pooling, normalize=normalize)
            embeddings.append(emb)
        return np.stack(embeddings)
    
    def compare_layers(self, text: str) -> Dict[int, float]:
        """
        Compara PHI de todas las capas para un texto dado.
        Útil para verificar que capa 4 sigue siendo óptima.
        
        Returns:
            Dict mapping layer_idx -> PHI value
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.gpt2(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True
            )
        
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
        attentions = outputs.attentions
        
        layer_phi = {}
        
        for layer_idx, (hidden, attn) in enumerate(zip(hidden_states, attentions)):
            # Calcular PHI simplificado por capa
            hidden_flat = hidden.view(hidden.size(0), -1)
            temporal = torch.sigmoid(hidden_flat.std()).item()
            
            if layer_idx > 0:
                prev_hidden = hidden_states[layer_idx - 1]
                prev_flat = prev_hidden.view(prev_hidden.size(0), -1)
                integration = ((torch.cosine_similarity(hidden_flat, prev_flat, dim=-1).mean().item() + 1) / 2)
            else:
                integration = 0.5
            
            complexity = torch.sigmoid(hidden.std(dim=-1).mean()).item()
            
            # Attention entropy
            if attn is not None:
                attn_float = attn.float()
                attn_flat = attn_float.mean(dim=1)
                attn_clamped = torch.clamp(attn_flat, min=1e-8)
                entropy = -torch.sum(attn_clamped * torch.log(attn_clamped), dim=-1)
                max_entropy = torch.log(torch.tensor(attn_flat.shape[-1], dtype=torch.float32))
                attention_div = (entropy / max_entropy).mean().item()
            else:
                attention_div = 0.5
            
            phi = (0.3 * temporal + 0.3 * integration + 0.2 * complexity + 0.2 * attention_div) * 10
            layer_phi[layer_idx] = phi
        
        return layer_phi
    
    def get_embedding_dim(self) -> int:
        """Retorna la dimensión del embedding (detectada dinámicamente del modelo)."""
        return self.model.gpt2.config.n_embd
    
    def __del__(self):
        """Cleanup: eliminar hook al destruir el objeto."""
        self.remove_hook()


class ConsciousEmbeddingModel:
    """
    Wrapper compatible con sentence-transformers para usar en ChromaDB.
    
    Uso con ChromaDB:
        from chromadb.utils import embedding_functions
        
        model = load_infinito_model()
        conscious_ef = ConsciousEmbeddingModel(model)
        
        collection = client.create_collection(
            name="conscious_embeddings",
            embedding_function=conscious_ef
        )
    """
    
    def __init__(self, model, pooling: str = 'mean'):
        self.extractor = Layer4Extractor(model)
        self.pooling = pooling
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Interface compatible con ChromaDB embedding functions.
        
        Args:
            texts: Lista de documentos a embeber
            
        Returns:
            Lista de embeddings (cada uno es List[float] de 768 elementos)
        """
        embeddings = self.extractor.extract_batch(texts, pooling=self.pooling)
        return embeddings.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Alias para compatibilidad con LangChain."""
        return self(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embebe una query individual."""
        return self.extractor.extract(text, pooling=self.pooling).tolist()


# Función de conveniencia para crear extractor
def create_layer4_extractor(
    model_path: str = "models/infinito_gpt2_spanish_phi.pt",
    use_search_layer: bool = True
):
    """
    Crea un extractor de embeddings conscientes.
    
    Args:
        model_path: Ruta al checkpoint del modelo
        use_search_layer: Si True, usa capa 11 (mejor búsqueda). 
                          Si False, usa capa 4 (máximo PHI).
        
    Returns:
        Layer4Extractor listo para usar
    """
    import sys
    import os
    
    # Añadir path del proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Cargando modelo INFINITO...")
    model = InfinitoGPT2WithObserver()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✓ Modelo cargado desde {model_path}")
    
    model = model.to(device)
    model.eval()
    
    extractor = Layer4Extractor(model, device=device, use_search_layer=use_search_layer)
    layer_name = f"Capa {extractor.TARGET_LAYER}"
    purpose = "búsqueda semántica" if use_search_layer else "máximo PHI"
    print(f"✓ Extractor de {layer_name} listo ({purpose})")
    
    return extractor


if __name__ == "__main__":
    # Test del extractor
    print("="*60)
    print("  TEST: Conscious Embeddings (Capa 11 para búsqueda)")
    print("="*60)
    
    extractor = create_layer4_extractor()
    
    print(f"\nCapa objetivo: {extractor.TARGET_LAYER} (SEARCH_LAYER={extractor.SEARCH_LAYER})")
    
    # Test básico
    text = "La consciencia es un fenómeno emergente de la complejidad neuronal"
    print(f"\nTexto: '{text}'")
    
    # Dimensión dinámica
    emb_dim = extractor.get_embedding_dim()
    print(f"Dimensión del embedding (detectada): {emb_dim}")
    
    embedding = extractor.extract(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm (L2 normalizado): {np.linalg.norm(embedding):.4f}")
    print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    
    # Test de discriminación semántica
    print("\n" + "="*60)
    print("  Test de Discriminación Semántica")
    print("="*60)
    
    test_texts = [
        "PHI es una medida de consciencia",
        "Los transformers usan atención",
        "El cerebro tiene neuronas",
        "Hoy hace buen tiempo"
    ]
    
    embeddings_test = [extractor.extract(t) for t in test_texts]
    
    print("\nMatriz de similitud (Capa 11):")
    for i, t1 in enumerate(test_texts):
        for j, t2 in enumerate(test_texts):
            sim = np.dot(embeddings_test[i], embeddings_test[j])
            marker = "●" if i == j else ""
            print(f"  [{i}x{j}] {sim:.3f} {marker}")
        print()
    
    # Test con PHI
    embedding, phi = extractor.extract_with_phi(text)
    print(f"PHI metrics:")
    for k, v in phi.items():
        print(f"  {k}: {v:.4f}")
    
    # Comparar capas
    print("\n" + "="*60)
    print("  Comparación PHI por Capa")
    print("="*60)
    layer_phi = extractor.compare_layers(text)
    
    peak_layer = max(layer_phi, key=layer_phi.get)
    print(f"\n{'Capa':<8} {'PHI':<8} {'Barra'}")
    print("-"*40)
    for layer, phi_val in sorted(layer_phi.items()):
        bar = "█" * int(phi_val * 2)
        marker = " ← MAX PHI" if layer == peak_layer else ""
        marker += " ← SEARCH" if layer == extractor.SEARCH_LAYER else ""
        print(f"{layer:<8} {phi_val:.3f}   {bar}{marker}")
    
    print(f"\n✓ Capa máximo PHI: {peak_layer}")
    print(f"✓ Capa búsqueda: {extractor.SEARCH_LAYER}")
    
    # Cleanup
    extractor.remove_hook()
    print("\n✓ Test completado")
