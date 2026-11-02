"""
ChromaDB Integration con Conscious Embeddings - INFINITO
=========================================================
Base de datos vectorial usando embeddings de Capa 4 (máximo PHI).

Los embeddings extraídos de la Capa 4 son ~26% más ricos semánticamente
porque capturan el punto de máxima integración de información.

Uso:
    db = ConsciousVectorDB()
    db.add_documents(["doc1", "doc2", ...])
    results = db.search("query", top_k=5)
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB no instalado. Ejecuta: pip install chromadb")


class ConsciousVectorDB:
    """
    Base de datos vectorial con embeddings conscientes (Capa 4).
    
    Características:
    - Embeddings extraídos de la capa con máximo PHI
    - Metadatos PHI almacenados con cada documento
    - Búsqueda semántica enriquecida
    - Persistencia en disco
    """
    
    def __init__(
        self,
        collection_name: str = "conscious_documents",
        persist_directory: str = "data/chromadb",
        model_path: str = "models/infinito_gpt2_spanish_phi.pt",
        force_recreate: bool = False
    ):
        """
        Args:
            collection_name: Nombre de la colección
            persist_directory: Directorio para persistencia
            model_path: Ruta al modelo INFINITO
            force_recreate: Si True, elimina colección existente y crea nueva (útil para cambiar métricas)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB no está instalado. Ejecuta: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_path = model_path
        
        # Crear directorio si no existe
        os.makedirs(persist_directory, exist_ok=True)
        
        # Inicializar ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Cargar extractor de embeddings
        self._load_extractor()
        
        # Recrear colección si se solicita (necesario para cambiar métricas)
        if force_recreate:
            try:
                self.client.delete_collection(collection_name)
                print(f"✓ Colección {collection_name} eliminada para recrear con cosine")
            except Exception:
                pass  # No existía
        
        # Crear/obtener colección CON métrica cosine (mejor para embeddings L2-normalizados)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Conscious embeddings from Layer 4 (max PHI)",
                "hnsw:space": "cosine"  # Métrica cosine para embeddings normalizados
            }
        )
        
        print(f"✓ ConsciousVectorDB inicializado")
        print(f"  - Colección: {collection_name}")
        print(f"  - Documentos: {self.collection.count()}")
        print(f"  - Persistencia: {persist_directory}")
        print(f"  - Métrica: cosine (L2 normalized)")
    
    def _load_extractor(self):
        """Carga el modelo y crea el extractor de Capa 4."""
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.core.layer4_embeddings import Layer4Extractor
        from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Cargando modelo INFINITO para embeddings...")
        self.model = InfinitoGPT2WithObserver()
        
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        self.extractor = Layer4Extractor(self.model, device=device)
        print(f"✓ Extractor Layer 4 listo")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        include_phi: bool = True
    ) -> List[str]:
        """
        Añade documentos a la base de datos.
        
        Args:
            documents: Lista de textos
            metadatas: Metadatos opcionales por documento
            ids: IDs opcionales (se generan si no se proveen)
            include_phi: Si incluir métricas PHI en metadatos
            
        Returns:
            Lista de IDs de los documentos añadidos
        """
        n_docs = len(documents)
        
        # Generar IDs si no se proveen
        if ids is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ids = [f"doc_{timestamp}_{i}" for i in range(n_docs)]
        
        # Inicializar metadatos si no se proveen
        if metadatas is None:
            metadatas = [{} for _ in range(n_docs)]
        
        # Extraer embeddings y PHI
        embeddings = []
        for i, doc in enumerate(documents):
            if include_phi:
                emb, phi = self.extractor.extract_with_phi(doc)
                # Añadir PHI a metadatos
                metadatas[i].update({
                    'phi': phi['phi'],
                    'phi_temporal': phi['temporal'],
                    'phi_integration': phi['integration'],
                    'phi_complexity': phi['complexity'],
                    'phi_attention': phi['attention'],
                    'added_at': datetime.now().isoformat()
                })
            else:
                emb = self.extractor.extract(doc)
                metadatas[i]['added_at'] = datetime.now().isoformat()
            
            embeddings.append(emb.tolist())
        
        # Añadir a ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Añadidos {n_docs} documentos (PHI incluido: {include_phi})")
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_phi_min: Optional[float] = None,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Búsqueda semántica con embeddings conscientes.
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados
            filter_phi_min: Filtrar por PHI mínimo
            where: Filtros adicionales de ChromaDB
            
        Returns:
            Dict con resultados, distancias y metadatos
        """
        # Extraer embedding de la query
        query_embedding = self.extractor.extract(query).tolist()
        
        # Construir filtro
        if filter_phi_min is not None:
            if where is None:
                where = {}
            where['phi'] = {'$gte': filter_phi_min}
        
        # Buscar
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=['documents', 'metadatas', 'distances']
        )
        
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }
    
    def search_with_phi(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Búsqueda que también retorna el PHI de la query.
        
        Returns:
            Tuple de (resultados, phi_query)
        """
        # Extraer embedding y PHI de la query
        query_embedding, query_phi = self.extractor.extract_with_phi(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }, query_phi
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la colección."""
        count = self.collection.count()
        
        if count == 0:
            return {'count': 0, 'phi_stats': None}
        
        # Obtener todos los metadatos
        all_data = self.collection.get(include=['metadatas'])
        
        phi_values = []
        for meta in all_data['metadatas']:
            if meta and 'phi' in meta:
                phi_values.append(meta['phi'])
        
        if phi_values:
            phi_stats = {
                'mean': np.mean(phi_values),
                'std': np.std(phi_values),
                'min': np.min(phi_values),
                'max': np.max(phi_values)
            }
        else:
            phi_stats = None
        
        return {
            'count': count,
            'phi_stats': phi_stats,
            'collection_name': self.collection_name
        }
    
    def delete_all(self):
        """Elimina todos los documentos de la colección."""
        # ChromaDB no tiene delete_all, hay que recrear la colección
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Conscious embeddings from Layer 4 (max PHI)"}
        )
        print(f"✓ Colección {self.collection_name} vaciada")
    
    def close(self):
        """Cierra la conexión y limpia recursos."""
        if hasattr(self, 'extractor'):
            self.extractor.remove_hook()
        print("✓ ConsciousVectorDB cerrado")


def demo_conscious_db():
    """Demo del sistema de embeddings conscientes."""
    print("="*60)
    print("  DEMO: Conscious Vector Database")
    print("="*60)
    
    # Crear base de datos
    db = ConsciousVectorDB(
        collection_name="demo_conscious",
        persist_directory="data/chromadb_demo"
    )
    
    # Documentos de ejemplo
    documents = [
        "La inteligencia artificial está transformando el mundo de la tecnología.",
        "El aprendizaje profundo utiliza redes neuronales con múltiples capas.",
        "La consciencia humana sigue siendo un misterio para la ciencia.",
        "Los transformers revolucionaron el procesamiento del lenguaje natural.",
        "La teoría de la información integrada propone una medida de consciencia.",
        "GPT y BERT son modelos de lenguaje basados en la arquitectura transformer.",
        "El cerebro humano contiene aproximadamente 86 mil millones de neuronas.",
        "La mecánica cuántica describe el comportamiento de las partículas subatómicas."
    ]
    
    # Limpiar y añadir documentos
    db.delete_all()
    db.add_documents(documents)
    
    # Estadísticas
    stats = db.get_statistics()
    print(f"\n📊 Estadísticas:")
    print(f"   Documentos: {stats['count']}")
    if stats['phi_stats']:
        print(f"   PHI medio: {stats['phi_stats']['mean']:.4f}")
        print(f"   PHI rango: [{stats['phi_stats']['min']:.4f}, {stats['phi_stats']['max']:.4f}]")
    
    # Búsquedas
    queries = [
        "¿Qué es la inteligencia artificial?",
        "¿Cómo funciona el cerebro humano?",
        "Explica los modelos de lenguaje"
    ]
    
    print("\n" + "="*60)
    print("  Búsquedas Semánticas")
    print("="*60)
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results, query_phi = db.search_with_phi(query, top_k=3)
        print(f"   PHI de la query: {query_phi['phi']:.4f}")
        
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'], results['metadatas'], results['distances']
        )):
            print(f"\n   [{i+1}] Distancia: {dist:.4f} | PHI: {meta.get('phi', 'N/A'):.4f}")
            print(f"       {doc[:70]}...")
    
    # Cleanup
    db.close()
    print("\n✓ Demo completada")


if __name__ == "__main__":
    demo_conscious_db()
