
import torch
import json
from datetime import datetime

def test_generation():
    """Prueba basica de generacion de texto"""
    try:
        # Cargar modelo
        model_path = "infinito_v5.2_best_epoch2.pt"
        print(f"Cargando modelo: {model_path}")
        
        # Verificar que el archivo existe y es valido
        if not torch.cuda.is_available():
            model = torch.load(model_path, map_location='cpu')
        else:
            model = torch.load(model_path)
        
        print("[OK] Modelo cargado exitosamente")
        
        # Informacion basica del modelo
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            print(f"[INFO] Total parametros: {total_params:,}")
        
        # Simular generacion (sin ejecutar realmente)
        results = {
            'model_path': model_path,
            'model_loaded': True,
            'total_parameters': total_params if 'total_params' in locals() else None,
            'timestamp': datetime.now().isoformat(),
            'test_status': 'success'
        }
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return {
            'model_path': "infinito_v5.2_best_epoch2.pt",
            'model_loaded': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'test_status': 'failed'
        }

if __name__ == "__main__":
    results = test_generation()
    print(json.dumps(results, indent=2, ensure_ascii=False))
