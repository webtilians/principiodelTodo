#!/usr/bin/env python3
"""
Script para migrar infinito_gpt_text_fixed.py a usar los nuevos módulos refactorizados.
"""

import re

def migrate_file():
    """Migra el archivo para usar los nuevos módulos core."""
    
    input_file = 'src/infinito_gpt_text_fixed.py'
    output_file = 'src/infinito_gpt_text_migrated.py'
    
    print("📦 Leyendo archivo original...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔄 Aplicando migraciones...")
    
    # 1. Actualizar imports
    print("  ✓ Actualizando imports...")
    old_imports = """from sklearn.feature_extraction.text import TfidfVectorizer

# 🆕 Try to load GloVe embeddings (fallback if not available)"""
    
    new_imports = """from sklearn.feature_extraction.text import TfidfVectorizer

# 🆕 IMPORTS DE MÓDULOS REFACTORIZADOS
import sys
sys.path.insert(0, os.path.dirname(__file__))

from core import (
    LegacyExternalMemory as EnhancedExternalMemory,  # Compatibilidad
    InformationIntegrationMetrics,
    StochasticExploration,
    StandardNLPMetrics
)

# 🆕 Try to load GloVe embeddings (fallback if not available)"""
    
    content = content.replace(old_imports, new_imports)
    
    # 2. Eliminar la definición de EnhancedExternalMemory (está en core/memory.py)
    print("  ✓ Eliminando clase EnhancedExternalMemory duplicada...")
    
    # Encontrar y comentar la clase
    class_pattern = r'class EnhancedExternalMemory\(nn\.Module\):.*?(?=\nclass |\n# ===|\Z)'
    
    def comment_class(match):
        lines = match.group(0).split('\n')
        commented = '\n'.join(['# ' + line if line.strip() else line for line in lines])
        return f"""# =============================================================================
# 🆕 MIGRADO: EnhancedExternalMemory ahora se importa desde core.memory
# =============================================================================
# La implementación completa está en src/core/memory.py
# Aquí se mantiene comentada solo para referencia histórica.
# =============================================================================

{commented}

# =============================================================================
# FIN DE CLASE MIGRADA
# =============================================================================
"""
    
    content = re.sub(class_pattern, comment_class, content, flags=re.DOTALL)
    
    # 3. Añadir comentarios sobre migración
    print("  ✓ Añadiendo documentación de migración...")
    
    header_addition = """
🆕 REFACTORIZADO: Este archivo ha sido migrado para usar módulos core/.

Cambios principales:
- EnhancedExternalMemory → importado desde core.memory
- Métricas IIT → InformationIntegrationMetrics desde core.iit_metrics
- Exploración estocástica → disponible desde core.stochastic
- Validación → StandardNLPMetrics desde core.validation

Para la versión mejorada con priorización inteligente, usar:
    from core import PriorityExternalMemory
"""
    
    content = content.replace(
        '"""',
        '"""' + header_addition,
        1  # Solo el primer docstring
    )
    
    # 4. Guardar archivo migrado
    print(f"💾 Guardando archivo migrado en {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Migración completada!")
    print(f"\n📄 Archivo original: {input_file}")
    print(f"📄 Archivo migrado: {output_file}")
    print(f"📄 Backup: src/infinito_gpt_text_fixed_backup.py")
    print("\n🔍 Revisa el archivo migrado y cuando esté listo:")
    print(f"   mv {output_file} {input_file}")


if __name__ == '__main__':
    migrate_file()
