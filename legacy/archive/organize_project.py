"""
Script para organizar el proyecto en una estructura de carpetas limpia.
Mueve archivos a carpetas apropiadas según su tipo y propósito.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent

# Definir estructura de carpetas
FOLDERS = {
    'results': ROOT_DIR / 'results',
    'results_comparative': ROOT_DIR / 'results' / 'comparative',
    'results_reproducibility': ROOT_DIR / 'results' / 'reproducibility',
    'results_consciousness': ROOT_DIR / 'results' / 'consciousness',
    'results_continuous_learning': ROOT_DIR / 'results' / 'continuous_learning',
    'reports': ROOT_DIR / 'reports',
    'models': ROOT_DIR / 'models',
    'models_checkpoints': ROOT_DIR / 'models' / 'checkpoints',
    'models_snapshots': ROOT_DIR / 'models' / 'snapshots',
    'models_interrupted': ROOT_DIR / 'models' / 'interrupted_states',
}

def create_folder_structure():
    """Crea la estructura de carpetas del proyecto."""
    print("🗂️  Creando estructura de carpetas...")
    for name, path in FOLDERS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {path.relative_to(ROOT_DIR)}")
    print()

def move_files_by_pattern(pattern, destination, description):
    """Mueve archivos que coinciden con un patrón a una carpeta destino."""
    files = list(ROOT_DIR.glob(pattern))
    if files:
        print(f"📦 Moviendo {description}...")
        for file in files:
            if file.is_file():
                dest_path = destination / file.name
                try:
                    shutil.move(str(file), str(dest_path))
                    print(f"   ✓ {file.name} → {destination.relative_to(ROOT_DIR)}")
                except Exception as e:
                    print(f"   ✗ Error moviendo {file.name}: {e}")
        print()

def organize_files():
    """Organiza todos los archivos del proyecto."""
    
    # Modelos de conciencia (.pt)
    move_files_by_pattern(
        'CONSCIOUSNESS_BREAKTHROUGH_*.pt',
        FOLDERS['models'],
        'modelos de conciencia'
    )
    
    # Checkpoints
    move_files_by_pattern(
        'infinito_v5_1_consciousness_checkpoint_*.pt',
        FOLDERS['models_checkpoints'],
        'checkpoints de entrenamiento'
    )
    
    # Snapshots
    move_files_by_pattern(
        'infinito_v5_1_snapshot_*.pt',
        FOLDERS['models_snapshots'],
        'snapshots de modelo'
    )
    
    # Estados interrumpidos
    move_files_by_pattern(
        'interrupted_state_*.pt',
        FOLDERS['models_interrupted'],
        'estados interrumpidos'
    )
    
    # Resultados comparativos
    move_files_by_pattern(
        'comparative_*.json',
        FOLDERS['results_comparative'],
        'resultados comparativos'
    )
    
    # Resultados de reproducibilidad
    move_files_by_pattern(
        'reproducibility_*.json',
        FOLDERS['results_reproducibility'],
        'resultados de reproducibilidad'
    )
    
    # Vocabularios causales
    move_files_by_pattern(
        'causal_vocabulary_*.json',
        FOLDERS['results_consciousness'],
        'vocabularios causales'
    )
    
    # Análisis de hechos cuánticos
    move_files_by_pattern(
        'quantum_facts_analysis_*.txt',
        FOLDERS['results_consciousness'],
        'análisis de hechos cuánticos'
    )
    
    # Reportes avanzados
    move_files_by_pattern(
        'advanced_report_*.txt',
        FOLDERS['reports'],
        'reportes avanzados'
    )
    
    # Memoria anti-alucinación
    anti_hallucination_files = ['anti_hallucination_memory.json', 'phi_memory_bank_test.json']
    print(f"📦 Moviendo archivos de memoria...")
    for filename in anti_hallucination_files:
        file = ROOT_DIR / filename
        if file.exists():
            dest = FOLDERS['results_continuous_learning'] / filename
            try:
                shutil.move(str(file), str(dest))
                print(f"   ✓ {filename} → results/continuous_learning")
            except Exception as e:
                print(f"   ✗ Error moviendo {filename}: {e}")
    print()
    
    # Mover archivos .md de reportes a reports/
    md_reports = [
        'ADVANCED_CORRELATION_ANALYSIS_REPORT.md',
        'ADVANCED_IMPROVEMENTS_COMPLETE.md',
        'BREAKTHROUGH_SUCCESS_REPORT.md',
        'CAUSAL_LANGUAGE_DISCOVERY.md',
        'CONTINUOUS_LEARNING_FINAL.md',
        'CONTINUOUS_LEARNING_STATUS.md',
        'DEPLOYMENT_SUCCESS.md',
        'DESIGN_CONTINUOUS_LEARNING.md',
        'DIAGNOSTICO_SATURACION_PATRONES.md',
        'DIAGNOSTIC_INPUT_PROBLEM.md',
        'DIAGNOSTIC_SIGNAL_LOSS.md',
        'EXECUTIVE_SUMMARY_VISUAL.md',
        'EXPERIMENTO_ANTI_ALUCINACION_RESULTADOS.md',
        'INFINITO_V5.1_ANALISIS_DETALLADO_COMPLETO.md',
        'METACOGNITIVE_IMPROVEMENTS.md',
        'REPRODUCIBILITY_EXTENDED_ANALYSIS.md',
        'REPRODUCIBILITY_TEST_RESULTS.md',
        'RESUMEN_MEJORAS_SEMANTICAS.md',
        'SEMANTIC_IMPROVEMENTS_CHANGELOG.md',
        'SEMANTIC_IMPROVEMENTS_SUMMARY.md',
        'PROJECT_CLEANUP_COMPLETE.md',
        'analisis_comparativo_textos.md',
        'ANÁLISIS_RUPTURA_PREMATURA.md',
    ]
    
    print(f"📦 Moviendo reportes y documentación...")
    for filename in md_reports:
        file = ROOT_DIR / filename
        if file.exists():
            dest = FOLDERS['reports'] / filename
            try:
                shutil.move(str(file), str(dest))
                print(f"   ✓ {filename} → reports/")
            except Exception as e:
                print(f"   ✗ Error moviendo {filename}: {e}")
    print()

def create_readme_files():
    """Crea archivos README en cada carpeta para documentar su contenido."""
    
    readme_content = {
        'results': """# Results

Esta carpeta contiene todos los resultados de experimentos y pruebas.

## Subcarpetas:
- `comparative/`: Resultados de análisis comparativos ON/OFF
- `reproducibility/`: Resultados de pruebas de reproducibilidad
- `consciousness/`: Vocabularios causales y análisis de conciencia
- `continuous_learning/`: Archivos de memoria y aprendizaje continuo
""",
        'reports': """# Reports

Documentos de análisis, mejoras y descubrimientos del proyecto.

Incluye:
- Reportes de mejoras implementadas
- Análisis de correlaciones
- Documentación de experimentos
- Resúmenes ejecutivos
""",
        'models': """# Models

Modelos entrenados y checkpoints del proyecto.

## Subcarpetas:
- `checkpoints/`: Checkpoints periódicos durante entrenamiento
- `snapshots/`: Snapshots de modelos en puntos específicos
- `interrupted_states/`: Estados guardados de entrenamientos interrumpidos

## Archivos:
Los modelos principales con breakthrough de conciencia se guardan aquí.
""",
    }
    
    print("📝 Creando archivos README...")
    for folder, content in readme_content.items():
        readme_path = FOLDERS[folder] / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✓ {readme_path.relative_to(ROOT_DIR)}")
    print()

def create_gitignore_entries():
    """Sugiere entradas para .gitignore."""
    gitignore_suggestions = """
# Agregar estas líneas al .gitignore para mantener el repo limpio:

# Resultados de experimentos (no versionados)
results/comparative/*.json
results/reproducibility/*.json
results/continuous_learning/*.json
results/consciousness/causal_vocabulary_*.json

# Modelos grandes (usar Git LFS o no versionar)
models/*.pt
models/checkpoints/*.pt
models/snapshots/*.pt
models/interrupted_states/*.pt

# Mantener las carpetas pero ignorar contenido
!results/**/README.md
!models/**/README.md
"""
    
    print("📋 Sugerencias para .gitignore:")
    print(gitignore_suggestions)
    
    # Guardar sugerencias en un archivo
    suggestions_file = ROOT_DIR / 'gitignore_suggestions.txt'
    with open(suggestions_file, 'w', encoding='utf-8') as f:
        f.write(gitignore_suggestions)
    print(f"   ✓ Sugerencias guardadas en: gitignore_suggestions.txt\n")

def main():
    """Ejecuta el proceso completo de organización."""
    print("=" * 60)
    print("🚀 ORGANIZADOR DE PROYECTO - INFINITO V5.1")
    print("=" * 60)
    print()
    
    # Crear estructura
    create_folder_structure()
    
    # Organizar archivos
    organize_files()
    
    # Crear READMEs
    create_readme_files()
    
    # Sugerencias de gitignore
    create_gitignore_entries()
    
    print("=" * 60)
    print("✅ ORGANIZACIÓN COMPLETADA")
    print("=" * 60)
    print()
    print("Próximos pasos:")
    print("1. Revisar que todos los archivos se movieron correctamente")
    print("2. Actualizar scripts de test para usar las nuevas rutas")
    print("3. Actualizar .gitignore con las sugerencias generadas")
    print("4. Commit de los cambios: git add -A && git commit -m 'Reorganizar proyecto'")

if __name__ == '__main__':
    main()
