#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 LIMPIEZA INTELIGENTE DEL PROYECTO INFINITO
==============================================

Elimina archivos obsoletos, mantiene solo lo esencial:
- Checkpoints: Solo best models
- Logs: Solo resultados recientes
- Scripts: Archiva código legacy
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

# Colores para output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def format_size(size_bytes):
    """Convierte bytes a formato legible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def get_folder_size(folder):
    """Calcula tamaño total de una carpeta."""
    total = 0
    try:
        for entry in os.scandir(folder):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_folder_size(entry.path)
    except Exception:
        pass
    return total

class ProjectCleaner:
    def __init__(self, project_root='.', dry_run=True):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'space_freed': 0
        }
        
        # Checkpoints a MANTENER (los demás se eliminan)
        self.keep_checkpoints = {
            'infinito_phase2_best.pt',  # Fase 2 IIT Transformer
            'infinito_v5.2_real_best.pt',  # V5.2 mejor resultado
            'infinito_gpt2_best.pt',  # GPT-2 + IIT híbrido
        }
        
        # Scripts ACTIVOS (el resto se archiva)
        self.active_scripts = {
            # Entrenamiento principal
            'train_phase2_iit_transformer.py',
            'train_v5_2_gpt2_lora.py',
            
            # Generación
            'generate_phase2_text.py',
            'generate_text_v5_2.py',
            
            # Testing
            'test_phase2_integration.py',
            'test_iit_v2_integration.py',
            
            # Core
            'infinito_v5_2_refactored.py',
            
            # Utilidades
            'cleanup_project_smart.py',
        }
        
        # Archivos de resultados recientes (mantener solo últimos 30 días)
        self.results_retention_days = 30
    
    def scan_checkpoints(self):
        """Escanea y categoriza checkpoints."""
        checkpoints_dir = self.project_root / 'models' / 'checkpoints'
        if not checkpoints_dir.exists():
            return
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔍 ESCANEANDO CHECKPOINTS{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        to_keep = []
        to_remove = []
        
        for pt_file in checkpoints_dir.glob('*.pt'):
            size = pt_file.stat().st_size
            
            if pt_file.name in self.keep_checkpoints:
                to_keep.append((pt_file, size))
                print(f"{Colors.GREEN}✓ MANTENER:{Colors.RESET} {pt_file.name} ({format_size(size)})")
            else:
                to_remove.append((pt_file, size))
                print(f"{Colors.RED}✗ ELIMINAR:{Colors.RESET} {pt_file.name} ({format_size(size)})")
        
        print(f"\n{Colors.BOLD}Resumen:{Colors.RESET}")
        print(f"  Mantener: {len(to_keep)} archivos ({format_size(sum(s for _, s in to_keep))})")
        print(f"  Eliminar: {len(to_remove)} archivos ({format_size(sum(s for _, s in to_remove))})")
        
        return to_remove
    
    def scan_legacy_checkpoints(self):
        """Escanea checkpoints legacy en otras carpetas."""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🗂️ ESCANEANDO CHECKPOINTS LEGACY{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        to_remove = []
        
        # V5.1 consciousness checkpoints (obsoletos)
        patterns = [
            'models/checkpoints/infinito_v5_1_consciousness_*.pt',
            'src/infinito_v5_1_consciousness_*.pt',
            'models/interrupted_states/*.pt',
            'src/interrupted_state_*.pt',
            'models/snapshots/*.pt',
            'models/CONSCIOUSNESS_BREAKTHROUGH_*.pt',
            'src/CONSCIOUSNESS_BREAKTHROUGH_*.pt',
            'models/PATTERN_BREAKTHROUGH_*.pt',
            'src/PATTERN_BREAKTHROUGH_*.pt',
        ]
        
        for pattern in patterns:
            for pt_file in self.project_root.glob(pattern):
                size = pt_file.stat().st_size
                to_remove.append((pt_file, size))
                print(f"{Colors.RED}✗ ELIMINAR:{Colors.RESET} {pt_file.relative_to(self.project_root)} ({format_size(size)})")
        
        if to_remove:
            print(f"\n{Colors.BOLD}Total legacy:{Colors.RESET} {len(to_remove)} archivos ({format_size(sum(s for _, s in to_remove))})")
        else:
            print(f"{Colors.GREEN}No se encontraron checkpoints legacy{Colors.RESET}")
        
        return to_remove
    
    def scan_old_results(self):
        """Escanea archivos de resultados antiguos."""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}📊 ESCANEANDO RESULTADOS ANTIGUOS{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        to_remove = []
        results_dir = self.project_root / 'results' / 'training'
        
        if not results_dir.exists():
            return to_remove
        
        cutoff_date = datetime.now().timestamp() - (self.results_retention_days * 24 * 3600)
        
        for json_file in results_dir.glob('*.json'):
            mtime = json_file.stat().st_mtime
            size = json_file.stat().st_size
            
            if mtime < cutoff_date:
                age_days = (datetime.now().timestamp() - mtime) / (24 * 3600)
                to_remove.append((json_file, size))
                print(f"{Colors.YELLOW}✗ ANTIGUO:{Colors.RESET} {json_file.name} ({int(age_days)} días, {format_size(size)})")
        
        if to_remove:
            print(f"\n{Colors.BOLD}Total antiguos:{Colors.RESET} {len(to_remove)} archivos ({format_size(sum(s for _, s in to_remove))})")
        else:
            print(f"{Colors.GREEN}Todos los resultados son recientes{Colors.RESET}")
        
        return to_remove
    
    def scan_legacy_scripts(self):
        """Escanea scripts legacy."""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}📜 ESCANEANDO SCRIPTS LEGACY{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        to_archive = []
        
        # Scripts de prueba/debug obsoletos
        legacy_patterns = [
            'debug_*.py',
            'test_*_old.py',
            'demo_*.py',
            'batch_infinito_*.py',
            'analyze_*.py',
            'inspect_*.py',
            'comparative_*.py',
            'consciousness_breakthrough_*.py',
            'causal_*.py',
            'quantum_*.py',
            'anti_hallucination*.py',
            'improve_*.py',
            'migrate_*.py',
            'monitor_*.py',
            'quick_*.py',
            'run_*.py',
        ]
        
        for pattern in legacy_patterns:
            for py_file in self.project_root.glob(pattern):
                if py_file.name not in self.active_scripts:
                    size = py_file.stat().st_size
                    to_archive.append((py_file, size))
                    print(f"{Colors.YELLOW}📦 ARCHIVAR:{Colors.RESET} {py_file.name} ({format_size(size)})")
        
        if to_archive:
            print(f"\n{Colors.BOLD}Total a archivar:{Colors.RESET} {len(to_archive)} archivos ({format_size(sum(s for _, s in to_archive))})")
        else:
            print(f"{Colors.GREEN}No hay scripts para archivar{Colors.RESET}")
        
        return to_archive
    
    def execute_cleanup(self, checkpoints, legacy_checkpoints, old_results, legacy_scripts):
        """Ejecuta la limpieza."""
        if self.dry_run:
            print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}")
            print(f"{Colors.MAGENTA}🔍 MODO DRY-RUN (sin cambios reales){Colors.RESET}")
            print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}")
            return
        
        print(f"\n{Colors.RED}{'='*70}{Colors.RESET}")
        print(f"{Colors.RED}🗑️ EJECUTANDO LIMPIEZA{Colors.RESET}")
        print(f"{Colors.RED}{'='*70}{Colors.RESET}\n")
        
        # Eliminar checkpoints obsoletos
        for file_path, size in checkpoints + legacy_checkpoints:
            try:
                file_path.unlink()
                self.stats['files_removed'] += 1
                self.stats['space_freed'] += size
                print(f"{Colors.GREEN}✓ Eliminado:{Colors.RESET} {file_path.name}")
            except Exception as e:
                print(f"{Colors.RED}✗ Error eliminando {file_path.name}: {e}{Colors.RESET}")
        
        # Eliminar resultados antiguos
        for file_path, size in old_results:
            try:
                file_path.unlink()
                self.stats['files_removed'] += 1
                self.stats['space_freed'] += size
                print(f"{Colors.GREEN}✓ Eliminado:{Colors.RESET} {file_path.name}")
            except Exception as e:
                print(f"{Colors.RED}✗ Error eliminando {file_path.name}: {e}{Colors.RESET}")
        
        # Archivar scripts legacy
        archive_dir = self.project_root / 'archive' / 'legacy_scripts'
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path, size in legacy_scripts:
            try:
                dest = archive_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                self.stats['files_removed'] += 1
                print(f"{Colors.GREEN}✓ Archivado:{Colors.RESET} {file_path.name} → archive/")
            except Exception as e:
                print(f"{Colors.RED}✗ Error archivando {file_path.name}: {e}{Colors.RESET}")
        
        # Limpiar directorios vacíos
        empty_dirs = [
            'models/interrupted_states',
            'models/snapshots',
        ]
        
        for dir_path in empty_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and not any(full_path.iterdir()):
                try:
                    full_path.rmdir()
                    self.stats['dirs_removed'] += 1
                    print(f"{Colors.GREEN}✓ Directorio vacío eliminado:{Colors.RESET} {dir_path}")
                except Exception as e:
                    print(f"{Colors.YELLOW}⚠ No se pudo eliminar {dir_path}: {e}{Colors.RESET}")
    
    def print_summary(self):
        """Imprime resumen final."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}📊 RESUMEN DE LIMPIEZA{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        if self.dry_run:
            print(f"{Colors.YELLOW}⚠️  Este fue un análisis (DRY-RUN). No se eliminó nada.{Colors.RESET}")
            print(f"{Colors.YELLOW}   Ejecuta con --execute para aplicar cambios.{Colors.RESET}\n")
        else:
            print(f"{Colors.GREEN}✅ Limpieza completada{Colors.RESET}\n")
        
        print(f"{Colors.BOLD}Estadísticas:{Colors.RESET}")
        print(f"  Archivos eliminados: {self.stats['files_removed']}")
        print(f"  Directorios eliminados: {self.stats['dirs_removed']}")
        print(f"  Espacio liberado: {format_size(self.stats['space_freed'])}")
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}Archivos mantenidos:{Colors.RESET}")
        for checkpoint in self.keep_checkpoints:
            print(f"  ✓ {checkpoint}")
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}Scripts activos:{Colors.RESET}")
        for script in sorted(self.active_scripts):
            if (self.project_root / script).exists():
                print(f"  ✓ {script}")
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}\n")
    
    def run(self):
        """Ejecuta la limpieza completa."""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}🧹 LIMPIEZA INTELIGENTE - PROYECTO INFINITO{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}\n")
        
        # Escanear
        checkpoints = self.scan_checkpoints()
        legacy_checkpoints = self.scan_legacy_checkpoints()
        old_results = self.scan_old_results()
        legacy_scripts = self.scan_legacy_scripts()
        
        # Calcular espacio total a liberar
        total_to_free = sum(
            s for _, s in (checkpoints + legacy_checkpoints + old_results)
        )
        
        print(f"\n{Colors.BOLD}{Colors.YELLOW}⚠️  RESUMEN PRE-LIMPIEZA:{Colors.RESET}")
        print(f"  Total archivos a eliminar: {len(checkpoints) + len(legacy_checkpoints) + len(old_results)}")
        print(f"  Total archivos a archivar: {len(legacy_scripts)}")
        print(f"  Espacio a liberar: {format_size(total_to_free)}")
        
        if not self.dry_run:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠️  ADVERTENCIA: Esta operación es IRREVERSIBLE{Colors.RESET}")
            response = input(f"\n¿Continuar con la limpieza? (escribe 'SI' para confirmar): ")
            if response != 'SI':
                print(f"\n{Colors.YELLOW}Limpieza cancelada{Colors.RESET}")
                return
        
        # Ejecutar
        self.execute_cleanup(checkpoints, legacy_checkpoints, old_results, legacy_scripts)
        
        # Resumen
        self.print_summary()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Limpieza inteligente del proyecto INFINITO')
    parser.add_argument('--execute', action='store_true', help='Ejecutar limpieza (sin este flag es dry-run)')
    parser.add_argument('--keep-days', type=int, default=30, help='Días de resultados a mantener (default: 30)')
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner(dry_run=not args.execute)
    cleaner.results_retention_days = args.keep_days
    cleaner.run()


if __name__ == '__main__':
    main()
