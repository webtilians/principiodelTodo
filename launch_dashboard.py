#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanzador del Dashboard INFINITO V5.2
====================================

Script para lanzar el dashboard de monitoreo fácilmente.
Instala dependencias automáticamente si es necesario.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_and_install_dependencies():
    """Verifica e instala dependencias necesarias"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[INSTALL] Instalando dependencias faltantes: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("[OK] Dependencias instaladas exitosamente")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] No se pudieron instalar las dependencias: {e}")
            return False
    
    return True

def launch_dashboard(port=8501, host="localhost"):
    """Lanza el dashboard de Streamlit"""
    dashboard_path = Path(__file__).parent / "dashboard_monitor.py"
    
    if not dashboard_path.exists():
        print(f"[ERROR] No se encontró el archivo del dashboard: {dashboard_path}")
        return False
    
    print(f"[START] Lanzando dashboard INFINITO V5.2...")
    print(f"[URL] http://{host}:{port}")
    print(f"[INFO] Para detener el dashboard, presiona Ctrl+C")
    print("-" * 50)
    
    try:
        # Lanzar Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(dashboard_path),
            '--server.port', str(port),
            '--server.headless', 'false',
            '--server.address', host,
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print(f"\n[STOP] Dashboard detenido por el usuario")
    except Exception as e:
        print(f"[ERROR] Error lanzando dashboard: {str(e)}")
        return False
    
    return True

def main():
    """Función principal"""
    print("="*60)
    print("🧠 INFINITO V5.2 - Dashboard Monitor")
    print("="*60)
    
    # Verificar dependencias
    if not check_and_install_dependencies():
        print("[ERROR] No se pudieron instalar las dependencias necesarias")
        sys.exit(1)
    
    # Obtener configuración de puerto
    import argparse
    parser = argparse.ArgumentParser(description='Lanzador del Dashboard INFINITO V5.2')
    parser.add_argument('--port', type=int, default=8501, help='Puerto del servidor')
    parser.add_argument('--host', type=str, default='localhost', help='Host del servidor')
    
    args = parser.parse_args()
    
    # Lanzar dashboard
    success = launch_dashboard(args.port, args.host)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()