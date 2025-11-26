#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 RELANZADOR DEL DASHBOARD - Sin errores de browser
===================================================

Script para relanzar el dashboard de forma limpia,
evitando errores de conexión del navegador.
"""

import subprocess
import sys
import time
import os

def kill_streamlit_processes():
    """Mata procesos de streamlit existentes para evitar conflictos."""
    try:
        import psutil
        streamlit_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'streamlit' in cmdline.lower() or 'dashboard_monitor.py' in cmdline:
                        streamlit_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if streamlit_processes:
            print(f"🔄 Cerrando {len(streamlit_processes)} procesos de Streamlit existentes...")
            for proc in streamlit_processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            time.sleep(2)
            print("✅ Procesos cerrados")
        else:
            print("✅ No hay procesos Streamlit activos")
            
    except ImportError:
        print("⚠️  psutil no disponible - salteando limpieza de procesos")

def launch_dashboard():
    """Lanza el dashboard con configuración optimizada."""
    
    print("🧠 INFINITO V5.2 - Dashboard Monitor")
    print("="*40)
    
    # Limpiar procesos existentes
    kill_streamlit_processes()
    
    # Encontrar puerto libre
    import socket
    
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    # Intentar puertos comunes primero
    preferred_ports = [8501, 8502, 8503, 8504]
    port = None
    
    for test_port in preferred_ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', test_port))
                if result != 0:  # Puerto libre
                    port = test_port
                    break
        except:
            pass
    
    if port is None:
        port = find_free_port()
    
    print(f"🌐 Lanzando dashboard en puerto: {port}")
    
    # Comando optimizado para Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "dashboard_monitor.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.runOnSave", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    print(f"🚀 Ejecutando: {' '.join(cmd)}")
    print(f"🔗 URL: http://localhost:{port}")
    print(f"{'='*40}")
    print("💡 Para detener: Ctrl+C")
    print("🔄 El dashboard se auto-actualiza cada 30 segundos")
    print(f"{'='*40}")
    
    try:
        # Lanzar Streamlit
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard detenido por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error lanzando dashboard: {e}")
        print("💡 Verifica que Streamlit esté instalado: pip install streamlit")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")

if __name__ == "__main__":
    try:
        launch_dashboard()
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        input("\nPresiona Enter para salir...")