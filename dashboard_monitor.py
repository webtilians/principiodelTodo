#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard de Monitoreo en Tiempo Real - INFINITO V5.2
=====================================================

Dashboard interactivo para monitorear entrenamientos, comparar modelos
y visualizar progreso en tiempo real.

Características:
- Monitoreo de entrenamiento en tiempo real
- Comparación de modelos lado a lado  
- Visualización de métricas IIT
- Alertas de convergencia
- Análisis de eficiencia
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import re

# Configurar página
st.set_page_config(
    page_title="INFINITO V5.2 Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InfinitoMonitor:
    """Monitor en tiempo real para entrenamientos INFINITO V5.2"""
    
    def __init__(self):
        self.refresh_interval = 30  # segundos
        self.models_dir = "models/checkpoints"
        self.logs_dir = "."
        self.results_dir = "."
        
    def detect_active_training(self):
        """Detecta entrenamientos activos en el sistema"""
        active_trainings = []
        
        try:
            import psutil
            current_time = time.time()
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
                try:
                    if proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        
                        # Detectar procesos de entrenamiento
                        if ('train_v5_2' in cmdline.lower() or 
                            'entrenamiento' in cmdline.lower() or
                            ('python' in cmdline.lower() and 'train' in cmdline.lower())):
                            
                            # Extraer información del proceso
                            runtime = current_time - proc.info['create_time']
                            memory_mb = proc.info['memory_info'].rss / (1024*1024) if proc.info['memory_info'] else 0
                            
                            # Extraer configuración del comando
                            config_info = {}
                            if '--model-size' in cmdline:
                                try:
                                    idx = cmdline.split().index('--model-size')
                                    if idx + 1 < len(cmdline.split()):
                                        config_info['model_size'] = cmdline.split()[idx + 1]
                                except:
                                    pass
                            
                            if '--epochs' in cmdline:
                                try:
                                    idx = cmdline.split().index('--epochs')
                                    if idx + 1 < len(cmdline.split()):
                                        config_info['epochs'] = cmdline.split()[idx + 1]
                                except:
                                    pass
                            
                            active_trainings.append({
                                'pid': proc.info['pid'],
                                'command': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                                'runtime_minutes': round(runtime / 60, 1),
                                'memory_mb': round(memory_mb, 1),
                                'config': config_info,
                                'status': 'running'
                            })
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
        except ImportError:
            # Si no hay psutil, no podemos detectar procesos
            pass
            
        return active_trainings
        
    def load_training_histories(self):
        """Carga historiales de entrenamiento disponibles"""
        histories = []
        
        # Buscar archivos de historial
        patterns = ["training_history*.json", "*history*.json", "*results*.json"]
        for pattern in patterns:
            for file_path in Path(self.logs_dir).glob(pattern):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extraer información básica
                    history_info = {
                        'file': str(file_path),
                        'name': file_path.stem,
                        'modified': file_path.stat().st_mtime,
                        'data': data
                    }
                    
                    histories.append(history_info)
                    
                except Exception as e:
                    st.sidebar.warning(f"Error cargando {file_path}: {str(e)}")
        
        # Ordenar por fecha de modificación
        histories.sort(key=lambda x: x['modified'], reverse=True)
        
        return histories
    
    def load_model_checkpoints(self):
        """Carga información de checkpoints de modelos"""
        checkpoints = []
        
        # Buscar archivos .pt
        for pattern in ["*.pt", f"{self.models_dir}/*.pt"]:
            for file_path in Path(".").glob(pattern):
                try:
                    import torch
                    
                    # Cargar metadatos básicos
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                    
                    checkpoint_info = {
                        'file': str(file_path),
                        'name': file_path.stem,
                        'size_mb': file_path.stat().st_size / (1024*1024),
                        'modified': file_path.stat().st_mtime,
                        'epoch': checkpoint.get('epoch', 'unknown'),
                        'val_loss': checkpoint.get('val_loss', float('inf')),
                        'val_ppl': checkpoint.get('val_ppl', float('inf')),
                        'config': checkpoint.get('config', {})
                    }
                    
                    checkpoints.append(checkpoint_info)
                    
                except Exception as e:
                    # Skip archivos corruptos o incompatibles
                    continue
        
        # Ordenar por fecha de modificación
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        
        return checkpoints
    
    def get_training_progress(self, history_data):
        """Extrae progreso de entrenamiento de datos de historial"""
        # Manejar diferentes formatos de datos
        if isinstance(history_data, list):
            # Si es una lista, tomar el primer elemento o convertir
            if len(history_data) > 0 and isinstance(history_data[0], dict):
                history_data = history_data[0]
            else:
                return None
        
        # Buscar datos de entrenamiento en diferentes ubicaciones
        training_history = None
        
        # Intentar diferentes estructuras de datos
        if isinstance(history_data, dict):
            # Estructura directa
            if 'train_loss' in history_data:
                training_history = history_data
            # Estructura anidada
            elif 'training_history' in history_data:
                training_history = history_data.get('training_history', {})
            # Estructura de historial de entrenamiento
            elif 'history' in history_data:
                training_history = history_data.get('history', {})
        
        if not training_history or not isinstance(training_history, dict):
            return None
        
        # Crear DataFrame con métricas
        df_data = {}
        
        # Métricas básicas
        for metric in ['train_loss', 'val_loss', 'train_perplexity', 'val_perplexity', 'learning_rate']:
            if metric in training_history:
                values = training_history[metric]
                # Asegurar que sea una lista
                if not isinstance(values, list):
                    if isinstance(values, (int, float)):
                        values = [values]
                    else:
                        continue
                df_data[metric] = values
        
        # Métricas IIT si están disponibles
        for metric in ['train_phi', 'train_loss_phi', 'delta_phi_loss', 'integration_phi']:
            if metric in training_history:
                values = training_history[metric]
                # Asegurar que sea una lista
                if not isinstance(values, list):
                    if isinstance(values, (int, float)):
                        values = [values]
                    else:
                        continue
                df_data[metric] = values
        
        if not df_data:
            return None
        
        # Asegurar que todas las listas tengan la misma longitud
        max_len = max(len(v) for v in df_data.values() if isinstance(v, list))
        for key in df_data:
            if isinstance(df_data[key], list) and len(df_data[key]) < max_len:
                # Rellenar con último valor
                last_val = df_data[key][-1] if df_data[key] else 0
                df_data[key].extend([last_val] * (max_len - len(df_data[key])))
        
        df = pd.DataFrame(df_data)
        df['epoch'] = range(1, len(df) + 1)
        
        return df
    
    def plot_training_progress(self, df, title="Progreso de Entrenamiento"):
        """Crea gráficos de progreso de entrenamiento"""
        if df is None or len(df) == 0:
            st.warning("No hay datos de entrenamiento disponibles")
            return
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pérdida', 'Perplexity', 'Métricas IIT', 'Convergencia'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Gráfico 1: Pérdida
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train_loss'], 
                          name='Train Loss', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['val_loss'], 
                          name='Val Loss', line=dict(color='red')),
                row=1, col=1
            )
        
        # Gráfico 2: Perplexity
        ppl_cols = [col for col in df.columns if 'perplexity' in col.lower() or 'ppl' in col.lower()]
        if len(ppl_cols) >= 2:
            train_ppl_col = next((col for col in ppl_cols if 'train' in col), ppl_cols[0])
            val_ppl_col = next((col for col in ppl_cols if 'val' in col), ppl_cols[1] if len(ppl_cols) > 1 else ppl_cols[0])
            
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df[train_ppl_col], 
                          name='Train PPL', line=dict(color='lightblue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df[val_ppl_col], 
                          name='Val PPL', line=dict(color='orange')),
                row=1, col=2
            )
        
        # Gráfico 3: Métricas IIT
        if 'train_phi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train_phi'], 
                          name='PHI Integration', line=dict(color='green')),
                row=2, col=1
            )
        
        if 'train_loss_phi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train_loss_phi'], 
                          name='ΔPhi Loss', line=dict(color='purple'),
                          yaxis="y2"),
                row=2, col=1, secondary_y=True
            )
        
        # Gráfico 4: Learning Rate
        if 'learning_rate' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['learning_rate'], 
                          name='Learning Rate', line=dict(color='darkorange')),
                row=2, col=2
            )
        elif 'val_loss' in df.columns and len(df) > 1:
            # Calcular mejora relativa como fallback
            improvement = [(df['val_loss'].iloc[0] - loss) / df['val_loss'].iloc[0] * 100 
                          for loss in df['val_loss']]
            
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=improvement, 
                          name='Mejora %', line=dict(color='darkgreen')),
                row=2, col=2
            )
        
        # Configurar layout
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True,
            template="plotly_white"
        )
        
        # Configurar ejes Y
        fig.update_yaxes(title_text="Pérdida", row=1, col=1)
        fig.update_yaxes(title_text="Perplexity", row=1, col=2, type="log")
        fig.update_yaxes(title_text="PHI", row=2, col=1)
        fig.update_yaxes(title_text="ΔPhi Loss", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=2, type="log")
        
        # Configurar ejes X
        fig.update_xaxes(title_text="Época", row=2, col=1)
        fig.update_xaxes(title_text="Época", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_model_comparison(self, checkpoints):
        """Muestra comparación de modelos"""
        if not checkpoints:
            st.warning("No hay checkpoints disponibles para comparar")
            return
        
        # Crear DataFrame para comparación
        comparison_data = []
        for checkpoint in checkpoints[:10]:  # Solo primeros 10
            comparison_data.append({
                'Modelo': checkpoint['name'],
                'Época': checkpoint['epoch'],
                'Val PPL': checkpoint['val_ppl'] if checkpoint['val_ppl'] != float('inf') else None,
                'Val Loss': checkpoint['val_loss'] if checkpoint['val_loss'] != float('inf') else None,
                'Tamaño (MB)': round(checkpoint['size_mb'], 1),
                'Fecha': datetime.fromtimestamp(checkpoint['modified']).strftime('%Y-%m-%d %H:%M')
            })
        
        df_models = pd.DataFrame(comparison_data)
        
        # Filtrar modelos válidos para gráficos
        valid_models = df_models[df_models['Val PPL'].notna()]
        
        if len(valid_models) > 1:
            # Gráfico de comparación PPL
            fig_ppl = px.bar(
                valid_models.head(8), 
                x='Modelo', 
                y='Val PPL',
                title="Comparación de Perplexity por Modelo",
                color='Val PPL',
                color_continuous_scale='RdYlBu_r'
            )
            fig_ppl.update_xaxes(tickangle=45)
            st.plotly_chart(fig_ppl, use_container_width=True)
        
        # Tabla de comparación
        st.subheader("📊 Tabla de Comparación")
        st.dataframe(df_models, use_container_width=True)
    
    def show_real_time_stats(self):
        """Muestra estadísticas en tiempo real"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Cargar datos actuales
        histories = self.load_training_histories()
        checkpoints = self.load_model_checkpoints()
        
        with col1:
            st.metric(
                "🧠 Modelos Entrenados", 
                len(checkpoints),
                delta=f"Último: {datetime.fromtimestamp(checkpoints[0]['modified']).strftime('%H:%M')}" if checkpoints else "N/A"
            )
        
        with col2:
            best_ppl = min([c['val_ppl'] for c in checkpoints if c['val_ppl'] != float('inf')]) if checkpoints else None
            st.metric(
                "🎯 Mejor PPL", 
                f"{best_ppl:.2f}" if best_ppl else "N/A",
                delta="↓ Menor es mejor"
            )
        
        with col3:
            total_size = sum([c['size_mb'] for c in checkpoints])
            st.metric(
                "💾 Almacenamiento", 
                f"{total_size:.1f} MB",
                delta=f"{len(checkpoints)} archivos"
            )
        
        with col4:
            latest_training = histories[0]['modified'] if histories else 0
            hours_ago = (time.time() - latest_training) / 3600
            st.metric(
                "⏰ Último Entrenamiento", 
                f"{hours_ago:.1f}h ago",
                delta="Actividad reciente" if hours_ago < 1 else "Inactivo"
            )


def main():
    """Función principal del dashboard"""
    st.title("🧠 INFINITO V5.2 - Monitor de Entrenamientos")
    st.markdown("*Dashboard en tiempo real para monitorear y comparar modelos*")
    
    # Inicializar monitor
    monitor = InfinitoMonitor()
    
    # Detectar entrenamientos activos
    active_trainings = monitor.detect_active_training()
    
    if active_trainings:
        st.success(f"🔥 {len(active_trainings)} entrenamiento(s) activo(s) detectado(s)")
        
        # Mostrar información de entrenamientos activos
        for training in active_trainings:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("PID", training['pid'])
            with col2:
                st.metric("Runtime", f"{training['runtime_minutes']} min")
            with col3:
                st.metric("Memoria", f"{training['memory_mb']} MB")
            with col4:
                config = training['config']
                model_size = config.get('model_size', 'N/A')
                epochs = config.get('epochs', 'N/A')
                st.metric("Configuración", f"{model_size} ({epochs} épocas)")
            
            st.caption(f"📝 Comando: {training['command']}")
            # Intentar encontrar un archivo de historial relacionado para mostrar progreso en vivo
            try:
                model_size = training.get('config', {}).get('model_size')
                found = False
                search_dirs = [monitor.logs_dir, os.path.join('results', 'training'), monitor.results_dir]
                for d in search_dirs:
                    if not d:
                        continue
                    p = Path(d)
                    if not p.exists():
                        continue
                    # Buscar archivos recientes que contengan el model_size o pid
                    patterns = []
                    if model_size:
                        patterns.append(f"*{model_size}*.json")
                    patterns.extend([f"*{training['pid']}*.json", "training_history*.json", "*history*.json"]) 
                    candidates = []
                    for pattern in patterns:
                        candidates.extend(list(p.glob(pattern)))

                    # Elegir el archivo más reciente
                    if candidates:
                        latest = max(candidates, key=lambda x: x.stat().st_mtime)
                        # Cargar y mostrar progreso si el archivo es reciente
                        try:
                            with open(latest, 'r', encoding='utf-8') as fh:
                                data = json.load(fh)
                            progress_df = monitor.get_training_progress(data)
                            if progress_df is not None:
                                st.markdown(f"**Progreso detectado en:** {latest} (última modificación: {datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
                                monitor.plot_training_progress(progress_df, title=f"Progreso en vivo: {latest.stem}")
                                found = True
                                break
                        except Exception:
                            # No es crítico, continuar buscando
                            pass
                if not found:
                    st.info("No se encontró historial parcial reciente para este entrenamiento (los archivos se actualizan por época).")
            except Exception as e:
                st.warning(f"Error intentando mostrar progreso en vivo: {e}")
        
        st.markdown("---")
    else:
        st.info("ℹ️  No hay entrenamientos activos detectados")
    
    # Sidebar para controles
    st.sidebar.header("⚙️ Configuración")
    
    # Control de auto-refresh
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=True)
    if auto_refresh:
        refresh_rate = st.sidebar.slider("Intervalo (segundos)", 10, 300, 30)
    
    # Filtros
    st.sidebar.header("🔍 Filtros")
    show_all_models = st.sidebar.checkbox("Mostrar todos los modelos", value=False)
    max_models = st.sidebar.slider("Máximo modelos a mostrar", 5, 20, 10)
    
    # Métricas en tiempo real
    st.header("📊 Estado Actual")
    monitor.show_real_time_stats()
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Entrenamiento Activo", "🔍 Comparación", "📂 Historial", "⚙️ Configuración"])
    
    with tab1:
        st.header("📈 Progreso de Entrenamiento en Tiempo Real")
        
        # Cargar historiales
        histories = monitor.load_training_histories()
        
        if histories:
            # Selector de entrenamiento
            selected_training = st.selectbox(
                "Seleccionar entrenamiento:",
                options=range(len(histories)),
                format_func=lambda i: f"{histories[i]['name']} ({datetime.fromtimestamp(histories[i]['modified']).strftime('%Y-%m-%d %H:%M')})"
            )
            
            # Mostrar progreso del entrenamiento seleccionado
            try:
                training_data = histories[selected_training]['data']
                progress_df = monitor.get_training_progress(training_data)
                
                if progress_df is not None:
                    monitor.plot_training_progress(
                        progress_df, 
                        title=f"Progreso: {histories[selected_training]['name']}"
                    )
                    
                    # Información adicional
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ppl_cols = [col for col in progress_df.columns if 'val' in col and ('ppl' in col or 'perplexity' in col)]
                        if ppl_cols:
                            current_ppl = progress_df[ppl_cols[0]].iloc[-1]
                            initial_ppl = progress_df[ppl_cols[0]].iloc[0]
                            if initial_ppl > 0:
                                improvement = (initial_ppl - current_ppl) / initial_ppl * 100
                                st.metric("PPL Actual", f"{current_ppl:.2f}", f"{improvement:+.1f}%")
                            else:
                                st.metric("PPL Actual", f"{current_ppl:.2f}")
                    
                    with col2:
                        phi_cols = [col for col in progress_df.columns if 'phi' in col.lower()]
                        if phi_cols:
                            current_phi = progress_df[phi_cols[0]].iloc[-1]
                            st.metric("PHI Integration", f"{current_phi:.4f}")
                
                else:
                    st.warning("No se pudo extraer progreso del entrenamiento seleccionado")
            
            except Exception as e:
                st.error(f"Error procesando datos de entrenamiento: {str(e)}")
                st.info("Verifica el formato de los archivos de historial")
        else:
            st.info("No hay entrenamientos disponibles. Inicia un entrenamiento para ver datos en tiempo real.")
    
    with tab2:
        st.header("🔍 Comparación de Modelos")
        
        # Cargar checkpoints
        checkpoints = monitor.load_model_checkpoints()
        
        if not show_all_models:
            checkpoints = checkpoints[:max_models]
        
        monitor.show_model_comparison(checkpoints)
    
    with tab3:
        st.header("📂 Historial de Entrenamientos")
        
        histories = monitor.load_training_histories()
        
        if histories:
            for i, history in enumerate(histories):
                try:
                    with st.expander(f"📄 {history['name']} ({datetime.fromtimestamp(history['modified']).strftime('%Y-%m-%d %H:%M')})"):
                        
                        # Mostrar información básica
                        config = {}
                        if isinstance(history['data'], dict):
                            config = history['data'].get('config', {})
                        
                        if config:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Épocas:** {config.get('epochs', 'N/A')}")
                                st.write(f"**Batch Size:** {config.get('batch_size', 'N/A')}")
                            with col2:
                                st.write(f"**Learning Rate:** {config.get('lr', 'N/A')}")
                                st.write(f"**Hidden Dim:** {config.get('hidden_dim', 'N/A')}")
                            with col3:
                                st.write(f"**Modelo:** {config.get('model_size', 'N/A')}")
                                st.write(f"**Dropout:** {config.get('dropout', 'N/A')}")
                        
                        # Mostrar progreso si está disponible
                        try:
                            progress_df = monitor.get_training_progress(history['data'])
                            if progress_df is not None:
                                st.write("**Progreso del entrenamiento:**")
                                monitor.plot_training_progress(progress_df, title=f"Historial: {history['name']}")
                        except Exception as e:
                            st.warning(f"No se pudo cargar el progreso: {str(e)}")
                            
                except Exception as e:
                    st.error(f"Error procesando historial {history.get('name', 'desconocido')}: {str(e)}")
        else:
            st.info("No hay historiales de entrenamiento disponibles")
    
    with tab4:
        st.header("⚙️ Configuración del Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Rutas de Archivos")
            new_models_dir = st.text_input("Directorio de modelos:", value=monitor.models_dir)
            new_logs_dir = st.text_input("Directorio de logs:", value=monitor.logs_dir)
            
            if st.button("💾 Guardar Configuración"):
                monitor.models_dir = new_models_dir
                monitor.logs_dir = new_logs_dir
                st.success("Configuración guardada!")
        
        with col2:
            st.subheader("🔧 Opciones Avanzadas")
            
            if st.button("🔄 Limpiar Cache"):
                st.cache_data.clear()
                st.success("Cache limpiado!")
            
            if st.button("📊 Regenerar Estadísticas"):
                st.rerun()
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error en el dashboard: {str(e)}")
        st.info("Verifica que todas las dependencias estén instaladas: streamlit, plotly")