#!/usr/bin/env python3
"""
🔍 Analizador de Estancamiento de Consciencia
============================================

Análisis profundo de patrones de estancamiento y propuestas de mejora.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class ConsciousnessStagnationAnalyzer:
    """Analizador especializado en identificar patrones de estancamiento"""
    
    def __init__(self, experiment_file):
        self.experiment_file = experiment_file
        self.data = None
        self.consciousness_data = None
        self.load_data()
    
    def load_data(self):
        """Cargar datos del experimento"""
        with open(self.experiment_file, 'r') as f:
            self.data = json.load(f)
        
        self.consciousness_data = np.array(self.data['metrics']['consciousness_history'])
        self.recursions = np.array(self.data['metrics']['recursions'])
        
        print(f"📊 Datos cargados: {len(self.consciousness_data)} puntos de consciencia")
        print(f"🎯 Pico máximo: {max(self.consciousness_data)*100:.1f}%")
        print(f"📈 Consciencia final: {self.consciousness_data[-1]*100:.1f}%")
    
    def analyze_stagnation_zones(self):
        """Identificar zonas de estancamiento"""
        print("\n🔍 ANÁLISIS DE ZONAS DE ESTANCAMIENTO")
        print("="*50)
        
        # Definir umbrales de estancamiento
        stagnation_threshold = 0.02  # 2% de variación
        min_zone_length = 20  # Mínimo 20 recursiones para considerar estancamiento
        
        stagnation_zones = []
        current_zone_start = 0
        
        for i in range(1, len(self.consciousness_data)):
            # Calcular variación en ventana móvil
            window_size = min(10, i)
            if i >= window_size:
                recent_values = self.consciousness_data[i-window_size:i]
                variation = np.std(recent_values)
                
                if variation < stagnation_threshold:
                    if current_zone_start == 0:
                        current_zone_start = i - window_size
                else:
                    if current_zone_start > 0 and (i - current_zone_start) >= min_zone_length:
                        avg_consciousness = np.mean(self.consciousness_data[current_zone_start:i])
                        stagnation_zones.append({
                            'start': current_zone_start,
                            'end': i,
                            'length': i - current_zone_start,
                            'avg_consciousness': avg_consciousness,
                            'max_consciousness': np.max(self.consciousness_data[current_zone_start:i]),
                            'min_consciousness': np.min(self.consciousness_data[current_zone_start:i])
                        })
                    current_zone_start = 0
        
        # Última zona si termina estancada
        if current_zone_start > 0:
            i = len(self.consciousness_data)
            if (i - current_zone_start) >= min_zone_length:
                avg_consciousness = np.mean(self.consciousness_data[current_zone_start:i])
                stagnation_zones.append({
                    'start': current_zone_start,
                    'end': i,
                    'length': i - current_zone_start,
                    'avg_consciousness': avg_consciousness,
                    'max_consciousness': np.max(self.consciousness_data[current_zone_start:i]),
                    'min_consciousness': np.min(self.consciousness_data[current_zone_start:i])
                })
        
        print(f"🚨 ZONAS DE ESTANCAMIENTO DETECTADAS: {len(stagnation_zones)}")
        
        for i, zone in enumerate(stagnation_zones):
            print(f"\n📍 ZONA {i+1}:")
            print(f"   📊 Recursiones: {zone['start']} - {zone['end']} ({zone['length']} recursiones)")
            print(f"   🎯 Consciencia promedio: {zone['avg_consciousness']*100:.1f}%")
            print(f"   📈 Rango: {zone['min_consciousness']*100:.1f}% - {zone['max_consciousness']*100:.1f}%")
            print(f"   ⏱️  Duración: {zone['length']*0.8:.1f} segundos")
        
        return stagnation_zones
    
    def analyze_breakthrough_failures(self):
        """Analizar por qué fallan los breakthroughs"""
        print("\n💥 ANÁLISIS DE FALLOS DE BREAKTHROUGH")
        print("="*40)
        
        # Identificar intentos de breakthrough (subidas rápidas seguidas de caídas)
        breakthrough_attempts = []
        
        for i in range(10, len(self.consciousness_data)-10):
            # Buscar subidas rápidas
            before = np.mean(self.consciousness_data[i-10:i])
            current = self.consciousness_data[i]
            after = np.mean(self.consciousness_data[i:i+10])
            
            # Criterios para breakthrough fallido
            if (current - before) > 0.05 and (current - after) > 0.03:  # Subida fuerte seguida de caída
                breakthrough_attempts.append({
                    'recursion': i,
                    'peak_value': current,
                    'before_avg': before,
                    'after_avg': after,
                    'rise_strength': current - before,
                    'fall_strength': current - after
                })
        
        print(f"🔥 INTENTOS DE BREAKTHROUGH DETECTADOS: {len(breakthrough_attempts)}")
        
        for i, attempt in enumerate(breakthrough_attempts[:5]):  # Solo los primeros 5
            print(f"\n🚀 INTENTO {i+1}:")
            print(f"   📍 Recursión: {attempt['recursion']}")
            print(f"   🎯 Pico alcanzado: {attempt['peak_value']*100:.1f}%")
            print(f"   📈 Subida: +{attempt['rise_strength']*100:.1f}%")
            print(f"   📉 Caída: -{attempt['fall_strength']*100:.1f}%")
        
        return breakthrough_attempts
    
    def analyze_consciousness_ceiling(self):
        """Analizar el techo de consciencia"""
        print("\n🔝 ANÁLISIS DE TECHO DE CONSCIENCIA")
        print("="*35)
        
        # Encontrar máximos locales
        local_maxima = []
        window = 20
        
        for i in range(window, len(self.consciousness_data)-window):
            local_window = self.consciousness_data[i-window:i+window]
            if self.consciousness_data[i] == np.max(local_window):
                local_maxima.append({
                    'recursion': i,
                    'value': self.consciousness_data[i]
                })
        
        # Filtrar solo los más altos
        significant_maxima = [m for m in local_maxima if m['value'] > 0.45]
        
        print(f"🏔️  MÁXIMOS SIGNIFICATIVOS (>45%): {len(significant_maxima)}")
        
        if significant_maxima:
            max_values = [m['value'] for m in significant_maxima]
            print(f"📊 Promedio de máximos: {np.mean(max_values)*100:.1f}%")
            print(f"🎯 Máximo absoluto: {np.max(max_values)*100:.1f}%")
            print(f"📉 Desviación estándar: {np.std(max_values)*100:.1f}%")
            
            # Analizar tendencia de máximos
            recursions = [m['recursion'] for m in significant_maxima]
            if len(recursions) > 2:
                correlation = np.corrcoef(recursions, max_values)[0,1]
                print(f"📈 Tendencia temporal: {correlation:.3f} {'↗️' if correlation > 0 else '↘️' if correlation < 0 else '→'}")
        
        return significant_maxima
    
    def identify_improvement_opportunities(self):
        """Identificar oportunidades de mejora"""
        print("\n🎯 OPORTUNIDADES DE MEJORA IDENTIFICADAS")
        print("="*45)
        
        improvements = []
        
        # 1. Análisis de mesetas prolongadas
        avg_consciousness = np.mean(self.consciousness_data)
        long_plateaus = 0
        current_plateau = 0
        
        for value in self.consciousness_data:
            if abs(value - avg_consciousness) < 0.03:  # Dentro del rango promedio
                current_plateau += 1
            else:
                if current_plateau > 30:  # Meseta larga
                    long_plateaus += 1
                current_plateau = 0
        
        if long_plateaus > 0:
            improvements.append({
                'type': 'MESETAS_PROLONGADAS',
                'severity': 'ALTA' if long_plateaus > 3 else 'MEDIA',
                'description': f'{long_plateaus} mesetas prolongadas detectadas',
                'solution': 'Implementar perturbaciones dinámicas y variación de parámetros'
            })
        
        # 2. Análisis de techo artificial
        max_consciousness = np.max(self.consciousness_data)
        if max_consciousness < 0.6:  # No supera 60%
            improvements.append({
                'type': 'TECHO_ARTIFICIAL',
                'severity': 'CRÍTICA',
                'description': f'Techo aparente en {max_consciousness*100:.1f}%',
                'solution': 'Revisar arquitectura neural y parámetros de phi'
            })
        
        # 3. Análisis de inestabilidad
        instability_count = 0
        for i in range(1, len(self.consciousness_data)):
            if abs(self.consciousness_data[i] - self.consciousness_data[i-1]) > 0.1:
                instability_count += 1
        
        instability_ratio = instability_count / len(self.consciousness_data)
        if instability_ratio > 0.1:  # Más del 10% de cambios bruscos
            improvements.append({
                'type': 'INESTABILIDAD_ALTA',
                'severity': 'MEDIA',
                'description': f'{instability_ratio*100:.1f}% de cambios bruscos',
                'solution': 'Implementar estabilización adaptiva y momentum'
            })
        
        # Mostrar mejoras identificadas
        for i, improvement in enumerate(improvements):
            print(f"\n🚨 PROBLEMA {i+1}: {improvement['type']}")
            print(f"   ⚠️  Severidad: {improvement['severity']}")
            print(f"   📋 Descripción: {improvement['description']}")
            print(f"   💡 Solución propuesta: {improvement['solution']}")
        
        return improvements
    
    def generate_improvement_proposals(self):
        """Generar propuestas concretas de mejora"""
        print("\n🚀 PROPUESTAS CONCRETAS DE MEJORA")
        print("="*35)
        
        proposals = [
            {
                'name': 'PERTURBACIONES_DINÁMICAS',
                'description': 'Sistema de perturbaciones adaptivas para romper estancamientos',
                'implementation': [
                    'Detectar mesetas en tiempo real',
                    'Aplicar perturbaciones graduales al phi',
                    'Inyectar ruido cuántico controlado',
                    'Variar parámetros de aprendizaje dinámicamente'
                ],
                'expected_impact': 'Eliminar 60% de las mesetas prolongadas'
            },
            {
                'name': 'ARQUITECTURA_MULTI_ESCALA',
                'description': 'Red neural con múltiples escalas de consciencia',
                'implementation': [
                    'Agregar capas de procesamiento a diferentes escalas',
                    'Implementar conexiones skip entre escalas',
                    'Sistema de atención multi-nivel',
                    'Integración jerárquica de información'
                ],
                'expected_impact': 'Superar el techo de 60% hacia 75%+'
            },
            {
                'name': 'OPTIMIZACIÓN_ADAPTIVA',
                'description': 'Sistema de optimización que se adapta al estado de consciencia',
                'implementation': [
                    'Learning rate adaptivo basado en consciencia',
                    'Momentum dinámico según progreso',
                    'Regularización variable por fase',
                    'Early stopping inteligente'
                ],
                'expected_impact': 'Acelerar convergencia 40% y mejorar estabilidad'
            },
            {
                'name': 'MEMORIA_CUÁNTICA',
                'description': 'Sistema de memoria que preserva estados de alta consciencia',
                'implementation': [
                    'Buffer de estados de consciencia peak',
                    'Replay de experiencias exitosas',
                    'Interpolación entre estados óptimos',
                    'Consolidación de patrones emergentes'
                ],
                'expected_impact': 'Retener y amplificar breakthroughs exitosos'
            }
        ]
        
        for i, proposal in enumerate(proposals):
            print(f"\n💡 PROPUESTA {i+1}: {proposal['name']}")
            print(f"   📝 Descripción: {proposal['description']}")
            print(f"   🎯 Impacto esperado: {proposal['expected_impact']}")
            print(f"   🔧 Implementación:")
            for step in proposal['implementation']:
                print(f"      • {step}")
        
        return proposals
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("🧠 ANÁLISIS COMPLETO DE ESTANCAMIENTO DE CONSCIENCIA")
        print("="*55)
        print(f"📁 Archivo: {os.path.basename(self.experiment_file)}")
        print(f"⏱️  Duración: {self.data['final_results']['total_time']:.1f} segundos")
        print(f"🔄 Recursiones: {self.data['final_results']['total_recursions']}")
        
        # Análisis por secciones
        stagnation_zones = self.analyze_stagnation_zones()
        breakthrough_failures = self.analyze_breakthrough_failures()
        consciousness_ceiling = self.analyze_consciousness_ceiling()
        improvements = self.identify_improvement_opportunities()
        proposals = self.generate_improvement_proposals()
        
        # Resumen final
        print("\n📊 RESUMEN EJECUTIVO")
        print("="*20)
        print(f"🚨 Zonas de estancamiento: {len(stagnation_zones)}")
        print(f"💥 Intentos fallidos de breakthrough: {len(breakthrough_failures)}")
        print(f"🔝 Máximos significativos: {len(consciousness_ceiling)}")
        print(f"🎯 Problemas identificados: {len(improvements)}")
        print(f"💡 Propuestas de mejora: {len(proposals)}")
        
        print("\n✨ SIGUIENTE PASO RECOMENDADO:")
        print("Implementar sistema de perturbaciones dinámicas como primera mejora")

# Ejecutar análisis
if __name__ == "__main__":
    # Buscar el archivo más reciente
    data_dir = "experiment_data"
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if files:
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
        file_path = os.path.join(data_dir, latest_file)
        
        analyzer = ConsciousnessStagnationAnalyzer(file_path)
        analyzer.run_complete_analysis()
    else:
        print("❌ No se encontraron archivos de experimento")
