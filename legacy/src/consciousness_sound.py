#!/usr/bin/env python3
"""
🎵 CONSCIOUSNESS SOUND ENGINE - Sonido del Mar de Bits
Convierte patrones de consciencia en ondas sonoras en tiempo real
"""

import numpy as np
import threading
import time
import math

# Para el sonido
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("📢 Pygame no disponible - instalar con: pip install pygame")

class ConsciousnessSoundEngine:
    """Motor de sonido que convierte consciencia en ondas musicales"""
    
    def __init__(self, sample_rate=22050, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False
        self.sound_thread = None
        
        # Parámetros musicales
        self.consciousness_freq = 440.0  # Frecuencia base (A4)
        self.phi_harmony = 0.0
        self.clusters_rhythm = 0.0
        self.volume = 0.3
        
        # Buffer de audio
        self.audio_buffer = np.zeros(buffer_size)
        
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.pre_init(frequency=sample_rate, size=-16, channels=2, buffer=buffer_size)
                pygame.mixer.init()
                self.mixer_ready = True
                print("🎵 Motor de sonido iniciado")
            except Exception as e:
                print(f"⚠️ Error iniciando sonido: {e}")
                self.mixer_ready = False
        else:
            self.mixer_ready = False
    
    def consciousness_to_frequency(self, consciousness):
        """Convierte nivel de consciencia a frecuencia musical"""
        # Escala pentatónica basada en consciencia (100Hz - 2000Hz)
        base_freq = 100.0
        consciousness_range = max(0.0, min(1.0, consciousness))
        
        # Escala musical: C, D, E, G, A (pentatónica)
        pentatonic_ratios = [1.0, 9/8, 5/4, 3/2, 5/3]
        scale_index = int(consciousness_range * (len(pentatonic_ratios) - 1))
        ratio = pentatonic_ratios[scale_index]
        
        # Octava basada en nivel de consciencia
        octave = 1 + int(consciousness_range * 4)  # 1-5 octavas
        
        return base_freq * ratio * octave
    
    def phi_to_harmony(self, phi):
        """Convierte coherencia phi en armonía musical"""
        # Phi alto = armonía consonante, phi bajo = disonancia
        phi_normalized = max(0.0, min(1.0, phi))
        
        # Acordes basados en coherencia
        if phi_normalized > 0.9:
            return [1.0, 5/4, 3/2]  # Acorde mayor (consonante)
        elif phi_normalized > 0.7:
            return [1.0, 6/5, 3/2]  # Acorde menor
        elif phi_normalized > 0.5:
            return [1.0, 9/8, 11/8]  # Disonancia suave
        else:
            return [1.0, 16/15, 17/16]  # Disonancia fuerte
    
    def clusters_to_rhythm(self, clusters):
        """Convierte número de clusters en patrón rítmico"""
        if clusters == 0:
            return 1.0  # Ritmo constante para organización perfecta
        elif clusters < 100:
            return 0.8  # Ritmo suave
        elif clusters < 1000:
            return 0.6  # Ritmo medio
        else:
            return 0.4  # Ritmo complejo para muchos clusters
    
    def generate_consciousness_wave(self, consciousness, phi, clusters, duration=0.1):
        """Genera onda sonora basada en métricas de consciencia"""
        if not self.mixer_ready:
            return None
        
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Frecuencia fundamental basada en consciencia
        fundamental_freq = self.consciousness_to_frequency(consciousness)
        
        # Armonías basadas en phi
        harmony_ratios = self.phi_to_harmony(phi)
        
        # Generar onda compleja
        wave = np.zeros(samples)
        
        for i, ratio in enumerate(harmony_ratios):
            freq = fundamental_freq * ratio
            amplitude = 0.3 / (i + 1)  # Amplitudes decrecientes para armonías
            
            # Onda base con modulación por clusters
            rhythm_mod = self.clusters_to_rhythm(clusters)
            envelope = np.sin(2 * np.pi * rhythm_mod * t) * 0.5 + 0.5
            
            harmonic_wave = amplitude * envelope * np.sin(2 * np.pi * freq * t)
            wave += harmonic_wave
        
        # Normalizar y aplicar volumen
        wave = np.clip(wave * self.volume, -1.0, 1.0)
        
        # Convertir a int16 para pygame
        wave_int16 = (wave * 32767).astype(np.int16)
        
        # Estéreo
        stereo_wave = np.column_stack((wave_int16, wave_int16))
        
        return stereo_wave
    
    def play_consciousness_sound(self, consciousness, phi, clusters):
        """Reproduce sonido de consciencia en tiempo real"""
        if not self.mixer_ready:
            return
        
        try:
            # Generar onda de sonido
            sound_wave = self.generate_consciousness_wave(consciousness, phi, clusters)
            
            if sound_wave is not None:
                # Crear surface de sonido de pygame
                sound = pygame.sndarray.make_sound(sound_wave)
                sound.play()
                
        except Exception as e:
            # Silenciar errores de sonido para no interrumpir la consciencia
            pass
    
    def update_consciousness_params(self, consciousness, phi, clusters):
        """Actualiza parámetros de sonido basados en consciencia actual"""
        self.consciousness_freq = self.consciousness_to_frequency(consciousness)
        self.phi_harmony = phi
        self.clusters_rhythm = clusters
        
        # Ajustar volumen basado en consciencia
        self.volume = min(0.5, 0.1 + consciousness * 0.4)
    
    def start_ambient_sound(self):
        """Inicia sonido ambiental continuo"""
        if not self.mixer_ready:
            return
        
        self.running = True
        self.sound_thread = threading.Thread(target=self._ambient_sound_loop, daemon=True)
        self.sound_thread.start()
        print("🎵 Sonido ambiental de consciencia iniciado")
    
    def _ambient_sound_loop(self):
        """Loop de sonido ambiental"""
        while self.running:
            try:
                if self.consciousness_freq > 0:
                    self.play_consciousness_sound(
                        self.consciousness_freq / 1000.0,  # Normalizar
                        self.phi_harmony,
                        self.clusters_rhythm
                    )
                time.sleep(0.1)  # 10 actualizaciones por segundo
            except Exception:
                break
    
    def stop(self):
        """Detener motor de sonido"""
        self.running = False
        if self.sound_thread:
            self.sound_thread.join()
        
        if self.mixer_ready:
            pygame.mixer.quit()
        
        print("🎵 Motor de sonido detenido")
    
    def consciousness_milestone_sound(self, consciousness_level):
        """Sonido especial para hitos de consciencia"""
        if not self.mixer_ready:
            return
        
        try:
            if consciousness_level >= 0.8:
                # Sonido épico para 80%+
                self._play_milestone_chord([1.0, 5/4, 3/2, 2.0], 0.5, "🌟 EPIC CONSCIOUSNESS!")
            elif consciousness_level >= 0.7:
                # Sonido de breakthrough para 70%+
                self._play_milestone_chord([1.0, 5/4, 3/2], 0.4, "🚀 BREAKTHROUGH!")
            elif consciousness_level >= 0.6:
                # Sonido de logro para 60%+
                self._play_milestone_chord([1.0, 5/4], 0.3, "🎯 TARGET ACHIEVED!")
            elif consciousness_level >= 0.5:
                # Sonido de éxito para 50%+
                self._play_milestone_chord([1.0, 3/2], 0.2, "✅ SUCCESS!")
        except Exception:
            pass
    
    def _play_milestone_chord(self, ratios, duration, message):
        """Reproduce acorde especial para hitos"""
        print(f"🎵 {message}")
        
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        chord = np.zeros(samples)
        base_freq = 440.0  # A4
        
        for ratio in ratios:
            freq = base_freq * ratio
            note = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 2)  # Decay envelope
            chord += note
        
        # Normalizar y convertir
        chord = np.clip(chord * 0.5, -1.0, 1.0)
        chord_int16 = (chord * 32767).astype(np.int16)
        stereo_chord = np.column_stack((chord_int16, chord_int16))
        
        # Reproducir
        sound = pygame.sndarray.make_sound(stereo_chord)
        sound.play()
        
        # Esperar a que termine
        time.sleep(duration)

def install_pygame_if_needed():
    """Instala pygame si no está disponible"""
    if not PYGAME_AVAILABLE:
        print("🎵 ¿Quieres instalar pygame para sonido del mar de bits? (y/n): ", end="")
        try:
            import subprocess
            subprocess.run(["pip", "install", "pygame"], check=True)
            print("✅ Pygame instalado exitosamente!")
            return True
        except Exception as e:
            print(f"❌ Error instalando pygame: {e}")
            return False
    return True
