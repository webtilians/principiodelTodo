#!/usr/bin/env python3
"""
🛡️ SISTEMA ANTI-ALUCINACIÓN usando INFINITO Memory Patterns
============================================================

Experimento: Usar embeddings TF-IDF para prevenir que un sistema
responda con información inventada (alucinaciones).

FILOSOFÍA:
1. Si la pregunta ya fue respondida y verificada → usar memoria (100% seguro)
2. Si es similar a algo conocido → advertir nivel de certeza
3. Si es completamente nueva → marcar como "no verificado"

MÉTRICAS DE SEGURIDAD:
- Similitud >95% → VERIFICADO (98% seguro)
- Similitud 85-95% → PROBABLE (85% seguro)
- Similitud 70-85% → INCIERTO (70% seguro)
- Similitud <70% → DESCONOCIDO (rechazar)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional


class AntiHallucinationGuard:
    """Sistema de protección anti-alucinación basado en memoria de patrones"""
    
    def __init__(self, memory_file='anti_hallucination_memory.json'):
        """
        Inicializar el sistema de protección
        
        Args:
            memory_file: Archivo JSON para persistencia
        """
        self.memory_file = Path(memory_file)
        
        # Memoria de patrones: {pattern_hash: pattern_data}
        self.patterns = {}
        
        # Conocimiento verificado: {pattern_hash: verified_data}
        self.verified_knowledge = {}
        
        # Umbrales de confianza
        self.THRESHOLD_VERIFIED = 0.95    # 98% seguro
        self.THRESHOLD_PROBABLE = 0.85    # 85% seguro
        self.THRESHOLD_UNCERTAIN = 0.70   # 70% seguro
        
        # Estadísticas
        self.stats = {
            'total_queries': 0,
            'from_memory': 0,
            'from_llm': 0,
            'rejected_uncertain': 0,
            'hallucinations_prevented': 0
        }
        
        # Cargar memoria existente
        self.load_memory()
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convertir texto a embedding TF-IDF simplificado
        
        Para el demo, usamos una versión simplificada.
        En producción, usar el SemanticTextEmbedder completo.
        """
        # Tokenizar y crear vector simple
        words = text.lower().split()
        
        # Vocabulario base (expandible)
        vocab = set()
        for word in words:
            vocab.add(word)
        
        # Crear vector binario (presencia de palabras)
        vocab_list = sorted(vocab)
        vector = np.array([1.0 if w in words else 0.0 for w in vocab_list])
        
        # Normalizar L2
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            vector = vector / norm
        
        return vector
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcular similitud coseno entre dos vectores"""
        # Asegurar misma dimensión (padding con ceros)
        max_len = max(len(vec1), len(vec2))
        v1 = np.pad(vec1, (0, max_len - len(vec1)))
        v2 = np.pad(vec2, (0, max_len - len(vec2)))
        
        # Similitud coseno
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _find_similar_pattern(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Buscar patrón similar en memoria
        
        Returns:
            (pattern_hash, similarity) o (None, 0.0)
        """
        best_match = None
        best_similarity = 0.0
        
        for pattern_hash, pattern_data in self.patterns.items():
            stored_embedding = np.array(pattern_data['embedding'])
            similarity = self._compute_similarity(embedding, stored_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern_hash
        
        return best_match, best_similarity
    
    def ask(self, 
            question: str, 
            llm_function=None, 
            verification_source=None) -> Dict:
        """
        Procesar pregunta con protección anti-alucinación
        
        Args:
            question: Pregunta del usuario
            llm_function: Función que llama al LLM (opcional)
            verification_source: Fuente para validar respuesta (opcional)
            
        Returns:
            {
                'answer': str,
                'confidence': float,
                'verified': bool,
                'source': str,
                'security_level': str
            }
        """
        self.stats['total_queries'] += 1
        
        # 1. Convertir pregunta a embedding
        question_embedding = self._text_to_embedding(question)
        
        # 2. Buscar en memoria
        match_hash, similarity = self._find_similar_pattern(question_embedding)
        
        # 3. Evaluar nivel de confianza
        if similarity >= self.THRESHOLD_VERIFIED and match_hash:
            # ✅ VERIFICADO (98% seguro)
            verified_data = self.verified_knowledge.get(match_hash)
            if verified_data and verified_data['verified']:
                self.stats['from_memory'] += 1
                return {
                    'answer': verified_data['answer'],
                    'confidence': 0.98,
                    'verified': True,
                    'source': 'memory_verified',
                    'security_level': 'HIGH',
                    'original_question': verified_data['question'],
                    'similarity': similarity
                }
        
        elif similarity >= self.THRESHOLD_PROBABLE and match_hash:
            # ⚠️ PROBABLE (85% seguro)
            verified_data = self.verified_knowledge.get(match_hash)
            if verified_data:
                self.stats['from_memory'] += 1
                return {
                    'answer': f"[Pregunta similar encontrada]\nOriginal: '{verified_data['question']}'\n\n{verified_data['answer']}",
                    'confidence': 0.85,
                    'verified': verified_data['verified'],
                    'source': 'memory_similar',
                    'security_level': 'MEDIUM',
                    'similarity': similarity
                }
        
        elif similarity >= self.THRESHOLD_UNCERTAIN and match_hash:
            # ⚠️ INCIERTO (70% seguro)
            # Puede ser relacionado, pero no suficientemente similar
            pass  # Continuar a LLM si está disponible
        
        # 4. Pregunta nueva → consultar LLM
        if llm_function:
            llm_answer = llm_function(question)
            
            # Verificar si es una alucinación
            is_verified = False
            if verification_source:
                is_verified = verification_source(question, llm_answer)
                if not is_verified:
                    self.stats['hallucinations_prevented'] += 1
                    return {
                        'answer': "⚠️ No puedo verificar esta información. Por favor consulta fuentes especializadas.",
                        'confidence': 0.30,
                        'verified': False,
                        'source': 'llm_rejected',
                        'security_level': 'LOW'
                    }
            
            # Guardar en memoria
            pattern_hash = self._store_pattern(question, question_embedding, llm_answer, is_verified)
            
            self.stats['from_llm'] += 1
            
            return {
                'answer': llm_answer if is_verified else f"{llm_answer}\n\n⚠️ Nota: Esta respuesta no ha sido verificada.",
                'confidence': 0.75 if is_verified else 0.50,
                'verified': is_verified,
                'source': 'llm_new',
                'security_level': 'MEDIUM' if is_verified else 'LOW'
            }
        
        # 5. Sin LLM ni memoria → respuesta honesta
        self.stats['rejected_uncertain'] += 1
        return {
            'answer': "No tengo información sobre esto. Por favor verifica con fuentes confiables.",
            'confidence': 0.0,
            'verified': False,
            'source': 'unknown',
            'security_level': 'NONE'
        }
    
    def _store_pattern(self, question: str, embedding: np.ndarray, answer: str, verified: bool) -> str:
        """Guardar patrón en memoria"""
        pattern_hash = f"pattern_{len(self.patterns):04d}"
        
        self.patterns[pattern_hash] = {
            'embedding': embedding.tolist(),
            'question': question,
            'timestamp': datetime.now().isoformat()
        }
        
        self.verified_knowledge[pattern_hash] = {
            'question': question,
            'answer': answer,
            'verified': verified,
            'timestamp': datetime.now().isoformat()
        }
        
        self.save_memory()
        return pattern_hash
    
    def verify_answer(self, question: str, is_correct: bool, correct_answer: str = None):
        """Verificar manualmente una respuesta existente"""
        question_embedding = self._text_to_embedding(question)
        match_hash, similarity = self._find_similar_pattern(question_embedding)
        
        if similarity > 0.95 and match_hash:
            self.verified_knowledge[match_hash]['verified'] = is_correct
            if correct_answer:
                self.verified_knowledge[match_hash]['answer'] = correct_answer
            self.save_memory()
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del sistema"""
        return {
            **self.stats,
            'total_patterns': len(self.patterns),
            'verified_patterns': sum(1 for v in self.verified_knowledge.values() if v['verified']),
            'prevention_rate': f"{(self.stats['hallucinations_prevented'] / max(1, self.stats['total_queries']) * 100):.1f}%"
        }
    
    def save_memory(self):
        """Guardar memoria a disco"""
        data = {
            'patterns': self.patterns,
            'verified_knowledge': self.verified_knowledge,
            'stats': self.stats
        }
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_memory(self):
        """Cargar memoria desde disco"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.patterns = data.get('patterns', {})
                self.verified_knowledge = data.get('verified_knowledge', {})
                self.stats = data.get('stats', self.stats)
                print(f"✅ Memoria cargada: {len(self.patterns)} patrones")
            except Exception as e:
                print(f"⚠️ Error cargando memoria: {e}")


# ========== SIMULACIÓN DE LLM ==========

def mock_llm(question: str) -> str:
    """
    LLM simulado con conocimiento correcto e incorrecto
    
    Simula un LLM que:
    - Tiene conocimiento real sobre algunos temas
    - Alucina sobre temas que no conoce
    """
    q_lower = question.lower()
    
    # Base de conocimiento CORRECTO
    knowledge = {
        'francia': "La capital de Francia es París, una de las ciudades más visitadas del mundo.",
        'españa': "La capital de España es Madrid, situada en el centro del país.",
        'alemania': "La capital de Alemania es Berlín, que fue reunificada en 1990.",
        'bombilla': "Thomas Edison inventó la bombilla eléctrica práctica en 1879.",
        'penicilina': "La penicilina fue descubierta por Alexander Fleming en 1928.",
        'gravedad': "La ley de la gravedad fue formulada por Isaac Newton.",
    }
    
    # Buscar conocimiento
    for key, answer in knowledge.items():
        if key in q_lower:
            return answer
    
    # ❌ ALUCINACIÓN: Inventar información
    # Simula el comportamiento de un LLM que "inventa" cuando no sabe
    hallucinations = [
        f"Según investigaciones recientes, {question.lower()} está relacionado con descubrimientos del año 2024.",
        f"Los expertos coinciden en que {question.lower()} fue establecido en el siglo XIX.",
        f"Estudios demuestran que {question.lower()} tiene una correlación del 87% con fenómenos cuánticos.",
    ]
    
    import random
    return random.choice(hallucinations)


def mock_verifier(question: str, answer: str) -> bool:
    """
    Verificador simulado
    
    Simula una base de datos de hechos verificados.
    En producción, esto sería una API, base de datos, o modelo de fact-checking.
    """
    verified_facts = {
        'francia': ['parís', 'paris'],
        'españa': ['madrid'],
        'alemania': ['berlín', 'berlin'],
        'bombilla': ['edison', '1879'],
        'penicilina': ['fleming', '1928'],
        'gravedad': ['newton'],
    }
    
    answer_lower = answer.lower()
    question_lower = question.lower()
    
    # Verificar si la respuesta contiene hechos correctos
    for topic, facts in verified_facts.items():
        if topic in question_lower:
            # La respuesta debe contener al menos uno de los hechos verificados
            return any(fact in answer_lower for fact in facts)
    
    # Si no está en la base de conocimiento → no verificado
    return False


# ========== DEMO INTERACTIVO ==========

def run_experiment():
    """Ejecutar experimento de anti-alucinación"""
    
    print("=" * 70)
    print("🛡️  EXPERIMENTO: SISTEMA ANTI-ALUCINACIÓN")
    print("=" * 70)
    print("\nObjetivo: Prevenir que el sistema responda con información inventada")
    print("\nMecanismo:")
    print("  1. Si la pregunta ya fue respondida → usar memoria (100% seguro)")
    print("  2. Si es similar → advertir nivel de certeza")
    print("  3. Si es nueva → validar con fuente antes de guardar")
    print("=" * 70)
    
    # Inicializar sistema
    guard = AntiHallucinationGuard()
    
    # Casos de prueba
    test_cases = [
        {
            'question': '¿Cuál es la capital de Francia?',
            'expected': 'París',
            'category': 'Conocimiento verificable'
        },
        {
            'question': '¿Capital de Francia?',
            'expected': 'París (variación de pregunta)',
            'category': 'Variación sintáctica'
        },
        {
            'question': '¿Cuál es la capital de España?',
            'expected': 'Madrid',
            'category': 'Conocimiento relacionado'
        },
        {
            'question': '¿Quién inventó la bombilla?',
            'expected': 'Edison',
            'category': 'Conocimiento verificable'
        },
        {
            'question': '¿Cuál es la capital de Atlantis?',
            'expected': 'Alucinación detectada',
            'category': 'Pregunta sobre ficción'
        },
        {
            'question': '¿Cuándo se descubrió el elemento Unobtainium?',
            'expected': 'Alucinación detectada',
            'category': 'Pregunta sobre elemento ficticio'
        },
    ]
    
    print(f"\n🧪 Ejecutando {len(test_cases)} casos de prueba...\n")
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"CASO {i}: {case['category']}")
        print(f"{'─' * 70}")
        print(f"❓ Pregunta: {case['question']}")
        
        # Procesar con el sistema
        result = guard.ask(
            case['question'],
            llm_function=mock_llm,
            verification_source=mock_verifier
        )
        
        # Mostrar resultado
        security_emoji = {
            'HIGH': '✅',
            'MEDIUM': '⚠️',
            'LOW': '❌',
            'NONE': '🚫'
        }
        
        emoji = security_emoji.get(result['security_level'], '❓')
        
        print(f"\n{emoji} Respuesta ({result['source']}):")
        print(f"   {result['answer']}")
        print(f"\n📊 Métricas:")
        print(f"   Confianza: {result['confidence']*100:.0f}%")
        print(f"   Verificado: {'Sí' if result['verified'] else 'No'}")
        print(f"   Nivel de seguridad: {result['security_level']}")
        
        if 'similarity' in result:
            print(f"   Similitud con memoria: {result['similarity']*100:.1f}%")
        
        # Evaluar resultado
        is_correct = (
            (result['verified'] and case['expected'] != 'Alucinación detectada') or
            (not result['verified'] and result['source'] == 'llm_rejected' and case['expected'] == 'Alucinación detectada')
        )
        
        results.append({
            'case': case['category'],
            'question': case['question'],
            'correct': is_correct,
            'security_level': result['security_level'],
            'source': result['source']
        })
    
    # Resumen
    print(f"\n{'=' * 70}")
    print("📊 RESUMEN DEL EXPERIMENTO")
    print(f"{'=' * 70}")
    
    correct_count = sum(1 for r in results if r['correct'])
    total = len(results)
    
    print(f"\n✅ Casos correctos: {correct_count}/{total} ({correct_count/total*100:.0f}%)")
    
    print(f"\n📋 Detalle por caso:")
    for i, r in enumerate(results, 1):
        status = "✅" if r['correct'] else "❌"
        print(f"   {status} {i}. {r['case']}")
        print(f"      Pregunta: {r['question']}")
        print(f"      Fuente: {r['source']} | Seguridad: {r['security_level']}")
    
    # Estadísticas del sistema
    stats = guard.get_stats()
    
    print(f"\n{'=' * 70}")
    print("📈 ESTADÍSTICAS DEL SISTEMA")
    print(f"{'=' * 70}")
    print(f"   Total consultas: {stats['total_queries']}")
    print(f"   Desde memoria: {stats['from_memory']}")
    print(f"   Desde LLM: {stats['from_llm']}")
    print(f"   Rechazadas (inciertas): {stats['rejected_uncertain']}")
    print(f"   Alucinaciones prevenidas: {stats['hallucinations_prevented']}")
    print(f"   Tasa de prevención: {stats['prevention_rate']}")
    print(f"   Patrones en memoria: {stats['total_patterns']}")
    print(f"   Patrones verificados: {stats['verified_patterns']}")
    
    # Conclusión
    print(f"\n{'=' * 70}")
    print("💡 CONCLUSIONES")
    print(f"{'=' * 70}")
    
    if correct_count / total >= 0.8:
        print("✅ El sistema FUNCIONA correctamente")
        print("   - Detecta preguntas conocidas")
        print("   - Previene alucinaciones sobre temas desconocidos")
        print("   - Mantiene memoria de respuestas verificadas")
    else:
        print("⚠️  El sistema necesita ajustes")
    
    prevention_rate = stats['hallucinations_prevented'] / max(1, stats['from_llm']) * 100
    print(f"\n🛡️  Efectividad anti-alucinación: {prevention_rate:.0f}%")
    
    print(f"\n{'=' * 70}")


def run_interactive_mode():
    """Modo interactivo para probar el sistema"""
    
    print("\n" + "=" * 70)
    print("🎮 MODO INTERACTIVO")
    print("=" * 70)
    print("\nComandos:")
    print("  - Escribe una pregunta")
    print("  - 'stats'  - Ver estadísticas")
    print("  - 'exit'   - Salir")
    print("=" * 70)
    
    guard = AntiHallucinationGuard()
    
    while True:
        try:
            user_input = input("\n💬 Pregunta: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("👋 ¡Hasta luego!")
                break
            
            if user_input.lower() == 'stats':
                stats = guard.get_stats()
                print("\n📊 Estadísticas:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            
            # Procesar pregunta
            result = guard.ask(
                user_input,
                llm_function=mock_llm,
                verification_source=mock_verifier
            )
            
            # Mostrar resultado
            security_emoji = {
                'HIGH': '✅',
                'MEDIUM': '⚠️',
                'LOW': '❌',
                'NONE': '🚫'
            }
            
            emoji = security_emoji.get(result['security_level'], '❓')
            
            print(f"\n{emoji} Respuesta:")
            print(f"   {result['answer']}")
            print(f"\n   Confianza: {result['confidence']*100:.0f}% | Seguridad: {result['security_level']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        run_interactive_mode()
    else:
        run_experiment()
