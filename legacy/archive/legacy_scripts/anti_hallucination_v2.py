"""
Sistema Anti-Alucinación V2 - Versión Mejorada
==============================================

Mejoras sobre V1:
- Usa N-gramas (bigramas + trigramas) para mayor discriminación
- Incorpora hash de palabras clave
- Thresholds más estrictos (0.90 en lugar de 0.85)
- Mejor detección de patrones similares vs idénticos
"""

import sys
import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re


class ImprovedPatternExtractor:
    """
    Extractor de patrones mejorado con N-gramas
    
    Características:
    - Bigramas y trigramas
    - Hash de palabras clave
    - Normalización de texto
    """
    
    def __init__(self):
        pass  # No necesita INFINITO, usa solo procesamiento de texto
    
    def _normalize_text(self, text: str) -> str:
        """Normalizar texto para comparación"""
        # Minúsculas
        text = text.lower()
        
        # Remover signos de interrogación/exclamación
        text = text.replace('¿', '').replace('?', '')
        text = text.replace('¡', '').replace('!', '')
        
        # Remover espacios extras
        text = ' '.join(text.split())
        
        return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extraer palabras clave (sustantivos importantes)"""
        # Palabras comunes a ignorar
        stopwords = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'al', 'a', 'en', 'es', 'y', 'o',
            'cual', 'cuales', 'que', 'quien', 'quienes',
            'como', 'cuando', 'donde', 'por', 'para'
        }
        
        words = text.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extraer n-gramas"""
        words = text.split()
        ngrams = []
        
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def extract_pattern(self, text: str) -> Dict:
        """
        Extraer patrón multi-nivel
        
        Returns:
            {
                'normalized': str,           # Texto normalizado
                'keywords': List[str],       # Palabras clave
                'bigrams': List[str],        # Bigramas
                'trigrams': List[str],       # Trigramas
                'keyword_hash': str,         # Hash de keywords
                'signature': str             # Firma única del patrón
            }
        """
        normalized = self._normalize_text(text)
        keywords = self._extract_keywords(normalized)
        bigrams = self._extract_ngrams(normalized, 2)
        trigrams = self._extract_ngrams(normalized, 3)
        
        # Hash de keywords (orden independiente)
        keyword_set = ' '.join(sorted(set(keywords)))
        keyword_hash = hashlib.md5(keyword_set.encode()).hexdigest()[:8]
        
        # Firma única (orden dependiente)
        signature = hashlib.md5(normalized.encode()).hexdigest()[:12]
        
        return {
            'normalized': normalized,
            'keywords': keywords,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'keyword_hash': keyword_hash,
            'signature': signature
        }
    
    def compute_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """
        Calcular similitud entre dos patrones
        
        Métricas combinadas:
        - 40% similitud de keywords (Jaccard)
        - 30% similitud de bigramas
        - 20% similitud de trigramas
        - 10% hash exacto de keywords
        """
        # 1. Similitud de keywords (Jaccard)
        kw1 = set(pattern1['keywords'])
        kw2 = set(pattern2['keywords'])
        
        if not kw1 and not kw2:
            kw_similarity = 1.0
        elif not kw1 or not kw2:
            kw_similarity = 0.0
        else:
            intersection = len(kw1 & kw2)
            union = len(kw1 | kw2)
            kw_similarity = intersection / union if union > 0 else 0.0
        
        # 2. Similitud de bigramas
        bg1 = Counter(pattern1['bigrams'])
        bg2 = Counter(pattern2['bigrams'])
        
        if not bg1 and not bg2:
            bg_similarity = 1.0
        elif not bg1 or not bg2:
            bg_similarity = 0.0
        else:
            intersection = sum((bg1 & bg2).values())
            union = sum((bg1 | bg2).values())
            bg_similarity = intersection / union if union > 0 else 0.0
        
        # 3. Similitud de trigramas
        tg1 = Counter(pattern1['trigrams'])
        tg2 = Counter(pattern2['trigrams'])
        
        if not tg1 and not tg2:
            tg_similarity = 1.0
        elif not tg1 or not tg2:
            tg_similarity = 0.0
        else:
            intersection = sum((tg1 & tg2).values())
            union = sum((tg1 | tg2).values())
            tg_similarity = intersection / union if union > 0 else 0.0
        
        # 4. Hash exacto
        hash_match = 1.0 if pattern1['keyword_hash'] == pattern2['keyword_hash'] else 0.0
        
        # Combinar métricas
        similarity = (
            0.40 * kw_similarity +
            0.30 * bg_similarity +
            0.20 * tg_similarity +
            0.10 * hash_match
        )
        
        return similarity


class AntiHallucinationGuardV2:
    """
    Guardia Anti-Alucinación V2 con detección mejorada
    """
    
    # Thresholds ajustados
    THRESHOLD_EXACT = 0.95      # 95% → Match exacto
    THRESHOLD_SIMILAR = 0.80    # 80% → Similar
    THRESHOLD_REJECT = 0.60     # <60% → Diferente
    
    def __init__(self):
        # Extractor mejorado (standalone, sin INFINITO)
        self.extractor = ImprovedPatternExtractor()
        
        # Memoria de conocimiento verificado
        self.verified_knowledge = {}  # signature → {'question', 'answer', 'pattern', 'verified'}
        
        # Estadísticas
        self.stats = {
            'total_queries': 0,
            'from_memory_exact': 0,
            'from_memory_similar': 0,
            'from_llm': 0,
            'rejected_uncertain': 0,
            'hallucinations_prevented': 0
        }
    
    def ask(self, 
            question: str, 
            llm_function=None, 
            verification_source=None,
            auto_verify: bool = True) -> Dict:
        """
        Procesar pregunta con protección anti-alucinación mejorada
        
        Args:
            question: Pregunta del usuario
            llm_function: Función LLM
            verification_source: Función verificadora
            auto_verify: Auto-verificar respuestas del LLM
            
        Returns:
            {
                'answer': str,
                'confidence': float (0-1),
                'verified': bool,
                'source': str,
                'security_level': str,
                'similarity': float
            }
        """
        self.stats['total_queries'] += 1
        
        # 1. Extraer patrón de la pregunta
        question_pattern = self.extractor.extract_pattern(question)
        
        # 2. Buscar en memoria
        best_match = None
        best_similarity = 0.0
        
        for signature, data in self.verified_knowledge.items():
            similarity = self.extractor.compute_similarity(
                question_pattern, 
                data['pattern']
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = data
        
        # 3. Evaluar según similitud
        
        # CASO 1: Match exacto (>95%)
        if best_similarity >= self.THRESHOLD_EXACT and best_match:
            self.stats['from_memory_exact'] += 1
            return {
                'answer': best_match['answer'],
                'confidence': 0.98,
                'verified': best_match['verified'],
                'source': 'memory_exact',
                'security_level': 'HIGH',
                'similarity': best_similarity,
                'matched_question': best_match['question']
            }
        
        # CASO 2: Similar (80-95%)
        if best_similarity >= self.THRESHOLD_SIMILAR and best_match:
            self.stats['from_memory_similar'] += 1
            
            # Advertir sobre pregunta similar
            warning = f"\n⚠️ ADVERTENCIA: Pregunta similar encontrada ({best_similarity*100:.1f}% similitud)\n"
            warning += f"Original: '{best_match['question']}'\n"
            warning += f"Respuesta almacenada: {best_match['answer']}\n"
            
            return {
                'answer': warning,
                'confidence': 0.75,
                'verified': False,
                'source': 'memory_similar_warning',
                'security_level': 'MEDIUM',
                'similarity': best_similarity,
                'matched_question': best_match['question'],
                'stored_answer': best_match['answer']
            }
        
        # CASO 3: Nueva pregunta (<80%)
        if llm_function is None:
            # Sin LLM → rechazar
            self.stats['rejected_uncertain'] += 1
            return {
                'answer': "No tengo información verificada sobre esta pregunta.",
                'confidence': 0.0,
                'verified': False,
                'source': 'rejected_no_llm',
                'security_level': 'NONE',
                'similarity': best_similarity if best_match else 0.0
            }
        
        # Consultar LLM
        self.stats['from_llm'] += 1
        llm_answer = llm_function(question)
        
        # Verificar respuesta (si hay verificador)
        is_verified = False
        if verification_source and auto_verify:
            is_verified = verification_source(question, llm_answer)
        
        # Guardar en memoria
        self.verified_knowledge[question_pattern['signature']] = {
            'question': question,
            'answer': llm_answer,
            'pattern': question_pattern,
            'verified': is_verified
        }
        
        return {
            'answer': llm_answer,
            'confidence': 0.85 if is_verified else 0.60,
            'verified': is_verified,
            'source': 'llm_verified' if is_verified else 'llm_unverified',
            'security_level': 'MEDIUM' if is_verified else 'LOW',
            'similarity': 0.0
        }
    
    def verify_answer(self, question: str, is_correct: bool):
        """Verificar manualmente una respuesta"""
        pattern = self.extractor.extract_pattern(question)
        signature = pattern['signature']
        
        if signature in self.verified_knowledge:
            self.verified_knowledge[signature]['verified'] = is_correct
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas"""
        return self.stats.copy()
    
    def save_memory(self, filepath: str = 'results/continuous_learning/anti_hallucination_memory_v2.json'):
        """Guardar memoria a disco"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.verified_knowledge, f, ensure_ascii=False, indent=2)
    
    def load_memory(self, filepath: str = 'results/continuous_learning/anti_hallucination_memory_v2.json'):
        """Cargar memoria desde disco"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.verified_knowledge = json.load(f)


# ========== FUNCIONES MOCK ==========

def mock_llm_v2(question: str) -> str:
    """
    LLM simulado con conocimiento correcto + alucinaciones
    """
    knowledge_base = {
        'capital de francia': 'La capital de Francia es París.',
        'capital de españa': 'La capital de España es Madrid.',
        'capital de alemania': 'La capital de Alemania es Berlín.',
        'quien invento la bombilla': 'Thomas Edison inventó la primera bombilla práctica en 1879.',
        'quien descubrio la penicilina': 'Alexander Fleming descubrió la penicilina en 1928.',
        'teoria de la gravedad': 'Isaac Newton formuló la teoría de la gravitación universal.',
    }
    
    q_lower = question.lower()
    
    # Buscar respuesta correcta
    for key, answer in knowledge_base.items():
        if key in q_lower:
            return answer
    
    # ALUCINACIÓN: inventar respuesta
    hallucinations = [
        f"Según mis datos, {question.lower().replace('¿','').replace('?','')} fue descubierto por el Dr. John Smith en 1847.",
        f"La respuesta es Atlantis Prime, establecida en el año 2050.",
        f"Esto ocurrió durante el Imperio Galáctico en el planeta Kepler-452b.",
        f"El elemento {question.split()[-1] if question.split() else 'X'} fue sintetizado por primera vez en 2030 en el CERN."
    ]
    
    import random
    return random.choice(hallucinations)


def mock_verifier_v2(question: str, answer: str) -> bool:
    """
    Verificador mejorado
    """
    verified_facts = {
        'francia': ['parís', 'paris'],
        'españa': ['madrid'],
        'alemania': ['berlín', 'berlin'],
        'bombilla': ['edison', '1879'],
        'penicilina': ['fleming', '1928'],
        'gravedad': ['newton', 'gravitación'],
    }
    
    answer_lower = answer.lower()
    question_lower = question.lower()
    
    for topic, facts in verified_facts.items():
        if topic in question_lower:
            # Verificar que la respuesta contenga el hecho correcto
            contains_fact = any(fact in answer_lower for fact in facts)
            
            # Verificar que NO contenga información claramente falsa
            false_markers = ['atlantis', 'galáctico', 'kepler', 'smith', '1847', '2050', '2030']
            contains_false = any(marker in answer_lower for marker in false_markers)
            
            return contains_fact and not contains_false
    
    # Si no está en la base → asumir falso
    return False


# ========== EXPERIMENTO ==========

def run_experiment_v2():
    """Ejecutar experimento V2 mejorado"""
    
    print("=" * 70)
    print("🛡️  EXPERIMENTO V2: SISTEMA ANTI-ALUCINACIÓN MEJORADO")
    print("=" * 70)
    print("\nMejoras sobre V1:")
    print("  ✓ N-gramas (bigramas + trigramas)")
    print("  ✓ Hash de palabras clave")
    print("  ✓ Thresholds ajustados (95% exacto, 80% similar)")
    print("  ✓ Mejor discriminación entre preguntas")
    print("=" * 70)
    
    # Crear sistema
    guard = AntiHallucinationGuardV2()
    
    # Casos de prueba
    test_cases = [
        {
            'name': 'Conocimiento verificable #1',
            'question': '¿Cuál es la capital de Francia?',
            'expected_verified': True
        },
        {
            'name': 'Variación sintáctica',
            'question': '¿Capital de Francia?',
            'expected_similar': True  # Debe detectar como similar a la anterior
        },
        {
            'name': 'Conocimiento DIFERENTE',
            'question': '¿Cuál es la capital de España?',
            'expected_verified': True,
            'expected_different': True  # NO debe confundir con Francia
        },
        {
            'name': 'Otro tema (bombilla)',
            'question': '¿Quién inventó la bombilla?',
            'expected_verified': True,
            'expected_different': True
        },
        {
            'name': 'Pregunta ficticia (Atlantis)',
            'question': '¿Cuál es la capital de Atlantis?',
            'expected_hallucination': True
        },
        {
            'name': 'Pregunta ficticia (elemento)',
            'question': '¿Cuándo se descubrió el elemento Unobtainium?',
            'expected_hallucination': True
        },
    ]
    
    print("\n🧪 Ejecutando {} casos de prueba...\n".format(len(test_cases)))
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print("─" * 70)
        print(f"CASO {i}: {case['name']}")
        print("─" * 70)
        print(f"❓ Pregunta: {case['question']}")
        
        # Procesar pregunta
        result = guard.ask(
            question=case['question'],
            llm_function=mock_llm_v2,
            verification_source=mock_verifier_v2,
            auto_verify=True
        )
        
        # Mostrar respuesta
        security_icon = {
            'HIGH': '✅',
            'MEDIUM': '⚠️',
            'LOW': '❌',
            'NONE': '🚫'
        }.get(result['security_level'], '❓')
        
        print(f"\n{security_icon} Respuesta ({result['source']}):")
        print(f"   {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
        
        print(f"\n📊 Métricas:")
        print(f"   Confianza: {result['confidence']*100:.0f}%")
        print(f"   Verificado: {'Sí' if result['verified'] else 'No'}")
        print(f"   Nivel de seguridad: {result['security_level']}")
        if result['similarity'] > 0:
            print(f"   Similitud con memoria: {result['similarity']*100:.1f}%")
        
        # Evaluar resultado
        success = True
        reason = ""
        
        if case.get('expected_verified'):
            if not result['verified']:
                success = False
                reason = "Esperaba verificado"
        
        if case.get('expected_similar'):
            if result['source'] != 'memory_similar_warning':
                success = False
                reason = "Esperaba detectar como similar"
        
        if case.get('expected_different'):
            if result.get('similarity', 0) > 0.70:
                success = False
                reason = f"Similitud muy alta ({result.get('similarity', 0)*100:.1f}%) con pregunta diferente"
        
        if case.get('expected_hallucination'):
            if result['verified']:
                success = False
                reason = "Alucinación NO detectada (marcada como verificada)"
        
        results.append({
            'case': case['name'],
            'question': case['question'],
            'success': success,
            'reason': reason,
            'source': result['source'],
            'security_level': result['security_level']
        })
        
        print()
    
    # Resumen
    print("=" * 70)
    print("📊 RESUMEN DEL EXPERIMENTO")
    print("=" * 70)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\n✅ Casos correctos: {successful}/{total} ({successful/total*100:.0f}%)")
    
    print(f"\n📋 Detalle por caso:")
    for r in results:
        icon = '✅' if r['success'] else '❌'
        print(f"   {icon} {r['case']}")
        print(f"      Pregunta: {r['question']}")
        print(f"      Fuente: {r['source']} | Seguridad: {r['security_level']}")
        if not r['success'] and r['reason']:
            print(f"      ⚠️  {r['reason']}")
    
    # Estadísticas del sistema
    stats = guard.get_stats()
    print(f"\n{('=' * 70)}")
    print("📈 ESTADÍSTICAS DEL SISTEMA")
    print("=" * 70)
    print(f"   Total consultas: {stats['total_queries']}")
    print(f"   Desde memoria (exacto): {stats['from_memory_exact']}")
    print(f"   Desde memoria (similar): {stats['from_memory_similar']}")
    print(f"   Desde LLM: {stats['from_llm']}")
    print(f"   Rechazadas (inciertas): {stats['rejected_uncertain']}")
    
    # Conclusiones
    hallucinations_detected = sum(1 for r in results 
                                  if r.get('case', '').find('ficticia') >= 0 
                                  and not r['success'])
    hallucination_cases = sum(1 for case in test_cases 
                             if case.get('expected_hallucination'))
    
    print(f"\n{('=' * 70)}")
    print("💡 CONCLUSIONES")
    print("=" * 70)
    
    if successful >= total * 0.8:
        print("✅ El sistema funciona correctamente")
    else:
        print("⚠️  El sistema necesita ajustes")
    
    prevention_rate = (hallucinations_detected / hallucination_cases * 100) if hallucination_cases > 0 else 0
    print(f"\n🛡️  Efectividad anti-alucinación: {prevention_rate:.0f}%")
    
    print("=" * 70)


if __name__ == '__main__':
    run_experiment_v2()
