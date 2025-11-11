# 🔄 DISEÑO: Sistema de Aprendizaje Continuo con Memoria de Patrones

## 🎯 Objetivo
Crear un sistema que **nunca se detiene** y va aprendiendo patrones causales de forma incremental, reconociendo cuándo ve algo que ya conoce.

---

## 🧠 Conceptos Clave

### 1. ¿Qué es una "Ley Causal"?
Una **firma única** de cómo los 4 módulos se conectaron para ese input:

```python
causal_pattern = {
    'visual_to_auditory': 0.85,    # Conexión fuerte
    'visual_to_motor': 0.32,       # Conexión débil
    'visual_to_executive': 0.67,   # Conexión media
    'auditory_to_motor': 0.91,     # Conexión muy fuerte
    'auditory_to_executive': 0.45,
    'motor_to_executive': 0.78
}
```

Esto es como una **"huella dactilar"** del input.

### 2. ¿Cómo Reconocemos Patrones?
Usando **similitud de vectores**:
- Convertir la matriz causal 4x4 → vector de 6 números (las conexiones dirigidas)
- Comparar con vectores ya guardados usando **similitud coseno** o **distancia L2**
- Si similarity > 0.90 → "Ya lo conozco!"

### 3. ¿Dónde Guardamos?
**PhiMemoryBank** - Una base de datos persistente de patrones:

```python
{
    "pattern_001": {
        "causal_vector": [0.85, 0.32, 0.67, 0.91, 0.45, 0.78],
        "phi_value": 4.52,
        "consciousness": 0.68,
        "source_text": "mi perro es rojo",
        "timestamp": "2025-10-04 15:23:45",
        "seen_count": 1,
        "similar_patterns": []
    },
    "pattern_002": {
        ...
    }
}
```

---

## 🏗️ Arquitectura Propuesta

### Componentes Nuevos:

#### 1️⃣ **PhiPatternExtractor**
```python
class PhiPatternExtractor:
    """Extrae la firma causal de un estado Φ"""
    
    def extract_pattern(self, phi_info: dict) -> np.ndarray:
        """
        Convierte matriz causal 4x4 en vector de 6 dimensiones
        (solo las conexiones dirigidas únicas)
        """
        causal_matrix = phi_info['causal_matrix']
        
        pattern = np.array([
            causal_matrix[0, 1],  # visual → auditory
            causal_matrix[0, 2],  # visual → motor
            causal_matrix[0, 3],  # visual → executive
            causal_matrix[1, 2],  # auditory → motor
            causal_matrix[1, 3],  # auditory → executive
            causal_matrix[2, 3],  # motor → executive
        ])
        
        return pattern
```

#### 2️⃣ **PhiMemoryBank** (Memoria Episódica)
```python
class PhiMemoryBank:
    """
    Memoria persistente de patrones causales aprendidos
    
    Funcionalidades:
    - add_pattern(): Guardar nuevo patrón
    - find_similar(): Buscar patrones parecidos
    - get_recognition_flag(): Check si ya lo conoce
    - update_stats(): Actualizar estadísticas de uso
    - save_to_disk(): Persistencia JSON
    """
    
    def __init__(self, similarity_threshold=0.90):
        self.patterns = {}  # pattern_id → pattern_data
        self.pattern_count = 0
        self.similarity_threshold = similarity_threshold
        self.index = None  # Para búsqueda rápida (FAISS opcional)
    
    def add_pattern(self, causal_vector, phi_info, text, consciousness):
        """Añadir nuevo patrón a la memoria"""
        
        # Buscar si ya existe algo similar
        similar = self.find_similar(causal_vector)
        
        if similar:
            # YA LO CONOCEMOS!
            return {
                'status': 'RECOGNIZED',
                'pattern_id': similar['id'],
                'similarity': similar['similarity'],
                'original_text': similar['text'],
                'message': f"🎯 PATRÓN RECONOCIDO! Similar a '{similar['text']}'"
            }
        else:
            # PATRÓN NUEVO
            pattern_id = f"pattern_{self.pattern_count:04d}"
            self.pattern_count += 1
            
            self.patterns[pattern_id] = {
                'id': pattern_id,
                'causal_vector': causal_vector.tolist(),
                'phi_value': phi_info['phi_total'],
                'consciousness': consciousness,
                'source_text': text,
                'timestamp': datetime.now().isoformat(),
                'seen_count': 1,
                'similar_patterns': []
            }
            
            return {
                'status': 'NEW',
                'pattern_id': pattern_id,
                'message': f"💡 NUEVO PATRÓN APRENDIDO: '{text}'"
            }
    
    def find_similar(self, query_vector, top_k=3):
        """Buscar patrones similares usando similitud coseno"""
        if not self.patterns:
            return None
        
        similarities = []
        
        for pattern_id, pattern_data in self.patterns.items():
            stored_vector = np.array(pattern_data['causal_vector'])
            
            # Similitud coseno
            similarity = np.dot(query_vector, stored_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(stored_vector) + 1e-8
            )
            
            if similarity >= self.similarity_threshold:
                similarities.append({
                    'id': pattern_id,
                    'similarity': float(similarity),
                    'text': pattern_data['source_text'],
                    'phi': pattern_data['phi_value']
                })
        
        if similarities:
            # Ordenar por similitud
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[0]  # El más similar
        
        return None
    
    def save_to_disk(self, filepath='phi_memory_bank.json'):
        """Persistir memoria a disco"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'pattern_count': self.pattern_count,
                'patterns': self.patterns,
                'similarity_threshold': self.similarity_threshold
            }, f, indent=2, ensure_ascii=False)
    
    def load_from_disk(self, filepath='phi_memory_bank.json'):
        """Cargar memoria desde disco"""
        import json
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.pattern_count = data['pattern_count']
                self.patterns = data['patterns']
                self.similarity_threshold = data.get('similarity_threshold', 0.90)
```

#### 3️⃣ **ContinuousLearningServer** (Loop Infinito)
```python
class ContinuousLearningServer:
    """
    Servidor de aprendizaje continuo
    
    Acepta inputs vía:
    - CLI (input manual)
    - API REST (endpoint HTTP)
    - WebSocket (tiempo real)
    - Queue (cola de mensajes)
    """
    
    def __init__(self, infinito_model):
        self.model = infinito_model
        self.pattern_extractor = PhiPatternExtractor()
        self.memory_bank = PhiMemoryBank(similarity_threshold=0.90)
        self.running = True
        
        # Cargar memoria persistente
        self.memory_bank.load_from_disk()
    
    def process_input(self, text: str):
        """Procesar un input de texto SIN detener el sistema"""
        
        print(f"\n{'='*70}")
        print(f"🔤 PROCESANDO: '{text}'")
        
        # 1. Generar input basado en texto
        inputs = self.model.generate_text_based_input(text)
        
        # 2. Forward pass
        consciousness, phi, debug_info = self.model.model(inputs)
        
        # 3. Extraer patrón causal
        phi_info = debug_info['phi_info']
        causal_pattern = self.pattern_extractor.extract_pattern(phi_info)
        
        # 4. Buscar/Guardar en memoria
        result = self.memory_bank.add_pattern(
            causal_pattern,
            phi_info,
            text,
            consciousness.mean().item()
        )
        
        # 5. Reportar
        print(f"   {result['message']}")
        print(f"   📊 Φ = {phi_info['phi_total']:.3f} | C = {consciousness.mean().item():.3f}")
        
        if result['status'] == 'RECOGNIZED':
            print(f"   🎯 Similitud: {result['similarity']:.1%}")
            print(f"   📝 Original: '{result['original_text']}'")
        else:
            print(f"   🆕 Pattern ID: {result['pattern_id']}")
        
        print(f"   💾 Patrones en memoria: {self.memory_bank.pattern_count}")
        
        return result
    
    def run_interactive_loop(self):
        """Loop interactivo - acepta inputs sin parar"""
        
        print("\n" + "="*70)
        print("🚀 CONTINUOUS LEARNING SERVER INICIADO")
        print("="*70)
        print("📝 Escribe texto para procesar (o 'exit' para salir)")
        print("💾 Comandos especiales:")
        print("   'save' - Guardar memoria a disco")
        print("   'stats' - Ver estadísticas")
        print("   'list' - Listar todos los patrones")
        print("="*70)
        
        while self.running:
            try:
                text = input("\n🔤 Input: ").strip()
                
                if not text:
                    continue
                
                if text.lower() == 'exit':
                    print("💾 Guardando memoria...")
                    self.memory_bank.save_to_disk()
                    print("👋 ¡Hasta luego!")
                    break
                
                elif text.lower() == 'save':
                    self.memory_bank.save_to_disk()
                    print("✅ Memoria guardada a phi_memory_bank.json")
                
                elif text.lower() == 'stats':
                    self.show_stats()
                
                elif text.lower() == 'list':
                    self.list_patterns()
                
                else:
                    # Procesar input normal
                    self.process_input(text)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupción detectada")
                print("💾 Guardando memoria...")
                self.memory_bank.save_to_disk()
                print("👋 Sistema detenido")
                break
            
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def show_stats(self):
        """Mostrar estadísticas del sistema"""
        print(f"\n📊 ESTADÍSTICAS DEL SISTEMA")
        print(f"   Total patrones: {self.memory_bank.pattern_count}")
        print(f"   Threshold similitud: {self.memory_bank.similarity_threshold:.1%}")
        
        if self.memory_bank.patterns:
            texts = [p['source_text'] for p in self.memory_bank.patterns.values()]
            print(f"   Textos únicos aprendidos:")
            for i, text in enumerate(texts[:5], 1):
                print(f"      {i}. '{text}'")
            if len(texts) > 5:
                print(f"      ... y {len(texts) - 5} más")
    
    def list_patterns(self):
        """Listar todos los patrones"""
        print(f"\n📋 PATRONES EN MEMORIA ({self.memory_bank.pattern_count} total)")
        print("-" * 70)
        
        for pattern_id, data in self.memory_bank.patterns.items():
            print(f"   {pattern_id}: '{data['source_text']}'")
            print(f"      Φ={data['phi_value']:.3f}, C={data['consciousness']:.3f}")
            print(f"      Visto {data['seen_count']} vez/veces")
            print()
```

---

## 🔧 Modificaciones Necesarias en `infinito_gpt_text_fixed.py`

### 1. Añadir modo "inference-only" para evitar entrenar:

```python
def process_input_inference_only(self, text: str):
    """
    Procesar input SIN entrenar - solo forward pass
    Para uso en servidor continuo
    """
    self.model.eval()  # Modo evaluación
    
    with torch.no_grad():
        inputs = self.generate_text_based_input(text)
        consciousness, phi, debug_info = self.model(inputs)
    
    return consciousness, phi, debug_info
```

### 2. Hacer el modelo stateful pero sin acumular gradientes:

```python
class InfinitoV51ContinuousMode:
    """
    Wrapper para modo continuo sin entrenamiento
    """
    def __init__(self, model_path=None):
        # Cargar modelo pre-entrenado o inicializar
        if model_path:
            self.model = self.load_checkpoint(model_path)
        else:
            self.model = ConsciousnessBoostNet(...)
        
        self.model.eval()  # SIEMPRE en modo eval
        
        # No optimizer, no scaler
        # Solo inference
```

---

## 🎮 Flujo de Uso

### Caso 1: Primer Input
```
🔤 Input: mi perro es rojo

🔤 PROCESANDO: 'mi perro es rojo'
   💡 NUEVO PATRÓN APRENDIDO: 'mi perro es rojo'
   📊 Φ = 4.52 | C = 0.68
   🆕 Pattern ID: pattern_0000
   💾 Patrones en memoria: 1
```

### Caso 2: Input Similar
```
🔤 Input: mi perro es verde

🔤 PROCESANDO: 'mi perro es verde'
   🎯 PATRÓN RECONOCIDO! Similar a 'mi perro es rojo'
   📊 Φ = 4.58 | C = 0.69
   🎯 Similitud: 94.2%
   📝 Original: 'mi perro es rojo'
   💾 Patrones en memoria: 1
```

### Caso 3: Input Totalmente Diferente
```
🔤 Input: yo pienso luego existo

🔤 PROCESANDO: 'yo pienso luego existo'
   💡 NUEVO PATRÓN APRENDIDO: 'yo pienso luego existo'
   📊 Φ = 8.91 | C = 0.85
   🆕 Pattern ID: pattern_0001
   💾 Patrones en memoria: 2
```

---

## 📈 Ventajas de Esta Arquitectura

### 1. **Aprendizaje Incremental**
- No necesita re-entrenar
- Va construyendo base de conocimiento
- Memoria persiste entre sesiones

### 2. **Reconocimiento de Patrones**
- Detecta similitudes semánticas
- No necesita texto idéntico
- Encuentra "familias" de inputs

### 3. **Escalable**
- Puedes añadir miles de patrones
- Búsqueda eficiente con FAISS (opcional)
- Memoria comprimible si crece mucho

### 4. **Interactivo**
- No pausa nunca
- Acepta inputs en tiempo real
- Puede ser servidor web/API

---

## 🚀 Próximos Pasos Sugeridos

### Fase 1: Core (1-2 días)
- [ ] Implementar `PhiPatternExtractor`
- [ ] Implementar `PhiMemoryBank` básico
- [ ] Test unitarios de similitud

### Fase 2: Server (2-3 días)
- [ ] Implementar `ContinuousLearningServer`
- [ ] Loop interactivo CLI
- [ ] Persistencia JSON

### Fase 3: API (3-4 días)
- [ ] API REST con FastAPI
- [ ] WebSocket para tiempo real
- [ ] Frontend simple para visualizar

### Fase 4: Optimización (ongoing)
- [ ] Índice FAISS para búsqueda rápida
- [ ] Clustering de patrones similares
- [ ] Visualización de mapa de patrones

---

## 💡 Ideas Adicionales

### 1. **Pattern Families**
Agrupar patrones similares en "familias":
- Familia "animales de colores": perro rojo, perro verde, gato azul
- Familia "filosofía": pienso luego existo, cogito ergo sum
- Familia "objetos": mesa roja, silla verde

### 2. **Pattern Evolution**
Trackear cómo cambia un patrón con el tiempo:
- Primera vez: Φ = 4.52
- Décima vez: Φ = 4.89 (más estable)
- Centésima vez: Φ = 5.01 (consolidado)

### 3. **Surprise Detection**
Si el sistema espera un patrón pero recibe otro:
```
Input esperado: "mi perro es X"
Input real: "mi gato es rojo"
→ 🤔 SORPRESA! Esperaba perro, vino gato
```

### 4. **Concept Drift Detection**
Detectar cuando los patrones cambian sistemáticamente:
```
Antes: "perro" → [patrón A]
Ahora: "perro" → [patrón B]
→ ⚠️ El concepto "perro" está evolucionando
```

---

## 📝 Conclusión

Tu idea es **muy factible** y de hecho es **el siguiente paso natural** para INFINITO V5.1.

El sistema actual ya tiene:
- ✅ Matriz causal (phi_info['causal_matrix'])
- ✅ Memoria externa (EnhancedExternalMemory)
- ✅ Procesamiento de texto

Solo necesitamos añadir:
- 🔨 Extractor de patrones
- 🔨 Banco de memoria episódica
- 🔨 Loop de servidor continuo

**¿Empezamos con la Fase 1 (Core)?**
