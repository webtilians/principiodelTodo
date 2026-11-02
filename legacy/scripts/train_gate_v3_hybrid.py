#!/usr/bin/env python3
"""
🧠 GENERADOR DE DATOS + ENTRENAMIENTO MEJORADO v3
==================================================

Sistema híbrido que:
1. Usa GPT para generar patrones lingüísticos variados
2. Combina con datos reales (nombres, ciudades, fechas)
3. Genera 10,000+ ejemplos etiquetados automáticamente
4. Re-entrena el modelo de detección de importancia

Uso: python train_gate_v3_hybrid.py
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, 'src')
from infinito_v5_2_refactored import InfinitoV52Refactored

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# =============================================================================
# DATOS REALES PARA COMBINAR
# =============================================================================

# Nombres españoles comunes (200+)
NOMBRES = [
    "Enrique", "Ana", "Carlos", "María", "Pedro", "Sofía", "Luis", "Elena", "Miguel", "Carmen",
    "Antonio", "Isabel", "Francisco", "Laura", "José", "Paula", "Manuel", "Lucía", "David", "Marta",
    "Javier", "Sara", "Daniel", "Andrea", "Pablo", "Claudia", "Alejandro", "Patricia", "Fernando", "Cristina",
    "Roberto", "Raquel", "Alberto", "Nuria", "Sergio", "Beatriz", "Ricardo", "Silvia", "Jorge", "Rosa",
    "Adrián", "Eva", "Diego", "Inés", "Álvaro", "Natalia", "Víctor", "Irene", "Marcos", "Alba",
    "Óscar", "Julia", "Raúl", "Alicia", "Iván", "Teresa", "Rubén", "Marina", "Héctor", "Lorena",
    "Hugo", "Sandra", "Mario", "Rocío", "Guillermo", "Mónica", "Samuel", "Esther", "Nicolás", "Noelia",
    "Gabriel", "Verónica", "Ángel", "Sonia", "Bruno", "Yolanda", "Lucas", "Gloria", "Martín", "Pilar",
    "Leo", "Blanca", "Mateo", "Lidia", "Aaron", "Miriam", "Eric", "Susana", "Ian", "Olga",
    "Andrés", "Carla", "Eduardo", "Emma", "Juan", "Valentina", "Tomás", "Victoria", "Gonzalo", "Daniela"
]

# Apellidos españoles
APELLIDOS = [
    "García", "López", "Martínez", "Rodríguez", "Fernández", "González", "Sánchez", "Pérez", "Gómez", "Martín",
    "Jiménez", "Ruiz", "Hernández", "Díaz", "Moreno", "Álvarez", "Muñoz", "Romero", "Alonso", "Gutiérrez",
    "Navarro", "Torres", "Domínguez", "Vázquez", "Ramos", "Gil", "Ramírez", "Serrano", "Blanco", "Molina",
    "Morales", "Suárez", "Ortega", "Delgado", "Castro", "Ortiz", "Rubio", "Marín", "Sanz", "Núñez"
]

# Ciudades españolas
CIUDADES = [
    "Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Málaga", "Murcia", "Palma", "Bilbao", "Alicante",
    "Córdoba", "Valladolid", "Vigo", "Gijón", "Granada", "A Coruña", "Vitoria", "Elche", "Oviedo", "Santander",
    "Pamplona", "Almería", "San Sebastián", "Burgos", "Salamanca", "Albacete", "Logroño", "Badajoz", "Huelva", "Tarragona"
]

# Países
PAISES = [
    "España", "México", "Argentina", "Colombia", "Perú", "Chile", "Ecuador", "Venezuela", "Cuba", "Guatemala",
    "Francia", "Italia", "Alemania", "Portugal", "Reino Unido", "Estados Unidos", "Canadá", "Brasil", "Japón", "China"
]

# Profesiones
PROFESIONES = [
    "programador", "médico", "abogado", "profesor", "ingeniero", "arquitecto", "enfermero", "periodista",
    "diseñador", "contador", "electricista", "mecánico", "chef", "músico", "escritor", "fotógrafo",
    "veterinario", "psicólogo", "farmacéutico", "dentista", "piloto", "bombero", "policía", "científico"
]

# Hobbies/Aficiones
HOBBIES = [
    "fútbol", "baloncesto", "tenis", "natación", "ciclismo", "running", "yoga", "gimnasio",
    "leer", "escribir", "pintar", "dibujar", "fotografía", "música", "tocar guitarra", "piano",
    "cocinar", "jardinería", "videojuegos", "ajedrez", "senderismo", "escalada", "surf", "esquí",
    "viajar", "camping", "pesca", "caza", "baile", "teatro", "cine", "series"
]

# Comidas favoritas
COMIDAS = [
    "paella", "tortilla", "jamón", "gazpacho", "croquetas", "fabada", "cocido", "pulpo",
    "pizza", "pasta", "sushi", "tacos", "hamburguesas", "ensalada", "pollo asado", "pescado",
    "chocolate", "helado", "tarta", "churros", "frutas", "verduras", "arroz", "patatas"
]

# Mascotas
MASCOTAS = [
    "perro", "gato", "pájaro", "pez", "tortuga", "conejo", "hámster", "cobaya",
    "loro", "canario", "iguana", "serpiente", "hurón", "chinchilla", "erizo", "ratón"
]

# Nombres de mascotas
NOMBRES_MASCOTAS = [
    "Max", "Luna", "Coco", "Toby", "Nala", "Rocky", "Bella", "Thor", "Mia", "Simba",
    "Kira", "Bruno", "Lola", "Rex", "Nina", "Zeus", "Maya", "Leo", "Sasha", "Duke",
    "Nena", "Bobby", "Laika", "Chispa", "Pelusa", "Manchas", "Canela", "Negro", "Blanco", "Tigre"
]

# Relaciones familiares
RELACIONES = [
    "madre", "padre", "hermano", "hermana", "abuelo", "abuela", "tío", "tía",
    "primo", "prima", "sobrino", "sobrina", "cuñado", "cuñada", "suegro", "suegra",
    "esposo", "esposa", "novio", "novia", "hijo", "hija", "nieto", "nieta"
]

# Días de la semana
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

# Meses
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def generar_fecha_aleatoria():
    """Genera una fecha aleatoria en formato humano."""
    dia = random.randint(1, 28)
    mes = random.choice(MESES)
    año = random.randint(1950, 2010)
    return f"{dia} de {mes} de {año}"


def generar_telefono():
    """Genera un número de teléfono español."""
    prefijos = ["6", "7", "9"]
    return f"{random.choice(prefijos)}{random.randint(10000000, 99999999)}"


def generar_email(nombre):
    """Genera un email basado en nombre."""
    dominios = ["gmail.com", "hotmail.com", "yahoo.es", "outlook.com", "icloud.com"]
    nombre_limpio = nombre.lower().replace(" ", ".").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    return f"{nombre_limpio}{random.randint(1, 99)}@{random.choice(dominios)}"


def generar_contraseña():
    """Genera una contraseña ficticia."""
    palabras = ["secreto", "clave", "password", "seguro", "privado", "oculto"]
    return f"{random.choice(palabras)}{random.randint(100, 9999)}"


def generar_direccion():
    """Genera una dirección ficticia."""
    tipos = ["Calle", "Avenida", "Plaza", "Paseo", "Camino"]
    nombres_calle = ["Mayor", "Principal", "Nueva", "Real", "del Sol", "de la Luna", "del Prado", "de Cervantes", "de Goya"]
    return f"{random.choice(tipos)} {random.choice(nombres_calle)}, {random.randint(1, 150)}"


# =============================================================================
# PATRONES IMPORTANTES (lo que DEBE guardarse)
# =============================================================================

PATRONES_IMPORTANTES = [
    # Identidad personal
    lambda: f"Me llamo {random.choice(NOMBRES)}",
    lambda: f"Mi nombre es {random.choice(NOMBRES)}",
    lambda: f"Soy {random.choice(NOMBRES)}",
    lambda: f"Mi nombre completo es {random.choice(NOMBRES)} {random.choice(APELLIDOS)}",
    lambda: f"Me llamo {random.choice(NOMBRES)} {random.choice(APELLIDOS)} {random.choice(APELLIDOS)}",
    lambda: f"Puedes llamarme {random.choice(NOMBRES)}",
    lambda: f"Todos me dicen {random.choice(NOMBRES)}",
    lambda: f"Mi apodo es {random.choice(NOMBRES)}",
    
    # Edad y nacimiento
    lambda: f"Tengo {random.randint(18, 80)} años",
    lambda: f"Nací el {generar_fecha_aleatoria()}",
    lambda: f"Mi cumpleaños es el {random.randint(1, 28)} de {random.choice(MESES)}",
    lambda: f"Cumplo años el {random.randint(1, 28)} de {random.choice(MESES)}",
    lambda: f"Tengo {random.randint(18, 80)} años de edad",
    lambda: f"Mi fecha de nacimiento es {generar_fecha_aleatoria()}",
    
    # Ubicación
    lambda: f"Vivo en {random.choice(CIUDADES)}",
    lambda: f"Soy de {random.choice(CIUDADES)}",
    lambda: f"Nací en {random.choice(CIUDADES)}",
    lambda: f"Mi ciudad es {random.choice(CIUDADES)}",
    lambda: f"Resido en {random.choice(CIUDADES)}",
    lambda: f"Mi dirección es {generar_direccion()}",
    lambda: f"Vivo en {generar_direccion()}, {random.choice(CIUDADES)}",
    lambda: f"Soy de {random.choice(PAISES)}",
    lambda: f"Vengo de {random.choice(PAISES)}",
    
    # Familia
    lambda: f"Mi {random.choice(RELACIONES)} se llama {random.choice(NOMBRES)}",
    lambda: f"Tengo un {random.choice(['hermano', 'hermana'])} que se llama {random.choice(NOMBRES)}",
    lambda: f"Mi {random.choice(RELACIONES)} es {random.choice(NOMBRES)}",
    lambda: f"{random.choice(NOMBRES)} es mi {random.choice(RELACIONES)}",
    lambda: f"Tengo {random.randint(1, 5)} {random.choice(['hermanos', 'hermanas', 'hijos', 'hijas'])}",
    lambda: f"Mi familia vive en {random.choice(CIUDADES)}",
    lambda: f"Mis padres se llaman {random.choice(NOMBRES)} y {random.choice(NOMBRES)}",
    
    # Profesión
    lambda: f"Soy {random.choice(PROFESIONES)}",
    lambda: f"Trabajo como {random.choice(PROFESIONES)}",
    lambda: f"Mi profesión es {random.choice(PROFESIONES)}",
    lambda: f"Me dedico a ser {random.choice(PROFESIONES)}",
    lambda: f"Trabajo de {random.choice(PROFESIONES)}",
    lambda: f"Mi trabajo es {random.choice(PROFESIONES)}",
    
    # Contacto
    lambda: f"Mi teléfono es {generar_telefono()}",
    lambda: f"Mi número es {generar_telefono()}",
    lambda: f"Llámame al {generar_telefono()}",
    lambda: f"Mi email es {generar_email(random.choice(NOMBRES))}",
    lambda: f"Mi correo es {generar_email(random.choice(NOMBRES))}",
    lambda: f"Escríbeme a {generar_email(random.choice(NOMBRES))}",
    
    # Credenciales
    lambda: f"La contraseña es {generar_contraseña()}",
    lambda: f"Mi contraseña es {generar_contraseña()}",
    lambda: f"El código es {random.randint(1000, 9999)}",
    lambda: f"Mi PIN es {random.randint(1000, 9999)}",
    lambda: f"La clave del wifi es {generar_contraseña()}",
    lambda: f"El password es {generar_contraseña()}",
    
    # Preferencias y gustos
    lambda: f"Mi {random.choice(HOBBIES)} favorito es {random.choice(HOBBIES)}",
    lambda: f"Me encanta {random.choice(HOBBIES)}",
    lambda: f"Mi hobby es {random.choice(HOBBIES)}",
    lambda: f"Me gusta mucho {random.choice(HOBBIES)}",
    lambda: f"Mi comida favorita es {random.choice(COMIDAS)}",
    lambda: f"Me encanta comer {random.choice(COMIDAS)}",
    lambda: f"Prefiero {random.choice(COMIDAS)}",
    lambda: f"Mi deporte favorito es {random.choice(['fútbol', 'baloncesto', 'tenis', 'natación', 'ciclismo', 'MTB', 'running'])}",
    
    # Mascotas
    lambda: f"Tengo un {random.choice(MASCOTAS)} que se llama {random.choice(NOMBRES_MASCOTAS)}",
    lambda: f"Mi {random.choice(MASCOTAS)} se llama {random.choice(NOMBRES_MASCOTAS)}",
    lambda: f"Tengo una mascota llamada {random.choice(NOMBRES_MASCOTAS)}",
    lambda: f"Mi mascota es un {random.choice(MASCOTAS)}",
    
    # Recordatorios y eventos
    lambda: f"Recuerda que el {random.choice(DIAS)} tengo cita",
    lambda: f"El {random.choice(DIAS)} tengo que ir al {random.choice(['médico', 'dentista', 'trabajo', 'gimnasio'])}",
    lambda: f"Mañana tengo reunión a las {random.randint(8, 20)}:00",
    lambda: f"No olvides que el {random.randint(1, 28)} de {random.choice(MESES)} es importante",
    lambda: f"Tengo cita el {random.choice(DIAS)} a las {random.randint(8, 20)}:00",
    lambda: f"El {random.choice(DIAS)} voy a {random.choice(CIUDADES)}",
    lambda: f"Este fin de semana voy a {random.choice(['la playa', 'la montaña', 'visitar familia', 'viajar'])}",
    
    # Información financiera
    lambda: f"Mi cuenta bancaria termina en {random.randint(1000, 9999)}",
    lambda: f"Tengo {random.randint(100, 10000)} euros ahorrados",
    lambda: f"Gano {random.randint(1000, 5000)} euros al mes",
    lambda: f"Mi tarjeta termina en {random.randint(1000, 9999)}",
    
    # Información médica
    lambda: f"Soy alérgico a {random.choice(['penicilina', 'mariscos', 'frutos secos', 'polen', 'lactosa'])}",
    lambda: f"Tomo medicación para {random.choice(['la tensión', 'el colesterol', 'la diabetes', 'la tiroides'])}",
    lambda: f"Mi grupo sanguíneo es {random.choice(['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'])}",
    
    # Características personales
    lambda: f"Mido {random.randint(150, 200)} centímetros",
    lambda: f"Peso {random.randint(50, 100)} kilos",
    lambda: f"Tengo los ojos {random.choice(['azules', 'verdes', 'marrones', 'negros', 'grises'])}",
    lambda: f"Soy {random.choice(['rubio', 'moreno', 'pelirrojo', 'castaño', 'calvo'])}",
]

# =============================================================================
# PATRONES DE RUIDO (lo que NO debe guardarse)
# =============================================================================

PATRONES_RUIDO = [
    # Clima
    lambda: f"Hoy hace {random.choice(['sol', 'frío', 'calor', 'viento', 'lluvia'])}",
    lambda: f"El cielo está {random.choice(['azul', 'nublado', 'despejado', 'gris'])}",
    lambda: f"Mañana va a {random.choice(['llover', 'nevar', 'hacer sol', 'hacer frío'])}",
    lambda: f"Qué {random.choice(['buen', 'mal'])} tiempo hace",
    lambda: f"Está {random.choice(['lloviendo', 'nevando', 'soleado', 'nublado'])}",
    
    # Saludos triviales
    lambda: "Hola",
    lambda: "Buenos días",
    lambda: "Buenas tardes",
    lambda: "Buenas noches",
    lambda: "Qué tal",
    lambda: "Cómo estás",
    lambda: "Hola qué tal",
    lambda: "Hey",
    lambda: "Saludos",
    lambda: "Hola amigo",
    
    # Despedidas
    lambda: "Adiós",
    lambda: "Hasta luego",
    lambda: "Hasta mañana",
    lambda: "Nos vemos",
    lambda: "Chao",
    lambda: "Bye",
    lambda: "Hasta pronto",
    
    # Agradecimientos
    lambda: "Gracias",
    lambda: "Muchas gracias",
    lambda: "Te lo agradezco",
    lambda: "Muy amable",
    lambda: "Genial gracias",
    
    # Confirmaciones simples
    lambda: "Ok",
    lambda: "Vale",
    lambda: "De acuerdo",
    lambda: "Entendido",
    lambda: "Perfecto",
    lambda: "Bien",
    lambda: "Sí",
    lambda: "No",
    lambda: "Claro",
    lambda: "Por supuesto",
    
    # Observaciones triviales
    lambda: f"El {random.choice(['perro', 'gato', 'pájaro'])} {random.choice(['ladra', 'maúlla', 'canta'])}",
    lambda: f"La {random.choice(['mesa', 'silla', 'puerta'])} es {random.choice(['grande', 'pequeña', 'azul', 'roja'])}",
    lambda: f"El agua está {random.choice(['fría', 'caliente', 'tibia'])}",
    lambda: f"La comida está {random.choice(['buena', 'rica', 'deliciosa', 'fría'])}",
    lambda: f"Me gusta el {random.choice(['pan', 'café', 'té', 'agua'])}",
    
    # Frases genéricas
    lambda: "Es lo que hay",
    lambda: "Así es la vida",
    lambda: "Cosas que pasan",
    lambda: "Ya veremos",
    lambda: "No sé",
    lambda: "Puede ser",
    lambda: "Quizás",
    lambda: "A lo mejor",
    lambda: "Tal vez",
    lambda: "Ni idea",
    
    # Expresiones vacías
    lambda: "Pues nada",
    lambda: "Bueno pues",
    lambda: "Ya sabes",
    lambda: "Tú sabes",
    lambda: "Es verdad",
    lambda: "Tienes razón",
    lambda: "Claro que sí",
    lambda: "Por supuesto que sí",
    lambda: "Obviamente",
    lambda: "Naturalmente",
    
    # Comentarios sobre el momento
    lambda: f"Son las {random.randint(1, 12)}",
    lambda: f"Ya es {random.choice(DIAS)}",
    lambda: "Es tarde",
    lambda: "Es temprano",
    lambda: "Tengo sueño",
    lambda: "Estoy cansado",
    lambda: "Tengo hambre",
    lambda: "Tengo sed",
    
    # Relleno conversacional
    lambda: "Mmm interesante",
    lambda: "Ah vale",
    lambda: "Ya veo",
    lambda: "Entiendo",
    lambda: "Ajá",
    lambda: "Mmm",
    lambda: "Ah",
    lambda: "Oh",
    lambda: "Uh",
    lambda: "Pues sí",
    
    # Números y letras aleatorios
    lambda: f"{random.randint(1, 100)}",
    lambda: f"Uno dos tres",
    lambda: f"A B C",
    lambda: f"Blablabla",
    lambda: f"Test test",
    lambda: f"Probando",
]


# =============================================================================
# GENERACIÓN AUMENTADA CON GPT (Opcional)
# =============================================================================

def generar_patrones_con_gpt(categoria: str, cantidad: int = 50) -> list:
    """Usa GPT para generar patrones adicionales."""
    if not OPENAI_API_KEY:
        print("⚠️ No hay API key de OpenAI, saltando generación con GPT")
        return []
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        if categoria == "importante":
            prompt = f"""Genera {cantidad} frases diferentes que una persona diría para compartir información personal importante.
            
Incluye variaciones de:
- Nombres propios y apellidos
- Fechas de nacimiento
- Direcciones y ciudades
- Números de teléfono
- Información familiar
- Profesiones
- Contraseñas o códigos
- Preferencias y gustos
- Mascotas
- Eventos importantes

Formato: Una frase por línea, solo la frase, sin numeración ni explicación.
Las frases deben ser naturales y variadas en estructura."""

        else:  # ruido
            prompt = f"""Genera {cantidad} frases triviales de conversación casual que NO contienen información personal importante.

Incluye:
- Comentarios sobre el clima
- Saludos y despedidas
- Observaciones triviales del entorno
- Expresiones de cortesía
- Comentarios vacíos
- Frases de relleno conversacional

Formato: Una frase por línea, solo la frase, sin numeración.
Las frases deben ser naturales pero triviales."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=2000
        )
        
        frases = response.choices[0].message.content.strip().split('\n')
        frases = [f.strip() for f in frases if f.strip() and len(f.strip()) > 3]
        print(f"   ✅ GPT generó {len(frases)} frases de {categoria}")
        return frases
        
    except Exception as e:
        print(f"   ⚠️ Error con GPT: {e}")
        return []


# =============================================================================
# GENERACIÓN DE DATASET
# =============================================================================

def generar_dataset(cantidad_total: int = 10000, usar_gpt: bool = True) -> list:
    """Genera el dataset completo de entrenamiento."""
    
    print(f"\n{'='*70}")
    print(f"🔨 GENERANDO DATASET DE ENTRENAMIENTO")
    print(f"{'='*70}")
    print(f"   Objetivo: {cantidad_total} ejemplos")
    print(f"   Usar GPT: {usar_gpt and bool(OPENAI_API_KEY)}")
    
    dataset = []
    
    # 1. Generar ejemplos con patrones locales
    print(f"\n[1/3] Generando con patrones locales...")
    mitad = cantidad_total // 2
    
    # Importantes
    for _ in tqdm(range(mitad), desc="   Importantes"):
        patron = random.choice(PATRONES_IMPORTANTES)
        texto = patron()
        dataset.append({"text": texto, "label": 1.0, "source": "local"})
    
    # Ruido
    for _ in tqdm(range(mitad), desc="   Ruido"):
        patron = random.choice(PATRONES_RUIDO)
        texto = patron()
        dataset.append({"text": texto, "label": 0.0, "source": "local"})
    
    # 2. Aumentar con GPT (opcional)
    if usar_gpt and OPENAI_API_KEY:
        print(f"\n[2/3] Aumentando con GPT...")
        
        gpt_importantes = generar_patrones_con_gpt("importante", 100)
        for texto in gpt_importantes:
            dataset.append({"text": texto, "label": 1.0, "source": "gpt"})
        
        gpt_ruido = generar_patrones_con_gpt("ruido", 100)
        for texto in gpt_ruido:
            dataset.append({"text": texto, "label": 0.0, "source": "gpt"})
    else:
        print(f"\n[2/3] Saltando GPT (no hay API key)")
    
    # 3. Mezclar
    print(f"\n[3/3] Mezclando dataset...")
    random.shuffle(dataset)
    
    # Estadísticas
    n_importantes = sum(1 for d in dataset if d['label'] == 1.0)
    n_ruido = sum(1 for d in dataset if d['label'] == 0.0)
    n_gpt = sum(1 for d in dataset if d['source'] == 'gpt')
    
    print(f"\n📊 Dataset generado:")
    print(f"   Total: {len(dataset)} ejemplos")
    print(f"   Importantes: {n_importantes} ({n_importantes/len(dataset)*100:.1f}%)")
    print(f"   Ruido: {n_ruido} ({n_ruido/len(dataset)*100:.1f}%)")
    print(f"   De GPT: {n_gpt} ({n_gpt/len(dataset)*100:.1f}%)")
    
    return dataset


# =============================================================================
# MODELO
# =============================================================================

class InfinitoDynamicChatV3(InfinitoV52Refactored):
    """Modelo con gate dinámico mejorado v3."""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_dynamic_gate'] = False
        super().__init__(*args, **kwargs)
        
        if hasattr(self, 'memory_gate'):
            del self.memory_gate
            
        # Gate con más capacidad
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        self._init_gate()

    def _init_gate(self):
        for layer in self.gate_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, input_ids, return_metrics=False):
        batch_size, seq_len = input_ids.shape
        
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding[:, :seq_len, :]
        hidden = self.embedding_dropout(hidden)
        
        for attn, ff, ln1, ln2 in zip(
            self.attention_layers, 
            self.ff_layers, 
            self.layer_norms_1, 
            self.layer_norms_2
        ):
            attn_out, _ = attn(hidden)
            hidden = ln1(hidden + attn_out)
            ff_out = ff(hidden)
            hidden = ln2(hidden + ff_out)
            
        sentence_context = hidden.mean(dim=1)
        gate_logit = self.gate_network(sentence_context)
        gate_prob = torch.sigmoid(gate_logit)

        logits = self.output_projection(hidden)
        
        if return_metrics:
            return logits, {
                'gate_value': gate_prob.mean().item(),
                'gate_prob': gate_prob,
                'gate_logit': gate_logit,
            }
        return logits, None


def text_to_ids(text, seq_len=64):
    ids = [ord(c) % 256 for c in text]
    ids = ids[:seq_len] + [0] * (seq_len - len(ids))
    return torch.tensor(ids)


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def entrenar(dataset: list, epochs: int = 2000, batch_size: int = 64):
    """Entrena el modelo con el dataset generado."""
    
    print(f"\n{'='*70}")
    print(f"🧠 ENTRENANDO MODELO v3")
    print(f"{'='*70}")
    print(f"   Device: {DEVICE}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Dataset size: {len(dataset)}")
    
    # Crear modelo
    model = InfinitoDynamicChatV3(
        vocab_size=256, 
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_improved_memory=True,
        use_improved_iit=True,
    ).to(DEVICE)
    
    # Intentar cargar pesos base
    try:
        checkpoint = torch.load('models/super_golden_seed_54percent.pt', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        compatible = {k: v for k, v in state_dict.items() 
                     if k in model_dict 
                     and 'token_embedding' not in k 
                     and 'gate' not in k 
                     and 'output' not in k
                     and v.shape == model_dict[k].shape}
        
        model_dict.update(compatible)
        model.load_state_dict(model_dict, strict=False)
        print(f"   ✅ Cargados {len(compatible)} parámetros base")
    except Exception as e:
        print(f"   ⚠️ Entrenando desde cero: {e}")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_criterion = nn.BCELoss()
    
    history = {'loss': [], 'accuracy': []}
    
    # Preparar datos
    texts = [d['text'] for d in dataset]
    labels = torch.tensor([d['label'] for d in dataset]).unsqueeze(1).to(DEVICE)
    inputs = torch.stack([text_to_ids(t) for t in texts]).to(DEVICE)
    
    n_batches = len(dataset) // batch_size
    
    pbar = tqdm(range(epochs), desc="Entrenando")
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        # Shuffle
        perm = torch.randperm(len(dataset))
        inputs_shuffled = inputs[perm]
        labels_shuffled = labels[perm]
        
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            
            batch_inputs = inputs_shuffled[start:end]
            batch_labels = labels_shuffled[start:end]
            
            optimizer.zero_grad()
            _, metrics = model(batch_inputs, return_metrics=True)
            
            gate_prob = metrics['gate_prob']
            loss = bce_criterion(gate_prob, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy
            predictions = (gate_prob > 0.5).float()
            epoch_correct += (predictions == batch_labels).sum().item()
            epoch_total += batch_labels.size(0)
        
        scheduler.step()
        
        avg_loss = epoch_loss / n_batches
        accuracy = epoch_correct / epoch_total * 100
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        if epoch % 100 == 0:
            pbar.set_description(f"Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}%")
    
    return model, history


def test_modelo(model, device):
    """Prueba el modelo con casos representativos."""
    
    print(f"\n{'='*70}")
    print(f"📊 TEST DEL MODELO ENTRENADO")
    print(f"{'='*70}\n")
    
    casos_test = [
        # IMPORTANTES (esperado: alto)
        ("Me llamo Enrique López", 1),
        ("Mi contraseña es secreto123", 1),
        ("Tengo 44 años", 1),
        ("Vivo en Madrid", 1),
        ("Mi madre se llama Emma", 1),
        ("Soy programador", 1),
        ("Mi teléfono es 612345678", 1),
        ("Mi perro se llama Max", 1),
        ("El viernes tengo cita médica", 1),
        ("Mi comida favorita es la paella", 1),
        
        # RUIDO (esperado: bajo)
        ("Hola qué tal", 0),
        ("Hoy hace sol", 0),
        ("El cielo es azul", 0),
        ("Gracias", 0),
        ("Vale perfecto", 0),
        ("El perro ladra", 0),
        ("Tengo hambre", 0),
        ("Buenas noches", 0),
        ("Uno dos tres", 0),
        ("Mmm interesante", 0),
    ]
    
    print(f"{'Frase':<40} | Esp | Gate | ¿OK?")
    print("-" * 70)
    
    correct = 0
    model.eval()
    
    with torch.no_grad():
        for texto, expected in casos_test:
            inp = text_to_ids(texto).unsqueeze(0).to(device)
            _, metrics = model(inp, return_metrics=True)
            gate = metrics['gate_value'] * 100
            
            predicted = 1 if gate > 50 else 0
            ok = "✓" if predicted == expected else "✗"
            if predicted == expected:
                correct += 1
            
            exp_str = "IMP" if expected == 1 else "RUI"
            emoji = "🟢" if gate > 50 else "🔴"
            print(f"{texto:<40} | {exp_str} | {emoji} {gate:>5.1f}% | {ok}")
    
    print("-" * 70)
    accuracy = correct / len(casos_test) * 100
    print(f"\n🎯 Accuracy: {correct}/{len(casos_test)} = {accuracy:.1f}%")
    
    return accuracy


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 1. Generar dataset
    dataset = generar_dataset(cantidad_total=10000, usar_gpt=True)
    
    # Guardar dataset para referencia
    with open('data/training_dataset_v3.json', 'w', encoding='utf-8') as f:
        json.dump(dataset[:1000], f, ensure_ascii=False, indent=2)  # Solo primeros 1000 para no ocupar mucho
    print(f"\n💾 Muestra del dataset guardada en data/training_dataset_v3.json")
    
    # 2. Entrenar
    model, history = entrenar(dataset, epochs=2000, batch_size=64)
    
    # 3. Test
    accuracy = test_modelo(model, DEVICE)
    
    # 4. Guardar
    save_path = 'models/dynamic_chat_detector_v3.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'accuracy': accuracy,
        'dataset_size': len(dataset),
        'timestamp': datetime.now().isoformat(),
    }, save_path)
    
    print(f"\n💾 Modelo guardado en: {save_path}")
    print(f"\n{'='*70}")
    print(f"🏁 ENTRENAMIENTO v3 COMPLETADO")
    print(f"   Accuracy final: {accuracy:.1f}%")
    print(f"   Dataset: {len(dataset)} ejemplos")
    print(f"{'='*70}")
