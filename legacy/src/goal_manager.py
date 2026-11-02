"""
🎯 GOAL MANAGER - Sistema de Objetivos Persistentes
====================================================

Gestiona objetivos a corto, medio y largo plazo para el cerebro Infinito.
Permite que el sistema sea PROACTIVO, no solo reactivo.

Tipos de objetivos:
- INMEDIATO: Responder al usuario
- CORTO PLAZO: Recordatorios, citas, tareas
- LARGO PLAZO: Aprender sobre el usuario, hábitos

Autor: INFINITO Project
Fecha: Noviembre 2025
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum


class GoalPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class GoalType(Enum):
    REMINDER = "reminder"           # Recordar algo al usuario
    FOLLOW_UP = "follow_up"         # Preguntar cómo fue algo
    LEARNING = "learning"           # Aprender más sobre un tema
    HABIT = "habit"                 # Comportamiento recurrente
    ALERT = "alert"                 # Avisar si detecta algo
    TASK = "task"                   # Tarea pendiente


class Goal:
    """Representa un objetivo individual."""
    
    def __init__(
        self,
        id: str,
        description: str,
        goal_type: GoalType,
        priority: GoalPriority = GoalPriority.MEDIUM,
        trigger_date: Optional[datetime] = None,
        trigger_keywords: Optional[List[str]] = None,
        context: Optional[Dict] = None,
        created_at: Optional[datetime] = None,
        completed: bool = False,
        completed_at: Optional[datetime] = None
    ):
        self.id = id
        self.description = description
        self.goal_type = goal_type
        self.priority = priority
        self.trigger_date = trigger_date
        self.trigger_keywords = trigger_keywords or []
        self.context = context or {}
        self.created_at = created_at or datetime.now()
        self.completed = completed
        self.completed_at = completed_at
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "goal_type": self.goal_type.value,
            "priority": self.priority.value,
            "trigger_date": self.trigger_date.isoformat() if self.trigger_date else None,
            "trigger_keywords": self.trigger_keywords,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "completed": self.completed,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Goal':
        return cls(
            id=data["id"],
            description=data["description"],
            goal_type=GoalType(data["goal_type"]),
            priority=GoalPriority(data["priority"]),
            trigger_date=datetime.fromisoformat(data["trigger_date"]) if data.get("trigger_date") else None,
            trigger_keywords=data.get("trigger_keywords", []),
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            completed=data.get("completed", False),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        )
    
    def is_triggered(self, current_text: str = "", current_time: Optional[datetime] = None) -> bool:
        """Verifica si el objetivo debe activarse ahora."""
        if self.completed:
            return False
        
        current_time = current_time or datetime.now()
        
        # Trigger por fecha
        if self.trigger_date:
            # Activar si estamos en el mismo día o ya pasó (y no está completado)
            if current_time.date() >= self.trigger_date.date():
                # Si tiene hora específica, verificar que estemos cerca
                if self.trigger_date.hour > 0:
                    time_diff = abs((current_time - self.trigger_date).total_seconds())
                    if time_diff < 3600:  # Dentro de 1 hora
                        return True
                else:
                    return True
        
        # Trigger por palabras clave
        if self.trigger_keywords and current_text:
            text_lower = current_text.lower()
            if any(kw.lower() in text_lower for kw in self.trigger_keywords):
                return True
        
        return False


class GoalManager:
    """
    Gestor de objetivos persistentes.
    
    Permite al sistema:
    - Crear objetivos basados en lo que dice el usuario
    - Recordar citas y tareas
    - Hacer seguimiento de eventos pasados
    - Ser proactivo en las interacciones
    """
    
    def __init__(self, db_file: str = "goals.json"):
        self.db_file = db_file
        self.goals: List[Goal] = []
        self._load()
    
    def _load(self):
        """Carga objetivos desde archivo."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.goals = [Goal.from_dict(g) for g in data]
                print(f"🎯 GoalManager: {len(self.goals)} objetivos cargados")
            except Exception as e:
                print(f"⚠️ Error cargando objetivos: {e}")
                self.goals = []
        else:
            self.goals = []
    
    def _save(self):
        """Guarda objetivos a archivo."""
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump([g.to_dict() for g in self.goals], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error guardando objetivos: {e}")
    
    def add_goal(
        self,
        description: str,
        goal_type: GoalType,
        priority: GoalPriority = GoalPriority.MEDIUM,
        trigger_date: Optional[datetime] = None,
        trigger_keywords: Optional[List[str]] = None,
        context: Optional[Dict] = None
    ) -> Goal:
        """Añade un nuevo objetivo."""
        goal_id = f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.goals)}"
        
        goal = Goal(
            id=goal_id,
            description=description,
            goal_type=goal_type,
            priority=priority,
            trigger_date=trigger_date,
            trigger_keywords=trigger_keywords,
            context=context
        )
        
        self.goals.append(goal)
        self._save()
        
        print(f"🎯 Nuevo objetivo: {description}")
        return goal
    
    def complete_goal(self, goal_id: str) -> bool:
        """Marca un objetivo como completado."""
        for goal in self.goals:
            if goal.id == goal_id:
                goal.completed = True
                goal.completed_at = datetime.now()
                self._save()
                return True
        return False
    
    def get_active_goals(self) -> List[Goal]:
        """Devuelve objetivos activos (no completados)."""
        return [g for g in self.goals if not g.completed]
    
    def get_due_goals(self) -> List[Goal]:
        """Devuelve objetivos cuyo trigger_date ya pasó o es inminente (próximos 30 min)."""
        now = datetime.now()
        due = []
        
        for goal in self.goals:
            if goal.completed:
                continue
            if goal.trigger_date and goal.trigger_date <= now + timedelta(minutes=30):
                due.append(goal)
        
        # Ordenar por prioridad (mayor primero)
        due.sort(key=lambda g: g.priority.value, reverse=True)
        return due
    
    def get_triggered_goals(self, current_text: str = "") -> List[Goal]:
        """Devuelve objetivos que deben activarse ahora."""
        now = datetime.now()
        triggered = []
        
        for goal in self.goals:
            if goal.is_triggered(current_text, now):
                triggered.append(goal)
        
        # Ordenar por prioridad (mayor primero)
        triggered.sort(key=lambda g: g.priority.value, reverse=True)
        return triggered
    
    def detect_from_text(self, text: str, openai_client=None) -> List[Goal]:
        """
        Detecta y crea objetivos a partir del texto del usuario.
        Wrapper conveniente que devuelve lista de objetivos creados.
        """
        goal = self.extract_goal_from_text(text, openai_client)
        if goal:
            return [goal]
        return []
    
    def get_proactive_message(self) -> Optional[str]:
        """
        Genera un mensaje proactivo si hay objetivos pendientes.
        Llamar al inicio de cada sesión.
        """
        triggered = self.get_triggered_goals()
        
        if not triggered:
            return None
        
        messages = []
        for goal in triggered[:3]:  # Máximo 3 recordatorios
            if goal.goal_type == GoalType.REMINDER:
                messages.append(f"📌 Recordatorio: {goal.description}")
            elif goal.goal_type == GoalType.FOLLOW_UP:
                messages.append(f"💬 Seguimiento: {goal.description}")
            elif goal.goal_type == GoalType.TASK:
                messages.append(f"✅ Pendiente: {goal.description}")
        
        if messages:
            return "\n".join(messages)
        return None
    
    def extract_goal_from_text(self, text: str, openai_client=None) -> Optional[Goal]:
        """
        Analiza el texto para detectar si contiene un objetivo implícito.
        
        Ejemplos:
        - "Mañana tengo reunión a las 10" → Crear recordatorio
        - "Recuérdame llamar a María" → Crear recordatorio
        - "El viernes voy al médico" → Crear recordatorio
        """
        t = text.lower()
        
        # Patrones de recordatorio explícito
        reminder_patterns = [
            'recuérdame', 'recuerdame', 'no olvides', 'no te olvides',
            'acuérdate', 'acuerdate', 'avísame', 'avisame'
        ]
        
        # Patrones de eventos futuros
        future_patterns = [
            'mañana', 'pasado mañana', 'el lunes', 'el martes', 'el miércoles',
            'el jueves', 'el viernes', 'el sábado', 'el domingo',
            'la próxima semana', 'la semana que viene', 'este fin de semana'
        ]
        
        # Patrones de citas/eventos
        event_patterns = [
            'tengo', 'voy a', 'voy al', 'tengo que', 'debo', 'necesito',
            'reunión', 'cita', 'médico', 'dentista', 'trabajo'
        ]
        
        # Detectar recordatorio explícito
        is_explicit_reminder = any(p in t for p in reminder_patterns)
        
        # Detectar evento futuro
        has_future_ref = any(p in t for p in future_patterns)
        has_event = any(p in t for p in event_patterns)
        
        if is_explicit_reminder or (has_future_ref and has_event):
            # Calcular fecha aproximada
            trigger_date = self._parse_date_from_text(t)
            
            # Crear objetivo
            return self.add_goal(
                description=text,
                goal_type=GoalType.REMINDER,
                priority=GoalPriority.HIGH if is_explicit_reminder else GoalPriority.MEDIUM,
                trigger_date=trigger_date,
                context={"original_text": text}
            )
        
        return None
    
    def _parse_date_from_text(self, text: str) -> datetime:
        """Extrae una fecha aproximada del texto."""
        now = datetime.now()
        
        if 'mañana' in text:
            return now + timedelta(days=1)
        elif 'pasado mañana' in text:
            return now + timedelta(days=2)
        elif 'próxima semana' in text or 'semana que viene' in text:
            return now + timedelta(days=7)
        
        # Días de la semana
        days = {
            'lunes': 0, 'martes': 1, 'miércoles': 2, 'miercoles': 2,
            'jueves': 3, 'viernes': 4, 'sábado': 5, 'sabado': 5, 'domingo': 6
        }
        
        for day_name, day_num in days.items():
            if day_name in text:
                current_day = now.weekday()
                days_ahead = day_num - current_day
                if days_ahead <= 0:  # Si ya pasó esta semana, ir a la próxima
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
        
        # Por defecto, mañana
        return now + timedelta(days=1)
    
    def create_follow_up(self, original_event: str, follow_up_delay_hours: int = 24) -> Goal:
        """
        Crea un objetivo de seguimiento después de un evento.
        
        Ej: Después de "reunión importante", preguntar "¿Cómo fue la reunión?"
        """
        trigger_date = datetime.now() + timedelta(hours=follow_up_delay_hours)
        
        return self.add_goal(
            description=f"Preguntar cómo fue: {original_event}",
            goal_type=GoalType.FOLLOW_UP,
            priority=GoalPriority.LOW,
            trigger_date=trigger_date,
            context={"original_event": original_event}
        )
    
    def get_summary(self) -> Dict:
        """Resumen de objetivos."""
        active = self.get_active_goals()
        return {
            "total": len(self.goals),
            "active": len(active),
            "completed": len(self.goals) - len(active),
            "by_type": {
                t.value: len([g for g in active if g.goal_type == t])
                for t in GoalType
            },
            "triggered_now": len(self.get_triggered_goals())
        }


# --- FUNCIONES DE UTILIDAD ---

def detect_goal_intent(text: str) -> Optional[Dict]:
    """
    Detecta si el texto contiene intención de crear un objetivo.
    Devuelve información estructurada o None.
    """
    t = text.lower()
    
    result = {
        "has_goal": False,
        "type": None,
        "urgency": "medium",
        "temporal_ref": None
    }
    
    # Detectar referencias temporales
    if any(x in t for x in ['mañana', 'pasado mañana']):
        result["temporal_ref"] = "tomorrow"
    elif any(x in t for x in ['hoy', 'esta tarde', 'esta noche']):
        result["temporal_ref"] = "today"
    elif any(x in t for x in ['semana', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes']):
        result["temporal_ref"] = "this_week"
    
    # Detectar tipo de objetivo
    if any(x in t for x in ['recuérdame', 'no olvides', 'avísame']):
        result["has_goal"] = True
        result["type"] = "reminder"
        result["urgency"] = "high"
    elif any(x in t for x in ['tengo', 'voy a', 'debo']) and result["temporal_ref"]:
        result["has_goal"] = True
        result["type"] = "event"
    
    return result if result["has_goal"] else None


if __name__ == "__main__":
    # Test básico
    gm = GoalManager("test_goals.json")
    
    # Crear objetivo de prueba
    gm.add_goal(
        description="Recordar reunión de trabajo",
        goal_type=GoalType.REMINDER,
        priority=GoalPriority.HIGH,
        trigger_date=datetime.now() + timedelta(hours=1)
    )
    
    # Ver resumen
    print(gm.get_summary())
    
    # Ver mensaje proactivo
    msg = gm.get_proactive_message()
    if msg:
        print(f"\n📢 Mensaje proactivo:\n{msg}")
