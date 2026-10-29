"""Explicit, allowlisted experiments. No shell, free-form command or file paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IntegerParam:
    label: str
    default: int
    minimum: int
    maximum: int
    step: int = 1
    description: str = ""


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    command: tuple[str, ...]
    params: dict[str, IntegerParam]
    unit: str
    scenario: str
    warning: str
    presets: dict[str, dict[str, int]]
    dependencies: tuple[str, ...] = ()


EXPERIMENTS = {
    "f4": Experiment(
        name="F4 · Memoria selectiva", description="Aprender relevancia cambiante sin conocer futuras consultas",
        command=("-m", "observatory.selective_memory"),
        params={"seeds": IntegerParam("Semillas independientes", 24, 4, 250)},
        unit="tasa de recuperación", scenario="demanda cambiante · control uniforme",
        warning="Control simbólico de frecuencias; no es memoria neuronal ni consciencia.",
        presets={"rapido": {"seeds": 4}, "estandar": {"seeds": 24}, "largo": {"seeds": 120}},
    ),
    "f5": Experiment(
        name="F5 · Memoria predictiva", description="Elegir recuerdos por sus consecuencias futuras estimadas",
        command=("-m", "observatory.predictive_memory"),
        params={
            "seeds": IntegerParam("Semillas", 20, 4, 100),
            "train": IntegerParam("Casos entrenamiento por semilla", 960, 96, 10000),
            "test": IntegerParam("Casos test por escenario", 240, 24, 6000),
        }, unit="error cuadrático (MSE)", scenario="estable · cambio · nulo · no lineal",
        warning="Modelo predictivo lineal y datos sintéticos. Menor MSE es mejor.",
        presets={"rapido": {"seeds": 4, "train": 160, "test": 48},
                 "estandar": {"seeds": 20, "train": 960, "test": 240},
                 "largo": {"seeds": 48, "train": 2000, "test": 650}},
        dependencies=("numpy",),
    ),
    "f6": Experiment(
        name="F6 · Modelo del mundo / RL", description="Aprender dinámicas y planificar acciones para regular necesidades",
        command=("scripts/f6_homeostatic_rl.py",),
        params={
            "seeds": IntegerParam("Semillas", 8, 2, 60),
            "train-episodes": IntegerParam("Episodios entrenamiento", 110, 10, 400),
            "test-episodes": IntegerParam("Episodios test", 12, 3, 40),
            "horizon": IntegerParam("Pasos por episodio", 48, 16, 100),
        }, unit="coste homeostático", scenario="mismo rango · reglas cambiadas",
        warning="F6 tiene limitaciones de emparejamiento de física; no usar como evidencia confirmatoria sin repararlas.",
        presets={"rapido": {"seeds": 2, "train-episodes": 12, "test-episodes": 3, "horizon": 16},
                 "estandar": {"seeds": 8, "train-episodes": 110, "test-episodes": 12, "horizon": 48},
                 "largo": {"seeds": 24, "train-episodes": 200, "test-episodes": 24, "horizon": 64}},
        dependencies=("numpy",),
    ),
    "f7": Experiment(
        name="F7 · Valor de información", description="Decidir si pagar por observar antes de actuar",
        command=("-m", "observatory.active_information_f7"),
        params={
            "seeds": IntegerParam("Semillas", 24, 4, 100),
            "train": IntegerParam("Observaciones calibración", 240, 40, 5000),
            "test": IntegerParam("Decisiones de evaluación", 600, 100, 5000, step=2),
        }, unit="recompensa neta", scenario="estable · cambio de sensor · ruido",
        warning="Valor de información programado; no equivale a curiosidad subjetiva.",
        presets={"rapido": {"seeds": 4, "train": 80, "test": 120},
                 "estandar": {"seeds": 24, "train": 240, "test": 600},
                 "largo": {"seeds": 80, "train": 900, "test": 2200}},
    ),
    "f7b": Experiment(
        name="F7b · Recalibración", description="Detectar predicciones fallidas y reexaminar fuentes de información",
        command=("-m", "observatory.active_recalibration_f7b"),
        params={
            "seeds": IntegerParam("Semillas", 24, 4, 100),
            "train": IntegerParam("Observaciones calibración", 240, 40, 5000),
            "test": IntegerParam("Decisiones de evaluación", 600, 100, 5000, step=2),
        }, unit="recompensa neta", scenario="estable · cambio de sensor · ruido",
        warning="F7b es exploratorio: detector y umbrales se diseñaron después de observar F7.",
        presets={"rapido": {"seeds": 4, "train": 80, "test": 120},
                 "estandar": {"seeds": 24, "train": 240, "test": 600},
                 "largo": {"seeds": 80, "train": 900, "test": 2200}},
    ),
    "f8": Experiment(
        name="F8 · Mundo activo integrado",
        description="Predecir consecuencias, detectar cambios y decidir cuándo observar",
        command=("-m","observatory.active_world_f8"),
        params={
            "seeds": IntegerParam("Semillas independientes",12,4,120),
            "train": IntegerParam("Transiciones de calibración por mundo",1800,300,15000),
            "test-episodes": IntegerParam("Episodios de prueba por escenario",10,3,40),
            "horizon": IntegerParam("Decisiones por episodio",42,18,80),
            "seed-offset": IntegerParam("Semilla inicial (para replicaciones)",1000,0,100000),
        },
        unit="recompensa neta por paso",
        scenario="estable · cambio oculto · sensores de ruido",
        warning=("Mundo sintético con hipótesis de dos regímenes predefinida. "
                 "No equivale a curiosidad consciente; comprobar controles y CI."),
        presets={
            "rapido":{"seeds":4,"train":320,"test-episodes":3,"horizon":18,"seed-offset":1000},
            "estandar":{"seeds":12,"train":1800,"test-episodes":10,"horizon":42,"seed-offset":1000},
            "largo":{"seeds":48,"train":5000,"test-episodes":24,"horizon":60,"seed-offset":10000},
        },
    ),
    "f9_memory": Experiment(
        name="F9-A · MiniGrid Memory",
        description="Benchmark externo: PPO vs LSTM real, sin módulos INFINITO todavía",
        command=("-m","observatory.f9_external_minigrid","--phase","development","--task","memory"),
        params={
            "seeds": IntegerParam("Semillas de entrenamiento",1,1,5),
            "timesteps": IntegerParam("Pasos por política y semilla",128,128,200000),
            "eval-episodes": IntegerParam("Episodios de validación",3,1,500),
        },
        unit="tasa de éxito", scenario="MiniGrid Memory · validación de desarrollo",
        warning=("F9-A son baselines externos PPO/LSTM, no INFINITO. "
                 "Requiere LAB_F9_ENABLED=1 y entorno Python F9 separado con memoria suficiente. "
                 "No abrir test retenido al ajustar hiperparámetros."),
        presets={"rapido":{"seeds":1,"timesteps":128,"eval-episodes":3},
                 "estandar":{"seeds":3,"timesteps":20000,"eval-episodes":50},
                 "largo":{"seeds":5,"timesteps":100000,"eval-episodes":200}},
    ),
    "f9_doorkey": Experiment(
        name="F9-A · MiniGrid DoorKey",
        description="Benchmark externo de planificación llave, puerta y meta",
        command=("-m","observatory.f9_external_minigrid","--phase","development","--task","doorkey"),
        params={
            "seeds": IntegerParam("Semillas de entrenamiento",1,1,5),
            "timesteps": IntegerParam("Pasos por política y semilla",128,128,200000),
            "eval-episodes": IntegerParam("Episodios de validación",3,1,500),
        },
        unit="tasa de éxito", scenario="MiniGrid DoorKey · validación de desarrollo",
        warning="F9-A: solo PPO/LSTM externos. Necesita entorno F9 y memoria suficiente; no demuestra mejoras de INFINITO.",
        presets={"rapido":{"seeds":1,"timesteps":128,"eval-episodes":3},
                 "estandar":{"seeds":3,"timesteps":20000,"eval-episodes":50},
                 "largo":{"seeds":5,"timesteps":100000,"eval-episodes":200}},
    ),
    "f9_dynamic": Experiment(
        name="F9-A · MiniGrid Obstáculos",
        description="Benchmark externo: navegación ante obstáculos móviles",
        command=("-m","observatory.f9_external_minigrid","--phase","development","--task","dynamic"),
        params={
            "seeds": IntegerParam("Semillas de entrenamiento",1,1,5),
            "timesteps": IntegerParam("Pasos por política y semilla",128,128,200000),
            "eval-episodes": IntegerParam("Episodios de validación",3,1,500),
        },
        unit="tasa de éxito", scenario="MiniGrid DynamicObstacles · desarrollo",
        warning="F9-A: solo PPO/LSTM externos. En servidor compartido se deshabilita hasta habilitación explícita.",
        presets={"rapido":{"seeds":1,"timesteps":128,"eval-episodes":3},
                 "estandar":{"seeds":3,"timesteps":20000,"eval-episodes":50},
                 "largo":{"seeds":5,"timesteps":100000,"eval-episodes":200}},
    ),
    "f9b_memory_baselines": Experiment(
        name="F9-B · Referencias MiniGrid Memory",
        description="Entrenar PPO y LSTM externos hasta establecer una referencia competente antes de probar F4",
        command=("-m", "observatory.f9b_causal_memory", "--phase", "baselines"),
        params={
            "seeds": IntegerParam("Semillas de entrenamiento", 1, 1, 5),
            "timesteps": IntegerParam("Pasos por modelo y semilla", 20000, 20000, 300000, step=20000),
            "eval-episodes": IntegerParam("Episodios de desarrollo por semilla", 50, 50, 500),
        },
        unit="tasa de éxito", scenario="MiniGrid Memory S7 · prueba de desarrollo",
        warning=("Solo entrena PPO y LSTM externos, NO INFINITO. "
                 "Umbral preregistrado: LSTM ≥70% y ventaja sobre random ≥15 pp en ≥4 semillas. "
                 "Test reservado cerrado; ejecución pesada en PC Windows."),
        presets={
            "rapido": {"seeds":1,"timesteps":20000,"eval-episodes":50},
            "estandar": {"seeds":4,"timesteps":100000,"eval-episodes":50},
            "largo": {"seeds":5,"timesteps":300000,"eval-episodes":100},
        },
    ),
}


class ParameterError(ValueError):
    pass


def normalize(experiment_id: str, raw: Any) -> dict[str, int]:
    if experiment_id not in EXPERIMENTS:
        raise ParameterError("Experimento no disponible")
    if not isinstance(raw, dict):
        raise ParameterError("Los parámetros deben ser un objeto")
    definition = EXPERIMENTS[experiment_id]
    unknown = set(raw) - set(definition.params)
    if unknown:
        raise ParameterError("Parámetros no permitidos: " + ", ".join(sorted(unknown)))
    result: dict[str, int] = {}
    for key, spec in definition.params.items():
        value = raw.get(key, spec.default)
        if type(value) is not int or not spec.minimum <= value <= spec.maximum:
            raise ParameterError(f"{key}: entero entre {spec.minimum} y {spec.maximum}")
        if (value - spec.minimum) % spec.step:
            raise ParameterError(f"{key}: debe aumentar en pasos de {spec.step}")
        result[key] = value
    return result


def command(experiment_id: str, params: dict[str, int], output: str) -> list[str]:
    exp = EXPERIMENTS[experiment_id]
    result = list(exp.command)
    for key, value in params.items():
        result += ["--" + key, str(value)]
    return result + ["--output", output]


def describe() -> list[dict]:
    return [dict(id=key, name=val.name, description=val.description,
                 params={name: vars(spec) for name, spec in val.params.items()},
                 presets=val.presets, scenario=val.scenario, warning=val.warning,
                 unit=val.unit, dependencies=val.dependencies)
            for key, val in EXPERIMENTS.items()]
