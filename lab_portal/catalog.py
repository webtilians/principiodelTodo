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
