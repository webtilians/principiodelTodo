"""Display raw experimental reports without inventing comparability or success."""
from __future__ import annotations

import math
from typing import Any


def _finite(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return round(float(value), 5)
    return None


def summarize(experiment: str, report: dict[str, Any]) -> dict:
    if experiment == "f4":
        categories = {name: {key: _finite(val["heldout_hit"])
                             for key, val in section["strategies"].items()}
                      for name, section in report["conditions"].items()}
        return {"label": "Recuperación de recuerdos (más es mejor)",
                "unit": "%", "scenarios": categories, "scale": 100}
    if experiment == "f5":
        return {"label": "MSE de predicción (menos es mejor)", "unit": "MSE",
                "scale": 1, "scenarios": {
                    scenario: {name: _finite(metrics["mse"]) for name, metrics
                               in entry["policies"].items()}
                    for scenario, entry in report["conditions"].items()
                    if isinstance(entry, dict) and "policies" in entry}}
    if experiment == "f6":
        return {"label": "Coste por paso (menos es mejor)", "unit": "coste",
                "scale": 1, "scenarios": {
                    scenario: {name: _finite(entry["mean_cost_per_step"])
                               for name, entry in block["policies"].items()}
                    for scenario, block in ((key, report[key]) for key in
                                            ("in_distribution", "shifted_mechanisms"))}}
    if experiment == "f7":
        return {"label": "Recompensa neta (más es mejor)", "unit": "recompensa",
                "scale": 1, "scenarios": {
                    name: {key: _finite(val["all"]["reward"])
                           for key, val in sec["policies"].items()}
                    for name, sec in report["conditions"].items()}}
    if experiment == "f7b":
        return {"label": "Recompensa neta F7b (más es mejor)", "unit": "recompensa",
                "scale": 1, "scenarios": {
                    name: {"recalibration": _finite(sec["mean_reward"]["all"]),
                           "late": _finite(sec["mean_reward"]["after_last_100"])}
                    for name, sec in report["conditions"].items()}}
    if experiment == "f8":
        return {"label": "Recompensa neta por paso (más es mejor)",
                "unit": "recompensa", "scale": 1,
                "scenarios": {
                    scenario: {
                        name: _finite(metrics["return_mean"])
                        for name, metrics in section["policies"].items()
                    } for scenario, section in report["scenarios"].items()
                }}
    if experiment.startswith("f9_"):
        return {
            "label": "Éxito por episodio (más es mejor) · PPO / LSTM externos",
            "unit": "%", "scale": 100,
            "scenarios": {
                report["task"]: {
                    name: _finite(metric["success_rate"])
                    for name, metric in report["summary"]["policies"].items()
                }
            },
        }
    return {"label": "Resultados", "unit": "", "scale": 1, "scenarios": {}}
