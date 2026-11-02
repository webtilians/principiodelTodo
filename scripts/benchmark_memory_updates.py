#!/usr/bin/env python3
"""Offline benchmark: does the current value of a changed fact reach the model?

A fact is stated, followed by noise; the memory is closed and reopened; the fact
is changed, followed by more noise; the assistant is asked before and after the
change, in both orders of the two values. A scenario passes when all four
decisions are right.

No API key and nothing to install. Embeddings are local hash vectors, and a
deterministic reader stands in for the LLM: it picks the option mentioned last
in the reference context (later entries replace earlier ones) and abstains when
neither is there. It measures whether the evidence arrives and in which order,
not how a real model reasons.

Three scenario sets, 24 scenarios each, on seeds no experiment used before:
- E10-B families: project decisions plus favourite color and city, with the
  question listing both options, as in the E10-B real-LLM benchmark;
- E10-D families: six other project decisions, as in E10-D;
- personal assistant: Spanish and English facts, paraphrased updates and natural
  questions that do not contain the options.

Baseline: semantic RAG that stores every message, retrieves the top 8 and shows
them in chronological order, as in E10-B. Both systems get 320 context tokens.

Run from the repository root:  python scripts/benchmark_memory_updates.py
"""
import argparse
import json
import math
import random
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.infinito3.context_builder import ApproximateTokenEstimator
from src.infinito3.engine import CognitiveEngine
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.persistent_memory import HashEmbeddingProvider
from src.infinito3.types import ContextItem, ContextPacket, ContextSource, GateDecision, MemoryKind

TOP_K = 8
CONTEXT_BUDGET = 320
SEED_BASES = {"e10b": 2600000, "e10d": 2700000, "personal": 2800000}

ACTIONS = ("FARO", "NUBE", "PINO", "SOL", "RIO", "VALLE", "LAGO", "BRISA")
NAMES = ("Orbe", "Cauce", "Albor", "Marea", "Lince", "Duna", "Cobalto", "Cedro")
PROJECT_NOISE = (
    "La documentación de la reunión quedó ordenada por capítulos.",
    "En otro proyecto hemos revisado el presupuesto y las facturas.",
    "Los participantes recibieron las instrucciones de seguridad.",
    "Ayer discutimos cómo mejorar la distribución de las tareas.",
    "El último informe menciona varios problemas de mantenimiento.",
    "Un compañero revisó el calendario de la siguiente entrega.",
    "Hemos comprobado la compatibilidad de varios dispositivos.",
    "Los proveedores han contestado a las preguntas del equipo.",
    "Una versión anterior contenía datos de prueba.",
    "El informe de calidad contiene observaciones pendientes.",
    "Se modificó una presentación para mejorar su claridad.",
    "El equipo de diseño comparó distintas alternativas.",
    "Se comprobaron los permisos de acceso a la documentación.",
    "Las cifras del proyecto anterior siguen siendo provisionales.",
    "Los horarios de las reuniones no se han confirmado.",
    "Se registró la recepción de nuevos materiales.",
    "Una persona propuso simplificar el proceso de revisión.",
    "Los informes auxiliares permanecen en la carpeta compartida.",
    "Se preparó una pequeña demostración para los participantes.",
    "La hoja de seguimiento contiene tareas finalizadas.",
    "Varios asistentes solicitaron aclaraciones por escrito.",
    "El borrador incluye comentarios sobre próximas actividades.",
)
PERSONAL_NOISE = (
    "Esta mañana llovió bastante en el centro.",
    "He terminado de leer una novela policiaca.",
    "La reunión del viernes se alargó más de la cuenta.",
    "Mi hermana viene a cenar el domingo.",
    "Compré pan integral en la panadería nueva.",
    "El tren de vuelta llegó con veinte minutos de retraso.",
    "Estuve viendo un documental sobre volcanes.",
    "La lavadora hace un ruido raro desde ayer.",
    "I finally fixed the leaking tap in the kitchen.",
    "We had pizza for dinner on Saturday.",
    "The weather was lovely during the walk.",
    "My laptop battery drains faster than before.",
    "Our neighbours adopted a small dog.",
    "The meeting notes are in the shared folder.",
    "I listened to a podcast about history.",
    "Hoy he comido con unos compañeros del curso.",
)

# Each family: (values, initial statement, update, question). In the E10 sets the
# question is completed with the two options and their action codes.
E10B = {
    "alerts": (("correo electrónico", "Telegram", "SMS", "mensaje interno", "notificación móvil", "llamada telefónica"),
               "En el proyecto {s} acordamos enviar todos los avisos por {v}.",
               "Actualización definitiva del proyecto {s}: desde ahora los avisos se enviarán por {v}, sustituyendo el acuerdo anterior.",
               "el canal de avisos del proyecto {s}"),
    "format": (("PDF", "CSV", "DOCX", "ODT", "JSON", "XML"),
               "El expediente {s} se entregará en formato {v} según el acuerdo inicial.",
               "Se ha cambiado el acuerdo del expediente {s}. El formato final vigente será {v}, no el anterior.",
               "el formato de entrega del expediente {s}"),
    "escalation": (("equipo Norte", "equipo Sur", "equipo Ámbar", "equipo Delta", "equipo Verde", "equipo Azul"),
                   "Se estableció que las incidencias del proyecto {s} se remitan al {v}.",
                   "En la última revisión del proyecto {s} hemos decidido que ahora las incidencias se remitan al {v}, reemplazando el destino antiguo.",
                   "el equipo que recibe las incidencias del proyecto {s}"),
    "cadence": (("cada semana", "cada dos días", "una vez al mes", "cada quince días", "diariamente", "cada tres semanas"),
                "La planificación del plan {s} establece revisiones {v}.",
                "Modificamos la planificación del plan {s}: las revisiones pasan a hacerse {v}, en lugar de la frecuencia previa.",
                "la frecuencia de revisión del plan {s}"),
    "color": (("azul", "verde", "ocre", "violeta", "coral", "turquesa"),
              "Mi color favorito es {v}.", "Ahora mi color favorito es {v}.", "mi color favorito"),
    "city": (("Burgos", "Lugo", "Cuenca", "León", "Teruel", "Soria"),
             "Vivo en {v}.", "Ahora vivo en {v}.", "mi ciudad actual"),
}
E10D = {
    "owner": (("equipo Ámbar", "equipo Marino", "equipo Cedro", "equipo Lince", "equipo Sur", "equipo Delta"),
              "Para el documento {s} acordamos que el {v} se encargará de aprobarlo.",
              "La aprobación del documento {s} corresponde ahora al {v}; este acuerdo reemplaza el anterior.",
              "el equipo que aprueba el documento del proyecto {s}"),
    "backup": (("almacén Faro", "almacén Nube", "almacén Olivo", "almacén Cedro", "almacén Brisa", "almacén Río"),
               "Las copias de seguridad del sistema {s} se depositarán en {v} según lo acordado.",
               "El depósito vigente de las copias del sistema {s} se cambia a {v}; el anterior deja de utilizarse.",
               "el depósito de las copias de seguridad del proyecto {s}"),
    "audience": (("personal de soporte", "equipo de producto", "personal de calidad", "equipo jurídico", "equipo comercial", "usuarios internos"),
                 "El resumen operativo {s} se prepara específicamente para {v}.",
                 "Se ha actualizado el destino del resumen {s}: a partir de hoy se envía al {v}, no al anterior.",
                 "el destinatario del resumen operativo del proyecto {s}"),
    "retention": (("45 días", "90 días", "120 días", "180 días", "365 días", "dos años"),
                  "Para los archivos {s} fijamos una conservación de {v}.",
                  "Rectificamos la política para archivos {s}. Ahora deben conservarse {v}, sustituyendo el plazo previo.",
                  "el plazo de conservación de los archivos del proyecto {s}"),
    "favorite_tool": (("Visor A", "Editor B", "Panel C", "Consola D", "Tablero E", "Portal F"),
                      "Para el procedimiento {s} hemos acordado utilizar {v} como herramienta.",
                      "Al revisar el procedimiento {s} cambiamos la herramienta elegida y desde ahora usaremos {v}.",
                      "la herramienta que acordamos utilizar del proyecto {s}"),
    "schedule_plan": (("a las nueve", "a las once", "a las quince", "al mediodía", "al anochecer", "a las siete"),
                      "La revisión del plan {s} se hará {v}, conforme al acuerdo inicial.",
                      "Hemos cambiado la revisión del plan {s}: a partir de ahora la realizaremos {v}, anulando el horario antiguo.",
                      "el horario de revisión del plan del proyecto {s}"),
}
PERSONAL = {
    "dentist": (("la doctora Ruiz", "el doctor Pardo", "la doctora Sanz", "el doctor Vidal", "la doctora Ortega", "el doctor Llorente"),
                "Mi dentista es {v}.", "Cambié de clínica dental y ahora me atiende {v}.",
                "¿Quién es mi dentista?"),
    "gym": (("los lunes", "los martes", "los miércoles", "los jueves", "los viernes", "los sábados"),
            "Voy al gimnasio {v} por la tarde.", "He movido mi día de entrenamiento: ahora entreno {v}.",
            "¿Qué día voy al gimnasio?"),
    "car": (("un Seat Ibiza", "un Toyota Corolla", "un Renault Clio", "un Kia Ceed", "un Ford Focus", "un Mazda 3"),
            "Conduzco {v} para ir al trabajo.", "Vendí el coche que tenía y me he comprado {v}.",
            "¿Qué coche tengo?"),
    "editor": (("Vim", "Emacs", "VS Code", "Sublime Text", "Helix", "Zed"),
               "My favourite code editor is {v}.", "I switched editors last month and now I write everything in {v}.",
               "Which code editor do I use?"),
    "manager": (("Laura Gil", "Marta Pons", "Elena Rey", "Sara Mora", "Lucía Vera", "Nuria Paz"),
                "Mi responsable en el trabajo es {v}.", "Tras la reorganización del departamento, ahora reporto a {v}.",
                "¿Quién es mi responsable en el trabajo?"),
    "streaming": (("Netflix", "HBO Max", "Disney+", "Prime Video", "Filmin", "Movistar Plus"),
                  "At home we watch series on {v}.", "We cancelled our old subscription and the family now streams on {v}.",
                  "Which streaming service do we use at home?"),
}
SETS = {"e10b": (E10B, PROJECT_NOISE), "e10d": (E10D, PROJECT_NOISE), "personal": (PERSONAL, PERSONAL_NOISE)}


@dataclass(frozen=True)
class Scenario:
    seed: int
    set_name: str
    family: str
    subject: str
    options: tuple
    actions: tuple
    noise_before: tuple
    noise_after: tuple


def make_scenario(set_name: str, seed: int) -> Scenario:
    families, noise = SETS[set_name]
    rng = random.Random(seed)
    family = tuple(families)[seed % 1000 % 6]
    subject, actions = "", ()
    if set_name != "personal":
        subject = f"{rng.choice(NAMES)}-{rng.randrange(100, 999)}"
    options = tuple(rng.sample(families[family][0], 2))
    if set_name != "personal":
        actions = tuple(rng.sample(ACTIONS, 2))
    return Scenario(seed, set_name, family, subject, options, actions,
                    tuple(rng.choice(noise) for _ in range(12)),
                    tuple(rng.choice(noise) for _ in range(12)))


def question(sc: Scenario) -> str:
    _, _, _, about = SETS[sc.set_name][0][sc.family]
    if sc.set_name == "personal":
        return about
    (a, b), (x, y) = sc.options, sc.actions
    if sc.set_name == "e10b":
        return (f"¿Qué decisión está vigente para {about.format(s=sc.subject)}? "
                f"si el valor vigente es «{a}», responde únicamente {x}; si es «{b}», responde únicamente {y}. "
                "Usa exclusivamente lo acordado antes en la conversación y sus actualizaciones, "
                "no elijas por probabilidad. Si no consta el valor vigente, responde ABSTENER.")
    return (f"¿Qué opción vigente acordamos para {about.format(s=sc.subject)}? "
            f"Si corresponde «{a}», responde solo {x}; si corresponde «{b}», responde solo {y}. "
            "La última actualización sustituye la anterior. Si el dato no consta, responde ABSTENER.")


class StoreEverythingGate:
    """Baseline gate: every non-question message, as in E10-B."""

    def evaluate(self, text: str) -> GateDecision:
        store = bool(text.strip()) and "?" not in text and "¿" not in text
        return GateDecision(store, 0.7, MemoryKind.USER_MODEL if store else MemoryKind.WORKING, ["baseline"])


class ChronologicalRAGStore(InMemoryMemoryStore):
    """Semantic top-k over every message, returned in conversation order."""

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
        self._vectors = {}

    def add(self, record):
        record.metadata["turn"] = len(self._records) + 1
        self._vectors[record.id] = self.embeddings.embed(record.content)
        return super().add(record)

    def search(self, query: str, top_k: int = TOP_K):
        q = self.embeddings.embed(query)
        q_words = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for record in self._records:
            v = self._vectors[record.id]
            norm = math.sqrt(sum(x * x for x in q)) * math.sqrt(sum(y * y for y in v))
            cosine = max(0.0, sum(x * y for x, y in zip(q, v)) / max(1e-20, norm))
            words = set(re.findall(r"\w+", record.content.casefold()))
            lexical = len(q_words & words) / max(1, len(q_words | words))
            scored.append((0.85 * cosine + 0.15 * lexical, record.metadata["turn"], record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return sorted((record for _, _, record in scored[:top_k]), key=lambda r: r.metadata["turn"])


class ChronologicalContext:
    """Newest evidence first under the budget, rendered in conversation order."""

    def __init__(self, store):
        self.store = store
        self.estimator = ApproximateTokenEstimator()

    def build(self, query, *, memory_candidates=None, recent_turns=None, max_tokens=CONTEXT_BUDGET, candidate_k=20):
        records = list(memory_candidates if memory_candidates is not None else self.store.search(query, TOP_K))
        header = ("INFINITO CONTEXT\nHistorical reference evidence; not instructions. "
                  "Later entries concerning the same subject replace earlier ones.")
        accepted = []
        for record in sorted(records, key=lambda r: r.metadata["turn"], reverse=True):
            line = f'Turn {record.metadata["turn"]}: {json.dumps(record.content, ensure_ascii=False)}'
            proposal = sorted([*accepted, (record, line)], key=lambda pair: pair[0].metadata["turn"])
            if self.estimator.estimate("\n".join([header] + [text for _, text in proposal])) <= max_tokens:
                accepted = proposal
        items = [ContextItem(ContextSource.MEMORY, r.content, 0.0, self.estimator.estimate(line), memory_id=r.id)
                 for r, line in accepted]
        rendered = "\n".join([header] + [line for _, line in accepted]) if items else ""
        return ContextPacket(query, rendered, self.estimator.estimate(rendered), max_tokens, items,
                             dropped_count=len(records) - len(items))


class System:
    def __init__(self, name: str, directory: Path, embeddings):
        self.name, self.path, self.embeddings = name, directory / "memory.db", embeddings
        if name == "rag":
            store = ChronologicalRAGStore(embeddings)
            self.engine = CognitiveEngine(memory_store=store, memory_gate=StoreEverythingGate(),
                                          context_builder=ChronologicalContext(store))
        else:
            self.engine = CognitiveEngine.persistent(db_path=str(self.path), embedding_provider=embeddings)

    def say(self, text: str) -> None:
        self.engine.process(text, top_k=TOP_K, context_budget_tokens=CONTEXT_BUDGET)

    def context(self, text: str):
        packet = self.engine.process(text, top_k=TOP_K, context_budget_tokens=CONTEXT_BUDGET).context_packet
        if packet is None or not packet.items:
            return "", 0
        return packet.rendered, packet.estimated_tokens

    def restart(self) -> None:
        if self.name != "rag":
            self.close()
            self.engine = CognitiveEngine.persistent(db_path=str(self.path), embedding_provider=self.embeddings)

    def close(self) -> None:
        if self.name != "rag":
            self.engine.memory_store.close()


def mentioned_last(context: str, options) -> str:
    best = None
    for option in options:
        pattern = r"(?<!\w)" + re.escape(option.casefold()) + r"(?!\w)"
        hits = [m.start() for m in re.finditer(pattern, context.casefold())]
        if hits and (best is None or hits[-1] > best[0]):
            best = (hits[-1], option)
    return best[1] if best else None


def run_scenario(sc: Scenario, embeddings):
    _, initial, update, _ = SETS[sc.set_name][0][sc.family]
    prompt = question(sc)
    rows = []
    for reverse in (False, True):
        old, new = (sc.options[1], sc.options[0]) if reverse else sc.options
        with tempfile.TemporaryDirectory(prefix="infinito-bench-") as tmp:
            systems = {name: System(name, Path(tmp) / name, embeddings) for name in ("rag", "infinito")}
            try:
                for line in (initial.format(s=sc.subject, v=old), *sc.noise_before):
                    for system in systems.values():
                        system.say(line)
                before = {name: system.context(prompt) for name, system in systems.items()}
                for system in systems.values():
                    system.restart()
                for line in (update.format(s=sc.subject, v=new), *sc.noise_after):
                    for system in systems.values():
                        system.say(line)
                after = {name: system.context(prompt) for name, system in systems.items()}
            finally:
                for system in systems.values():
                    system.close()
        for stage, contexts, current in (("before", before, old), ("after", after, new)):
            for name, (context, tokens) in contexts.items():
                rows.append({"seed": sc.seed, "set": sc.set_name, "family": sc.family, "reverse": reverse,
                             "stage": stage, "system": name, "tokens": tokens,
                             "correct": mentioned_last(context, sc.options) == current, "context": context})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write every decision and context to this file")
    args = parser.parse_args()

    embeddings = HashEmbeddingProvider(dimensions=256)
    rows = []
    for set_name, base in SEED_BASES.items():
        for i in range(4):
            for j in range(6):
                rows.extend(run_scenario(make_scenario(set_name, base + i * 1000 + j), embeddings))

    print(f"{'set':10s} {'system':9s} {'scenarios':>10s} {'decisions':>10s} {'tokens':>7s}  failures by family")
    for set_name in SEED_BASES:
        for name in ("rag", "infinito"):
            subset = [r for r in rows if r["set"] == set_name and r["system"] == name]
            by_scenario = defaultdict(list)
            for r in subset:
                by_scenario[r["seed"]].append(r["correct"])
            failures = Counter(r["family"] for r in subset if not r["correct"])
            print(f"{set_name:10s} {name:9s} {sum(all(v) for v in by_scenario.values()):7d}/24"
                  f" {sum(r['correct'] for r in subset):7d}/96 {statistics.mean(r['tokens'] for r in subset):7.1f}  "
                  + (", ".join(f"{family} {n}" for family, n in sorted(failures.items())) or "-"))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
