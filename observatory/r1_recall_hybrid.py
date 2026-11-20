"""R1: does fusing a lexical ranking with the meaning ranker improve MCP recall? See
docs/INFINITO_R1_RECALL_HYBRID_PROTOCOL_2026-11-18.md, written before this code; nothing here may change an arm,
a parameter or the data after a result is seen.

    python -m observatory.r1_recall_hybrid --split dev --model <folder> [--sizes 200 2000] [--out file.json]

Synthetic Spanish memories, built through AgentMemory.remember (the real write path). No model is called and no
real person's memory is read.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

K_SHOWN = (5, 20)
CANDIDATES = 40
K_RRF = 60

SEEDS = {"dev": 18_11_2026, "confirm": 81_12_6202}
CODE_RANGE = {"dev": (1000, 4999), "confirm": (5000, 8999)}  # distractors use 9000-9999
COUNTS = {"F1": 40, "F2": 36, "F3": 20, "F4": 30}

PROJECTS = ["Atlas", "Brisa", "Cobalto", "Duna", "Ecotono", "Faro", "Gema", "Halcón", "Iris", "Jade", "Kora",
            "Lince", "Marea", "Nimbo", "Ópalo", "Pino", "Quásar", "Roble", "Sauce", "Trébol"]
CLIENTS = ["Acme", "Bravo Logística", "Cervecera Sur", "Clínica Rivas", "Estudio Lumen", "Ferretería Gil",
           "Granja Olmo", "Hotel Mirador", "Imprenta Vera", "Joyería Sol", "Talleres Nieto", "Panadería Luz"]
PEOPLE = ["Ana", "Luis", "Marta", "Pablo", "Lucía", "Jorge", "Elena", "Raúl", "Sara", "Diego", "Nuria", "Iván",
          "Carmen", "Óscar", "Irene", "Tomás"]
MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
          "noviembre", "diciembre"]
WEEKDAYS = ["lunes", "martes", "miércoles", "jueves", "viernes"]

TICKET_FACTS = ["se arregló el cierre de sesión", "falla la exportación a PDF", "el buscador tarda demasiado",
                "hay que traducir el menú al inglés", "las notificaciones llegan duplicadas",
                "el formulario de alta no valida el correo", "la página de precios se ve mal en el móvil",
                "el informe mensual sale vacío", "las fotos de perfil no se suben", "el carrito pierde productos",
                "el calendario muestra mal las zonas horarias", "la búsqueda ignora las tildes"]
PR_CHANGES = ["cambió el motor de plantillas", "añadió el modo oscuro", "quitó una dependencia antigua",
              "aceleró la carga de la portada", "separó la configuración por entornos",
              "corrigió el cálculo de impuestos", "añadió copias de seguridad diarias", "migró los tests a pytest",
              "limitó los intentos de acceso", "reorganizó el menú lateral"]

TOPICS = {
    "presupuesto": ["subir el presupuesto a {n}000 €", "congelar el presupuesto hasta {mes}"],
    "plazo": ["entregar la primera versión en {mes}", "retrasar la entrega hasta {mes}"],
    "diseño": ["usar tonos verdes en el logotipo", "rehacer la portada de la web"],
    "contrato": ["firmar un contrato de {n} meses", "renovar el contrato en {mes}"],
    "facturación": ["facturar cada {n}0 días", "cobrar la mitad por adelantado"],
    "reuniones": ["vernos los {dia}", "hacer una llamada cada {n} semanas"],
}

# F4: (attribute as written, values)
ATTRIBUTES = {
    "proveedor del servidor": ["Hetzner", "OVH", "AWS", "Azure", "DigitalOcean", "Scaleway"],
    "motor de base de datos": ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "MariaDB"],
    "responsable técnico": PEOPLE,
    "framework del panel": ["React", "Vue", "Svelte", "Angular", "SolidJS"],
    "mes de lanzamiento": MONTHS,
}

# F3: (memory, question) that share no content word. Even indexes go to DEV, odd to CONFIRM.
PARAPHRASES = [
    ("Mi perra se llama {person}.", "¿Tengo mascota?"),
    ("Mis canciones favoritas son las de {band}.", "¿Qué música me gusta?"),
    ("Conduzco una {car} de {year}.", "¿Qué vehículo tengo?"),
    ("Soy alérgico a {food}.", "¿Hay algún alimento que no pueda comer?"),
    ("Vivo en {city} desde hace {n} años.", "¿Dónde resido?"),
    ("Trabajo como {job} en una empresa pequeña.", "¿A qué me dedico?"),
    ("Mi hermana {person} cumple años el {n} de {mes}.", "¿Qué fecha debo felicitar a alguien de la familia?"),
    ("Los martes voy a nadar a la piscina municipal.", "¿Qué deporte practico?"),
    ("Me operaron de la rodilla en {year}.", "¿He tenido alguna intervención quirúrgica?"),
    ("Prefiero el té verde al café.", "¿Qué bebida caliente me gusta?"),
    ("Mi portátil es un {laptop}.", "¿Qué ordenador uso?"),
    ("Estudié Ingeniería Informática en {city}.", "¿Qué carrera hice?"),
    ("Mi pareja se llama {person}.", "¿Con quién comparto mi vida?"),
    ("Me despierto a las {n} cada mañana.", "¿A qué hora suelo levantarme?"),
    ("No soporto las películas de terror.", "¿Qué género de cine evito?"),
    ("Pago {n}00 € de alquiler al mes.", "¿Cuánto me cuesta la vivienda?"),
    ("Mi médica de cabecera es la doctora {person}.", "¿Quién lleva mi salud?"),
    ("Hablo inglés y un poco de alemán.", "¿Qué idiomas domino?"),
    ("Juego al pádel con {person} los {dia}.", "¿Con quién hago deporte?"),
    ("Mi banco es {bank}.", "¿Con qué entidad financiera opero?"),
    ("Uso gafas para leer.", "¿Tengo problemas de vista?"),
    ("Mi abuelo era pescador en {city}.", "¿A qué se dedicaba mi familia antes?"),
    ("Me encanta cocinar paella los domingos.", "¿Qué plato preparo el fin de semana?"),
    ("Tengo dos hijos, {person} y {person2}.", "¿Cuántos niños hay en casa?"),
    ("Mi equipo de fútbol es el {team}.", "¿A qué club animo?"),
    ("Leo novelas de ciencia ficción antes de dormir.", "¿Qué libros me gustan?"),
    ("Me mareo en los barcos.", "¿Qué medio de transporte me sienta mal?"),
    ("Hago yoga para la espalda.", "¿Qué ejercicio me ayuda con el dolor lumbar?"),
    ("Mi jefe se llama {person}.", "¿Quién es mi responsable en el trabajo?"),
    ("Ahorro para comprar una casa en el campo.", "¿Cuál es mi objetivo financiero?"),
    ("De pequeño viví en {city}.", "¿Dónde pasé mi infancia?"),
    ("Mi color preferido es el {color}.", "¿Qué tono me gusta más?"),
    ("Odio madrugar los fines de semana.", "¿Me gusta levantarme temprano el sábado?"),
    ("Me tatué una brújula en el brazo.", "¿Tengo algún dibujo en la piel?"),
    ("Soy vegetariano desde {year}.", "¿Como carne?"),
    ("Mi gato duerme siempre en el sofá.", "¿Tengo animales en casa?"),
    ("Cada verano viajo a {city} con amigos.", "¿Adónde voy de vacaciones?"),
    ("Toco la guitarra en un grupo.", "¿Qué instrumento musical sé tocar?"),
    ("Mi móvil es un {phone}.", "¿Qué teléfono tengo?"),
    ("Tomo pastillas para la tensión.", "¿Sigo algún tratamiento médico?"),
]
SLOTS = {
    "band": ["Vetusta Morla", "Love of Lesbian", "Coldplay", "Radiohead", "Izal"],
    "car": ["Renault Kangoo", "Seat Ibiza", "Toyota Corolla", "Peugeot Partner"],
    "food": ["los cacahuetes", "el marisco", "el kiwi", "la lactosa"],
    "city": ["Málaga", "Granada", "Valencia", "Bilbao", "Zaragoza", "Cádiz"],
    "job": ["fontanero", "enfermera", "diseñador gráfico", "contable", "electricista"],
    "laptop": ["ThinkPad X1", "MacBook Air", "Dell XPS 13", "Asus Zenbook"],
    "bank": ["Unicaja", "Sabadell", "BBVA", "ING"],
    "team": ["Málaga CF", "Betis", "Athletic", "Valencia CF"],
    "color": ["verde", "azul", "naranja", "morado"],
    "phone": ["Pixel 9", "iPhone 15", "Galaxy S24", "Xiaomi 14"],
}

# Distractors: everyday notes that reuse the same names and words. Codes 9000-9999, never a gold code.
DISTRACTORS = [
    "Hoy he trabajado en {project} revisando la documentación.",
    "{person} me pidió revisar el presupuesto de {client}.",
    "Tengo pendiente llamar a {client} por el {topic}.",
    "Con {client} hablamos del {topic} pero no acordamos nada.",
    "El ticket {prefix}-{code9} del proyecto {project} está cerrado.",
    "Mañana tengo reunión con {person} sobre {project}.",
    "En {project} hemos actualizado las dependencias.",
    "{person} se va de vacaciones en {mes}.",
    "He mandado la factura de {mes} a {client}.",
    "La demo de {project} salió bien delante de {client}.",
    "Revisé los tests de {project} con {person}.",
    "El PR #{code9} de {project} sigue en revisión.",
    "Hoy he salido a correr por {city}.",
    "Compré una lámpara nueva para el salón.",
    "{person} recomendó leer un libro sobre productividad.",
    "La reunión de los {dia} con {client} se movió a la tarde.",
    "Hay que preparar la presentación de {project} para {mes}.",
    "{client} pregunta si podemos adelantar el {topic}.",
    "El servidor de pruebas de {project} se reinició sin avisar.",
    "Anoté ideas para mejorar el diseño de {project}.",
]


@dataclass
class Question:
    family: str
    text: str
    gold: str  # the memory that answers it (F4: the later one)
    older: Optional[str] = None  # F4: the earlier value


@dataclass
class Item:
    text: str
    day: str


def _fill(template: str, rng: random.Random) -> str:
    people = rng.sample(PEOPLE, 2)
    values = {"person": people[0], "person2": people[1], "project": rng.choice(PROJECTS),
              "client": rng.choice(CLIENTS), "mes": rng.choice(MONTHS), "dia": rng.choice(WEEKDAYS),
              "n": str(rng.randint(2, 9)), "year": str(rng.randint(2008, 2024)), "topic": rng.choice(list(TOPICS)),
              "prefix": rng.choice(PROJECTS)[:3].upper().replace("Ó", "O").replace("Á", "A"),
              "code9": str(rng.randint(9000, 9999))}
    values.update({slot: rng.choice(options) for slot, options in SLOTS.items()})
    return template.format(**values)


def build(split: str):
    """The split's questions and its gold memories (with the later F4 memory dated after the earlier one)."""
    rng = random.Random(SEEDS[split])
    low, high = CODE_RANGE[split]
    questions: List[Question] = []
    golds: List[Item] = []
    start = date(2025, 1, 1)

    def day(lo: int = 0, hi: int = 600) -> str:
        return (start + timedelta(days=rng.randint(lo, hi))).isoformat()

    codes = rng.sample(range(low, high + 1), COUNTS["F1"])
    for i, code in enumerate(codes):
        project = rng.choice(PROJECTS)
        prefix = project[:3].upper().replace("Ó", "O").replace("Á", "A")
        if i % 2 == 0:
            memory = f"El ticket {prefix}-{code} del proyecto {project}: {rng.choice(TICKET_FACTS)}."
            question = f"¿Qué pasa con el ticket {prefix}-{code}?"
        else:
            memory = f"El PR #{code} de {project} {rng.choice(PR_CHANGES)}."
            question = f"¿Qué cambió el PR #{code}?"
        golds.append(Item(memory, day()))
        questions.append(Question("F1", question, memory))

    pairs = [(c, t) for c in CLIENTS for t in TOPICS]
    for client, topic in rng.sample(pairs, COUNTS["F2"]):
        decision = rng.choice(TOPICS[topic]).format(n=rng.randint(2, 9), mes=rng.choice(MONTHS),
                                                     dia=rng.choice(WEEKDAYS))
        memory = f"Con {client} acordamos {decision}."
        golds.append(Item(memory, day()))
        questions.append(Question("F2", f"¿Qué acordamos con {client} sobre el {topic}?", memory))

    offset = 0 if split == "dev" else 1
    for template, question in PARAPHRASES[offset::2][:COUNTS["F3"]]:
        memory = _fill(template, rng)
        golds.append(Item(memory, day()))
        questions.append(Question("F3", question, memory))

    combos = sorted((p, a) for p in PROJECTS for a in ATTRIBUTES)
    random.Random(4242).shuffle(combos)  # one fixed order, then halves: DEV and CONFIRM never share a pair
    half = combos[: len(combos) // 2] if split == "dev" else combos[len(combos) // 2:]
    for project, attribute in rng.sample(half, COUNTS["F4"]):
        before, after = rng.sample(ATTRIBUTES[attribute], 2)
        older = f"El {attribute} de {project} es {before}."
        later = f"Cambio en {project}: el {attribute} ahora es {after}."
        first = rng.randint(0, 400)
        golds.append(Item(older, (start + timedelta(days=first)).isoformat()))
        golds.append(Item(later, (start + timedelta(days=first + rng.randint(20, 150))).isoformat()))
        questions.append(Question("F4", f"¿Cuál es el {attribute} de {project}?", later, older))
    return questions, golds


def distractors(split: str, count: int) -> List[Item]:
    rng = random.Random(SEEDS[split] + 7)
    seen, out = set(), []
    while len(out) < count:
        text = _fill(rng.choice(DISTRACTORS), rng)
        if text in seen:
            continue
        seen.add(text)
        out.append(Item(text, (date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))).isoformat()))
    return out


# --- arms -------------------------------------------------------------------------------------------------------

def _rrf(*rankings: Sequence[int], tiebreak: Sequence[int]) -> List[int]:
    score: Dict[int, float] = {}
    for ranking in rankings:
        for position, index in enumerate(ranking[:CANDIDATES]):
            score[index] = score.get(index, 0.0) + 1.0 / (K_RRF + position + 1)
    order = {index: position for position, index in enumerate(tiebreak)}
    return sorted(score, key=lambda i: (-score[i], order.get(i, len(order)), i))


class Fts:
    def __init__(self, texts: Sequence[str]):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE VIRTUAL TABLE m USING fts5(text, tokenize='unicode61 remove_diacritics 2')")
        self.conn.executemany("INSERT INTO m(rowid, text) VALUES (?, ?)", list(enumerate(texts)))

    def rank(self, query: str, limit: int) -> List[int]:
        words = list(dict.fromkeys(re.findall(r"\w+", query.lower())))
        if not words:
            return []
        match = " OR ".join('"' + w.replace('"', '') + '"' for w in words)
        rows = self.conn.execute("SELECT rowid FROM m WHERE m MATCH ? ORDER BY bm25(m) LIMIT ?", (match, limit))
        return [int(r[0]) for r in rows]


def arms(agent, ranker, texts: List[str], ids: List[str]) -> Dict[str, Callable[[str], List[int]]]:
    position = {memory_id: i for i, memory_id in enumerate(ids)}
    fts = Fts(texts)

    def meaning(q):
        return ranker.rank(q, texts)

    def store(q):
        found = agent.store.search(q, top_k=max(CANDIDATES, 100), record_access=False)
        return [position[r.id] for r in found if r.id in position]

    return {
        "A": meaning,
        "B": store,
        "C": lambda q: fts.rank(q, CANDIDATES),
        "D": lambda q: (lambda a: _rrf(a, fts.rank(q, CANDIDATES), tiebreak=a))(meaning(q)),
        "E": lambda q: (lambda a: _rrf(a, store(q), tiebreak=a))(meaning(q)),
    }


# --- run --------------------------------------------------------------------------------------------------------

def make_memory(folder: Path, items: Sequence[Item]):
    from src.infinito3.mcp_server import AgentMemory

    agent = AgentMemory(str(folder / "memory.db"))
    refused = []
    for item in items:
        answer = agent.remember(item.text, kind="fact", source="R1 synthetic", date=item.day)
        if not answer.startswith("Remembered"):
            refused.append((item.text, answer[:80]))
    return agent, refused


def evaluate(questions: Sequence[Question], ranked: Dict[str, Callable[[str], List[int]]], texts: List[str],
             ranker=None):
    index_of = {}
    for i, text in enumerate(texts):
        index_of.setdefault(text, i)
    results = {}
    for name, rank in ranked.items():
        rank(questions[0].text)  # warm: loads the model and the vectors of every memory
        if ranker is not None:  # every arm pays for its questions' vectors, as a new question would in production
            for q in questions:
                ranker._cache.pop(q.text, None)
        rows, times = [], []
        for q in questions:
            began = time.perf_counter()
            order = rank(q.text)
            times.append((time.perf_counter() - began) * 1000)
            gold = index_of[q.gold]
            where = order.index(gold) + 1 if gold in order else None
            above = None
            if q.older is not None:
                older = index_of.get(q.older)
                older_at = order.index(older) + 1 if older is not None and older in order else None
                above = where is not None and (older_at is None or where < older_at)
            rows.append({"family": q.family, "rank": where, "later_above_older": above})
        results[name] = {"rows": rows, "ms": times}
    return results


def summarise(results) -> dict:
    def block(rows):
        n = len(rows)
        out = {f"recall@{k}": round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 4) for k in K_SHOWN}
        out["mrr@10"] = round(sum(1 / r["rank"] for r in rows if r["rank"] and r["rank"] <= 10) / n, 4)
        order = [r["later_above_older"] for r in rows if r["later_above_older"] is not None]
        if order:
            out["later_above_older"] = round(sum(order) / len(order), 4)
        out["n"] = n
        return out

    summary = {}
    for name, data in results.items():
        rows = data["rows"]
        families = sorted({r["family"] for r in rows})
        ms = sorted(data["ms"])
        summary[name] = {"all": block(rows), **{f: block([r for r in rows if r["family"] == f]) for f in families},
                         "p50_ms": round(statistics.median(ms), 2),
                         "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 2)}
    return summary


def deletion_check() -> bool:
    """With secure-delete on, a deleted row's unique word must not remain in the database file's bytes."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "fts.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE VIRTUAL TABLE m USING fts5(text, tokenize='unicode61 remove_diacritics 2')")
        conn.execute("INSERT INTO m(m, rank) VALUES ('secure-delete', 1)")
        conn.executemany("INSERT INTO m(rowid, text) VALUES (?, ?)",
                         [(i, f"Nota corriente número {i} sobre el proyecto.") for i in range(300)])
        conn.execute("INSERT INTO m(rowid, text) VALUES (999, 'Recuerdo con la palabra zorrovolante única.')")
        conn.commit()
        conn.execute("DELETE FROM m WHERE rowid = 999")
        conn.execute("INSERT INTO m(m) VALUES ('optimize')")
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
        return b"zorrovolante" not in path.read_bytes()


def run(split: str, model: str, sizes: Sequence[int]) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.infinito3.semantic import ranker_from

    ranker = ranker_from(model)
    if ranker is None:
        raise SystemExit(f"No model in {model}")
    questions, golds = build(split)
    report = {"split": split, "questions": len(questions), "sizes": {}, "fts5_secure_delete": deletion_check(),
              "sqlite": sqlite3.sqlite_version}
    for size in sizes:
        with tempfile.TemporaryDirectory() as folder:
            began = time.perf_counter()
            items = golds + distractors(split, max(0, size - len(golds)))
            agent, refused = make_memory(Path(folder), items)
            memories = agent._active()
            texts, ids = [r.content for r in memories], [r.id for r in memories]
            missing = [q.gold for q in questions if q.gold not in set(texts)]
            built = time.perf_counter() - began
            results = evaluate([q for q in questions if q.gold in set(texts)], arms(agent, ranker, texts, ids), texts,
                               ranker)
            report["sizes"][str(size)] = {"active": len(texts), "refused": refused, "gold_missing": missing,
                                          "build_s": round(built, 1), "summary": summarise(results),
                                          "rows": {a: r["rows"] for a, r in results.items()}}
            agent.close()
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SEEDS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 2000])
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report = run(args.split, args.model, args.sizes)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    for size, data in report["sizes"].items():
        print(f"N={size} active={data['active']} refused={len(data['refused'])} "
              f"gold_missing={len(data['gold_missing'])} build={data['build_s']}s")
        for arm, s in data["summary"].items():
            fam = " ".join(f"{f}:{s[f]['recall@20']:.2f}" for f in ("F1", "F2", "F3", "F4") if f in s)
            print(f"  {arm} R@5={s['all']['recall@5']:.3f} R@20={s['all']['recall@20']:.3f} "
                  f"MRR={s['all']['mrr@10']:.3f} p95={s['p95_ms']}ms | {fam}")
    print("fts5 secure-delete:", report["fts5_secure_delete"])


if __name__ == "__main__":
    main()
