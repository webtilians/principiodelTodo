"""The day's report for the person, in Spanish: what was seen, what was thought, what waits for them, what it cost
and what could not be looked at."""
from datetime import datetime

from .diary import proposals


def _ids(ids) -> str:
    return ", ".join(ids) if ids else "—"


def render(engine, run: str, now: datetime, inventory: dict, done: dict, *, nothing_new: bool = False) -> str:
    day = now.date()
    mode = ("real (modelos de pago)" if engine.live
            else "PRUEBA: no se llamó a ningún modelo, no se escribió nada en INFINITO, coste 0")
    lines = [f"# Motor cognitivo · revisión del {day:%d/%m/%Y}", "",
             f"Ejecución {run} · modo {mode}.", ""]
    if nothing_new:
        lines += ["No hay recuerdos nuevos desde la última revisión: no se llamó a ningún modelo.", ""]
        return "\n".join(lines)

    cover = done["coverage"]
    lines += ["## Cobertura", "",
              f"- Recuerdos nuevos leídos: {cover['inventory']} "
              f"(INFINITO informó de {inventory['reported_new']}).",
              "- Citados por: " + ", ".join(f"{role} {n}" for role, n in cover["cited_by"].items()) + ".",
              f"- Sin citar por nadie: {len(cover['cited_by_nobody'])}."]
    lines += [f"- Límite: {limit}." for limit in cover["limits"]] or ["- Sin límites: el supervisor tuvo el día entero."]
    lines.append("")

    if done.get("summary"):
        lines += ["## Lo que te dice el supervisor", "", done["summary"], ""]

    lines += ["## Su reflexión", ""]
    reflection = done.get("reflection")
    if reflection:
        lines += [reflection["reflection"], ""]
        if reflection.get("advice"):
            lines += [f"*Consejo que se da:* {reflection['advice']}", ""]
        for question in reflection.get("questions") or []:
            lines.append(f"- ¿{question.strip('¿?')}?")
        lines += ["", f"Se apoya en: {_ids(reflection['sources'])}."]
        if done.get("published"):
            lines.append(f"Publicada en INFINITO como reflexión {done['published']}: la ves, comentas, ocultas o "
                         "borras en tu página de memoria, en Reflexiones. Mañana leerá tus comentarios.")
        else:
            lines.append("No se publicó en INFINITO" + (" (modo prueba)." if not engine.live else "."))
    else:
        lines.append("Hoy no deja reflexión.")
    if done.get("dropped"):
        lines += ["", "El árbitro quitó de la reflexión: " + "; ".join(done["dropped"]) + "."]
    lines.append("")

    if done["audit"]:
        lines += ["## Auditoría de los hijos", ""]
        lines += [f"- **{a.get('role')}** · {a.get('issue')}: {a.get('note', '')} ({_ids(a['ids'])})"
                  for a in done["audit"]]
        lines.append("")

    lines += ["## Hipótesis del investigador y veredictos", ""]
    verdicts = {v.get("hypothesis"): v for v in done["verdicts"]}
    if not done["hypotheses"]:
        lines.append("Ninguna.")
    for hypothesis in done["hypotheses"]:
        verdict = verdicts.get(hypothesis.get("id"), {})
        lines += [f"- **{hypothesis.get('id')}** {hypothesis.get('statement', '')}",
                  f"  - La refutaría: {hypothesis.get('falsifier', '—')}",
                  f"  - Veredicto del supervisor: **{verdict.get('verdict', 'sin juzgar')}**"
                  + (f" — {verdict.get('why')}" if verdict.get("why") else ""),
                  f"  - Se apoya en: {_ids(hypothesis.get('evidence_ids'))}"]
    lines += ["", "Una hipótesis apoyada sigue siendo una hipótesis: no es un hecho hasta que la compruebes.", ""]

    aside = done["observations_aside"] + done["hypotheses_aside"] + done["aside"]
    if aside:
        lines += ["## Apartado por el árbitro (citaba algo que no se le mostró)", ""]
        lines += [f"- {a.get('note') or a.get('statement') or a.get('why') or a.get('id')} "
                  f"(ids desconocidos: {_ids(a.get('unknown_ids'))})" for a in aside]
        lines.append("")

    if done["blind"]:
        lines += ["## Para valorar a ciegas", "",
                  "Hipótesis de dos fuentes mezcladas (la jerarquía y una pasada barata sola); no se dice cuál es "
                  "cuál. Valora las que quieras: `python -m cognition rate <etiqueta> util` o `... no`.", ""]
        lines += [f"- **{b['label']}** {b['statement']}" for b in done["blind"]]
        lines.append("")

    waiting = [p for p in proposals(engine.diary) if p["status"] == "pending"]
    lines += ["## Pautas que esperan tu aprobación", ""]
    if not waiting:
        lines.append("Ninguna.")
    for p in waiting:
        lines += [f"- **{p['id']}** para el **{p['target_role']}**: {p['guideline']}",
                  f"  - Por qué: {p['rationale'] or '—'}", f"  - Cómo se comprobaría: {p['test'] or '—'}"]
    if waiting:
        lines += ["", "`python -m cognition approve <id>`, `reject <id>`; una aprobada se puede retirar con "
                      "`revoke <id>`. Ninguna se aplica sin ti."]
    lines.append("")

    calls = [e for e in engine.diary.entries("call") if e.get("run") == run]
    skipped = [e for e in engine.diary.entries("budget_skip") if e.get("run") == run]
    errors = [e for e in engine.diary.entries("error") if e.get("run") == run]
    lines += ["## Coste", "",
              f"Hoy {engine.ledger.spent(day):.3f} € de un tope de {engine.ledger.cap(day):.2f} € "
              "(precios que configuraste, redondeados al alza; el cambio de dólar a euro es aproximado).", ""]
    lines += [f"- {c['role']} ({c['model']}): {c['usage']['input']} tokens de entrada, {c['usage']['output']} de "
              f"salida, {c['usage']['thinking']} de razonamiento · {c['cost_eur']:.4f} €"
              + ("" if c["understood"] else " · respuesta no utilizable") for c in calls]
    lines += [f"- {s['role']}: no se llamó, el presupuesto no daba ({s['estimate_eur']} € estimados, "
              f"{s['remaining_eur']} € libres)" for s in skipped]
    lines += [f"- {e.get('role') or e.get('where')}: error {e['code']}" for e in errors]
    lines.append("")
    return "\n".join(lines)
