"""The day's review, and the hourly tick that decides whether one is due.

tick() is what the scheduled task runs every hour. It does nothing while the engine is paused or another run holds
the lock. Once the local clock passes review_hour, the day's review is due; a day the computer was off is caught up
the next time it runs, from where the last review left off, so nothing new is skipped. Between reviews, enough new
memories wake the observer for a pass of its own, which the supervisor then reads with the rest.

A review:
1. reads from INFINITO what is new since the last review (inventory), the engine's latest reflections with the
   person's comments on them, and the task queue;
2. the observer marks what matters; the investigator turns it into falsifiable hypotheses, each with a check that
   the engine runs as a read-only recall; the baseline makes one cheap pass of its own, kept blind;
3. the supervisor reads all of it, the whole inventory included, audits its children, judges the hypotheses against
   the checks, writes its reflection and proposes guidelines;
4. the referee (validate.py) sets aside every claim that cites something it was not shown;
5. the reflection goes to INFINITO's reflections (only in a live run); the proposals wait for the person;
6. a report for the person, and the diary, record all of it, costs included.
Without the person's authorization to spend (python -m cognition authorize), a review is a dry run: no model is
called (stand-ins answer), nothing is written to INFINITO, nothing is spent.
"""
import json
import os
import random
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import roles, validate
from .diary import Diary, approved_guidelines, proposals
from .infinito import InfinitoError
from .ledger import Ledger, UnknownPrice, cost_eur, estimate_eur
from .providers import Fake, ProviderError

LOCK_STALE_SECONDS = 3 * 3600
SUPERVISOR_EXTRA = 40000  # characters the children's reports, the checks and the instructions may add
REFLECTIONS_DAYS = 21
CHECK_LIMIT = 8
NEW_HEADER = re.compile(r"Memories saved [^\n]*?\((\d+)\)")
EARLIER = "\nYour latest earlier reflections"
KEPT_AS = re.compile(r"Kept as your reflection ([0-9a-f]{8})")


class Busy(Exception):
    pass


def _utc(moment: datetime) -> str:
    """How INFINITO's reflection_context reads a time: ISO, naive, in UTC."""
    return moment.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _parse_at(at: str) -> datetime:
    return datetime.fromisoformat(at)


class Engine:
    def __init__(self, folder: Path, config: dict, infinito, make_model: Callable, *, live: bool,
                 clock: Callable[[], datetime] = lambda: datetime.now().astimezone()):
        self.folder, self.config, self.infinito, self.make_model = folder, config, infinito, make_model
        self.live, self.clock = live, clock
        self.diary = Diary(folder, clock=lambda: self.clock().astimezone(timezone.utc).isoformat(timespec="seconds"))
        self.ledger = Ledger(folder, config)
        folder.mkdir(parents=True, exist_ok=True)

    # State

    def paused(self) -> bool:
        return (self.folder / "PAUSED").exists()

    @contextmanager
    def lock(self):
        path = self.folder / "run.lock"
        try:
            if time.time() - path.stat().st_mtime > LOCK_STALE_SECONDS:
                path.unlink()
        except FileNotFoundError:
            pass
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise Busy() from None
        try:
            os.write(handle, str(os.getpid()).encode())
            os.close(handle)
            yield
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def last_review(self) -> Optional[dict]:
        """The last review of this mode: a dry run never moves where a live review starts reading from."""
        return self.diary.last("review_end", live=self.live)

    def _observation_sets(self, since: datetime) -> List[dict]:
        return [self.diary.body(e) or {} for e in self.diary.entries("observation_set")
                if e.get("live") == self.live and e.get("pass_") == "threshold" and _parse_at(e["at"]) > since]

    def review_due(self, now: datetime) -> bool:
        last = self.last_review()
        if not last:
            return True
        last_day = _parse_at(last["at"]).astimezone(now.tzinfo).date()
        if last_day >= now.date():
            return False
        if last_day < now.date() - timedelta(days=1):
            return True  # a day was missed: catch up now
        return now.hour >= int(self.config["review_hour"])

    def tick(self) -> str:
        if self.paused():
            return "paused"
        try:
            with self.lock():
                now = self.clock()
                if self.live:
                    self.sync()
                    self.publish_unreported()
                if self.review_due(now):
                    return self.review()
                return self.watch()
        except Busy:
            return "another run is in progress"

    # The person's word, given on their memory page

    def sync(self) -> int:
        """Bring in what the person decided on their page about each guideline. Their latest word wins."""
        try:
            found = json.loads(self.infinito.call("cognition_decisions"))
        except (InfinitoError, ValueError):
            return 0  # the server is not there yet, or does not have the tool: nothing decided there
        local = {p["id"]: p["status"] for p in proposals(self.diary)}
        changed = 0
        for decided in found if isinstance(found, list) else []:
            pid, status = decided.get("id"), decided.get("status")
            if pid in local and status in ("approved", "rejected", "revoked") and local[pid] != status:
                self.diary.append("policy_decision", proposal=pid, decision=status, actor="owner (web)",
                                  decided_at=decided.get("decided_at"), body={"why": str(decided.get("note") or "")})
                changed += 1
        return changed

    def payload(self, run: str) -> Optional[dict]:
        """A live review as the person's page shows it: built from the diary, so it can be sent again if it failed."""
        entries = [e for e in self.diary.entries() if e.get("run") == run]
        end = next((e for e in entries if e["type"] == "review_end" and e.get("outcome") == "reviewed"), None)
        if end is None:
            return None
        of = lambda kind, **match: next((e for e in entries if e["type"] == kind  # noqa: E731
                                         and all(e.get(k) == v for k, v in match.items())), None)
        review = self.diary.body(of("supervisor_review") or {"seq": 0}) or {}
        hypotheses = (self.diary.body(of("hypothesis_set", arm="hierarchy") or {"seq": 0}) or {}).get("hypotheses", [])
        verdicts = {v.get("hypothesis"): v for v in review.get("verdicts", [])}
        published = of("reflection_published")
        reflection, reflection_id = (self.diary.body(published) or {}) if published else {}, end.get("reflection")
        for again in self.diary.entries("reflection_republished"):  # republished after its source was forgotten
            if reflection_id and again.get("replaces") == reflection_id:
                reflection, reflection_id = self.diary.body(again) or {}, again.get("reflection")
        inventory = of("inventory") or {}
        supervisor = of("call", role="supervisor") or {}
        cover = end.get("coverage") or {}
        return {
            "run": run, "summary": review.get("summary_for_owner", ""),
            "reflection": {"id": reflection_id or "", "text": reflection.get("reflection", ""),
                           "advice": reflection.get("advice") or "", "questions": reflection.get("questions") or [],
                           "sources": reflection.get("sources") or [], "model": supervisor.get("model", "")},
            "coverage": {"read": cover.get("inventory", 0), "reported": inventory.get("reported_new", 0),
                         "nobody": len(cover.get("cited_by_nobody", [])), "limits": cover.get("limits", [])},
            "audit": review.get("audit", []),
            "hypotheses": [{"id": h.get("id"), "statement": h.get("statement"), "falsifier": h.get("falsifier"),
                            "verdict": verdicts.get(h.get("id"), {}).get("verdict", "untested"),
                            "why": verdicts.get(h.get("id"), {}).get("why", ""),
                            "ids": verdicts.get(h.get("id"), {}).get("ids") or h.get("evidence_ids", [])}
                           for h in hypotheses],
            "costs": {"spent_eur": end.get("spent_eur", 0), "cap_eur": end.get("cap_eur", 0)},
            "proposals": [{"id": e["proposal"], "target_role": e["target_role"], **(self.diary.body(e) or {})}
                          for e in entries if e["type"] == "policy_proposal"],
        }

    def publish(self, run: str) -> Optional[str]:
        """Send a live review to the person's page and tell them it is there. None when it could not go."""
        review = self.payload(run)
        if review is None:
            return None
        for proposal in review["proposals"]:
            proposal.pop("salt", None)
        try:
            answer = self.infinito.call("cognition_report", review=review)
        except InfinitoError as error:
            self.diary.append("error", run=run, where="report", code=str(error)[:200])
            return None
        if not answer.startswith(("Kept as review", "Already kept")):
            self.diary.append("error", run=run, where="report", code=answer[:200])
            return None
        self.diary.append("review_reported", run=run, proposals=[p["id"] for p in review["proposals"]])
        if self.config.get("notify", True):
            from .notify import review_waiting

            review_waiting(self.config["server"], len(review["proposals"]), bool(review["reflection"]["text"]))
        return answer

    def publish_unreported(self) -> int:
        """Live reviews of the last week that never reached the person's page (the server was down or older): again."""
        reported = {e.get("run") for e in self.diary.entries("review_reported")}
        week_ago = self.clock() - timedelta(days=7)
        sent = 0
        for end in self.diary.entries("review_end"):
            if (end.get("live") and end.get("outcome") == "reviewed" and end.get("run") not in reported
                    and _parse_at(end["at"]) > week_ago and self.publish(end["run"])):
                sent += 1
        return sent

    # Reading INFINITO

    def _since(self, now: datetime) -> datetime:
        last = self.last_review()
        return _parse_at(last["at"]) if last else now - timedelta(days=1)

    def inventory(self, since: datetime, now: datetime) -> dict:
        context = self.infinito.call("reflection_context", since=_utc(since))
        new_part, _, earlier_part = context.partition(EARLIER)
        reported = int(NEW_HEADER.search(new_part).group(1)) if NEW_HEADER.search(new_part) else 0
        ids = validate.memory_ids(new_part)
        listing = self.infinito.call("list_memories")
        whole = {}
        for line in listing.splitlines():
            found = validate.memory_ids(line)
            if found:
                whole.setdefault(found[0], line.strip())
        missing = reported - len(ids)
        if missing > 0:
            # INFINITO lists only the latest 30 new ones. The others are the ones saved just before them: in the full
            # listing (by day, then in the order they were saved), the last ones from that day on that it did not list.
            first_day = since.astimezone(timezone.utc).date().isoformat()
            candidates = [m for m, line in whole.items() if m not in ids
                          and (re.search(r"\] \((\d{4}-\d{2}-\d{2})", line) or [None, ""])[1] >= first_day]
            ids = candidates[-missing:] + ids
        short = {m: line.strip() for line in new_part.splitlines() for m in validate.memory_ids(line)[:1]}
        lines = [whole.get(m) or short.get(m, f"[{m}]") for m in ids]
        reflections = self.infinito.call("reflection_list",
                                         since=(now - timedelta(days=REFLECTIONS_DAYS)).date().isoformat())
        try:
            tasks = self.infinito.call("task_list")
        except InfinitoError:
            tasks = "(the task queue could not be read)"
        return {"ids": ids, "text": "\n".join(lines), "reported_new": reported, "shown_new": len(ids),
                "reflections": reflections + "\n" + earlier_part, "tasks": tasks}

    # Calling a model

    def _fit(self, role: str, text: str, other_chars: int) -> Tuple[str, bool]:
        room = int(self.config["roles"][role]["max_input_chars"]) - other_chars
        if len(text) <= room:
            return text, False
        return text[:max(room, 0)].rsplit("\n", 1)[0] + "\n[... cortado por el límite de entrada]", True

    def supervisor_share(self, inventory: dict) -> float:
        """What the supervisor's call may cost, held back from the children: the review needs it most. Nothing is held
        when it could not be paid for anyway, or its price is unknown."""
        if not self.live:
            return 0.0
        settings = self.config["roles"]["supervisor"]
        chars = (len(inventory["text"]) + len(inventory["reflections"]) + len(inventory["tasks"]) + SUPERVISOR_EXTRA)
        try:
            share = estimate_eur(self.config, settings["model"], min(chars, int(settings["max_input_chars"])),
                                 int(settings["max_output"]))
        except UnknownPrice:
            return 0.0
        remaining = self.ledger.remaining(self.clock().date())
        return share if share <= remaining else 0.0

    def ask(self, run: str, role: str, prompt: str, *, hold: float = 0.0) -> Optional[dict]:
        settings = self.config["roles"][role]
        guidelines = approved_guidelines(self.diary, role)
        system = roles.system(role, guidelines, self.config.get("missions") or [])
        day = self.clock().date()
        header = {"run": run, "role": role, "provider": settings.get("provider") if self.live else "fake",
                  "model": settings.get("model") if self.live else "fake", "prompt_version": roles.PROMPT_VERSION,
                  "guidelines": [g["id"] for g in guidelines], "prompt_chars": len(system) + len(prompt)}
        estimate = 0.0
        if self.live:
            try:
                estimate = estimate_eur(self.config, settings["model"], len(system) + len(prompt),
                                        int(settings["max_output"]))
            except UnknownPrice:
                self.diary.append("error", **header, code="unknown_price")
                return None
            if estimate + hold > self.ledger.remaining(day):
                self.diary.append("budget_skip", **header, estimate_eur=round(estimate, 4),
                                  remaining_eur=round(self.ledger.remaining(day), 4), held_eur=round(hold, 4))
                return None
            try:
                model = self.make_model(role, settings)
            except ProviderError as error:
                self.diary.append("error", **header, code=str(error))
                return None
        else:
            model = Fake(role)
        try:
            reply = model.complete(system, prompt, int(settings["max_output"]))
        except ProviderError as error:
            self.diary.append("error", **header, code=str(error))
            return None
        cost = cost_eur(self.config, settings["model"], reply.usage.as_dict()) if self.live else 0.0
        if self.live:
            self.ledger.charge(day, cost, overrun=(cost - estimate) if cost > estimate else None)
        data = roles.parse(reply.text) if reply.finished else None
        self.diary.append("call", **header, usage=reply.usage.as_dict(), cost_eur=round(cost, 5),
                          finished=reply.finished, understood=data is not None)
        return data

    # Between reviews

    def watch(self) -> str:
        now = self.clock()
        last = self.diary.last("observation_set", live=self.live)
        since = max([self._since(now)] + ([_parse_at(last["at"])] if last else []))
        try:
            context = self.infinito.call("reflection_context", since=_utc(since))
        except InfinitoError as error:
            return f"INFINITO did not answer: {error}"
        match = NEW_HEADER.search(context)
        new = int(match.group(1)) if match else 0
        if new < int(self.config["observer_threshold"]):
            return f"nothing due ({new} new memories)"
        run = f"w{now:%Y%m%d%H%M}"
        inventory = self.inventory(since, now)
        text, cut = self._fit("observer", inventory["text"], 4000)
        data = self.ask(run, "observer", roles.section("Recuerdos nuevos", text))
        kept, aside = validate.grounded(roles.items(data, "observations"), "memory_ids", set(inventory["ids"]))
        self.diary.append("observation_set", run=run, ids=inventory["ids"], pass_="threshold", truncated=cut,
                          live=self.live, body={"observations": kept, "set_aside": aside})
        return f"observer pass on {len(inventory['ids'])} new memories"

    # The review

    def review(self) -> str:
        now = self.clock()
        run = f"r{now:%Y%m%d%H%M}"
        since = self._since(now)
        self.diary.append("review_start", run=run, live=self.live, since=since.isoformat())
        try:
            inventory = self.inventory(since, now)
        except InfinitoError as error:
            self.diary.append("error", run=run, where="inventory", code=str(error)[:200])
            return f"INFINITO did not answer: {error}"
        known = set(inventory["ids"])
        still_open = validate.open_ids(inventory["text"])
        self.diary.append("inventory", run=run, ids=inventory["ids"], reported_new=inventory["reported_new"],
                          shown_new=inventory["shown_new"])
        if not inventory["ids"]:
            self.diary.append("review_end", run=run, outcome="nothing new", live=self.live)
            return self._report(run, now, inventory, {}, nothing_new=True)

        cut = {}
        hold = self.supervisor_share(inventory)
        earlier = self._observation_sets(since)
        # Observer
        text, cut["observer"] = self._fit("observer", inventory["text"], 4000)
        observed = self.ask(run, "observer", roles.section("Recuerdos nuevos desde la última revisión", text),
                            hold=hold)
        observations, observations_aside = validate.grounded(roles.items(observed, "observations"), "memory_ids",
                                                             known)
        for body in earlier:
            observations += body.get("observations", [])
        self.diary.append("observation_set", run=run, ids=[i for o in observations for i in o.get("memory_ids", [])],
                          pass_="review", live=self.live,
                          body={"observations": observations, "set_aside": observations_aside})

        # Investigator
        report = json.dumps({"observations": observations}, ensure_ascii=False, indent=1)
        text, cut["investigator"] = self._fit("investigator", inventory["text"], len(report) + 4000)
        investigated = self.ask(run, "investigator", roles.section("Recuerdos nuevos", text)
                                + roles.section("Informe del observador", report), hold=hold)
        hypotheses, hypotheses_aside = validate.grounded(roles.items(investigated, "hypotheses"), "evidence_ids", known)
        self.diary.append("hypothesis_set", run=run, arm="hierarchy",
                          ids=[i for h in hypotheses for i in h.get("evidence_ids", [])],
                          body={"hypotheses": hypotheses, "set_aside": hypotheses_aside})

        # Checks: read-only recalls the investigator asked for, to put each hypothesis to the test
        checks, check_ids = [], set()
        for hypothesis in hypotheses[:int(self.config["max_checks"])]:
            query = str(hypothesis.get("check_query") or "").strip()[:300]
            if not query:
                continue
            try:
                answer = self.infinito.call("recall", query=query, limit=CHECK_LIMIT)
            except InfinitoError as error:
                checks.append({"hypothesis": hypothesis.get("id"), "query": query, "error": str(error)[:200]})
                continue
            lines = [line.strip() for line in answer.splitlines() if validate.memory_ids(line)]
            found = [validate.memory_ids(line)[0] for line in lines]
            check_ids.update(found)
            still_open |= validate.open_ids(answer)
            checks.append({"hypothesis": hypothesis.get("id"), "query": query, "found_ids": found, "lines": lines})
        self.diary.append("check_set", run=run, ids=sorted(check_ids),
                          body={"checks": [{k: v for k, v in c.items() if k != "lines"} for c in checks]})

        # Baseline: one cheap pass over the same inventory, for the blind comparison
        text, cut["baseline"] = self._fit("baseline", inventory["text"], 2000)
        baseline = self.ask(run, "baseline", roles.section("Recuerdos nuevos", text), hold=hold)
        baseline_hypotheses, baseline_aside = validate.grounded(roles.items(baseline, "hypotheses"), "evidence_ids",
                                                                known)
        self.diary.append("hypothesis_set", run=run, arm="baseline",
                          ids=[i for h in baseline_hypotheses for i in h.get("evidence_ids", [])],
                          body={"hypotheses": baseline_hypotheses, "set_aside": baseline_aside})

        # Supervisor
        pending = [p for p in proposals(self.diary) if p["status"] == "pending"]
        children = (roles.section("Informe del observador (lo apartado por el árbitro va aparte)",
                                  json.dumps({"observations": observations, "set_aside": observations_aside},
                                             ensure_ascii=False, indent=1))
                    + roles.section("Hipótesis del investigador",
                                    json.dumps({"hypotheses": hypotheses, "set_aside": hypotheses_aside},
                                               ensure_ascii=False, indent=1))
                    + roles.section("Pruebas buscadas en la memoria para cada hipótesis",
                                    "\n".join(f"{c['hypothesis']} · consulta «{c['query']}»:\n  "
                                              + "\n  ".join(c.get("lines") or [c.get("error", "nada encontrado")])
                                              for c in checks))
                    + roles.section("Tus reflexiones recientes y los comentarios de la persona sobre ellas",
                                    inventory["reflections"])
                    + roles.section("Pautas que ya propusiste y esperan a la persona (no las repitas)",
                                    "\n".join(f"[{p['id']}] {p['target_role']}: {p['guideline']}" for p in pending))
                    + roles.section("Cola de tareas", inventory["tasks"]))
        text, cut["supervisor"] = self._fit("supervisor", inventory["text"], len(children) + 6000)
        judged = self.ask(run, "supervisor",
                          roles.section(f"Inventario íntegro del día ({len(inventory['ids'])} recuerdos)", text)
                          + children)
        evidence = known | check_ids  # the supervisor was shown the checks too, so it may cite them
        audit, audit_aside = validate.grounded(roles.items(judged, "audit"), "ids", evidence)
        named = {h.get("id") for h in hypotheses}
        verdicts, verdicts_aside = validate.grounded(
            [v for v in roles.items(judged, "verdicts") if v.get("hypothesis") in named], "ids", evidence)
        reflection, dropped = validate.reflection((judged or {}).get("reflection"), evidence,
                                                  set(validate.reflection_ids(inventory["reflections"])),
                                                  avoid=still_open)
        self.diary.append("supervisor_review", run=run, ids=sorted(set(i for a in audit for i in a["ids"])
                                                                   | set(i for v in verdicts for i in v["ids"])),
                          understood=judged is not None,
                          body={"audit": audit, "verdicts": verdicts,
                                "set_aside": audit_aside + verdicts_aside,
                                "summary_for_owner": str((judged or {}).get("summary_for_owner") or "")[:1500]})

        # The reflection: the person reads it, comments on it, hides or deletes it on their memory page
        published = None
        if reflection and self.live:
            try:
                answer = self.infinito.call("reflection_add", model=self.config["roles"]["supervisor"]["model"],
                                            **reflection)
                kept_as = KEPT_AS.search(answer)
                published = kept_as.group(1) if kept_as else None
                self.diary.append("reflection_published" if published else "reflection_refused", run=run,
                                  ids=reflection["sources"], reflection=published,
                                  body={**reflection, "server": answer[:500], "dropped": dropped})
            except InfinitoError as error:
                self.diary.append("reflection_refused", run=run, ids=reflection["sources"],
                                  body={**reflection, "server": str(error)[:500], "dropped": dropped})
        elif reflection:
            self.diary.append("reflection_not_published", run=run, ids=reflection["sources"], reason="dry run",
                              body={**reflection, "dropped": dropped})
        else:
            self.diary.append("reflection_not_published", run=run, reason="none kept", body={"dropped": dropped})

        # Proposals: they wait for the person
        made = []
        for number, proposal in enumerate(roles.items(judged, "proposals"), 1):
            target = proposal.get("target_role")
            guideline = str(proposal.get("guideline") or "").strip()[:600]
            if target not in ("observer", "investigator", "supervisor") or not guideline:
                continue
            pid = f"P{run[1:]}-{number}"  # unique per run: two reviews of one day never share an id
            self.diary.append("policy_proposal", run=run, proposal=pid, target_role=target,
                              body={"guideline": guideline, "rationale": str(proposal.get("rationale") or "")[:600],
                                    "test": str(proposal.get("test") or "")[:400]})
            made.append(pid)

        cover = validate.coverage(inventory["ids"], {
            "observer": [i for o in observations for i in o.get("memory_ids", [])],
            "investigator": [i for h in hypotheses for i in h.get("evidence_ids", [])],
            "supervisor": [i for a in audit for i in a["ids"]] + [i for v in verdicts for i in v["ids"]]
                          + (reflection.get("sources", []) if reflection else [])},
            reported_new=inventory["reported_new"], shown_new=inventory["shown_new"], truncated=cut)
        blind = self._blind(run, now, hypotheses, baseline_hypotheses)
        self.diary.append("review_end", run=run, outcome="reviewed", live=self.live, coverage=cover,
                          proposals=made, reflection=published, spent_eur=round(self.ledger.spent(now.date()), 4),
                          cap_eur=self.ledger.cap(now.date()))
        if self.live:  # to the person's page, where they read it all and decide on the guidelines
            self.publish(run)
        return self._report(run, now, inventory, {
            "observations": observations, "observations_aside": observations_aside, "hypotheses": hypotheses,
            "hypotheses_aside": hypotheses_aside, "checks": checks, "verdicts": verdicts, "audit": audit,
            "aside": audit_aside + verdicts_aside + baseline_aside, "reflection": reflection, "dropped": dropped,
            "published": published, "proposals": made, "coverage": cover, "blind": blind,
            "summary": str((judged or {}).get("summary_for_owner") or "")})

    def _blind(self, run: str, now: datetime, hierarchy: List[dict], baseline: List[dict]) -> List[dict]:
        """Both arms' hypotheses under neutral labels, shuffled, for the person to rate without knowing whose is
        whose. Which arm each came from stays in the diary until the comparison."""
        mixed = [("hierarchy", h) for h in hierarchy] + [("baseline", h) for h in baseline]
        random.SystemRandom().shuffle(mixed)
        shown = []
        for number, (arm, hypothesis) in enumerate(mixed, 1):
            label = f"X{run[5:]}-{number}"
            shown.append({"label": label, "statement": hypothesis.get("statement", ""),
                          "falsifier": hypothesis.get("falsifier", "")})
            self.diary.append("blind_item", run=run, label=label, ids=hypothesis.get("evidence_ids", []),
                              body={"arm": arm, "hypothesis": hypothesis.get("id")})
        return shown

    # The person's report

    def _report(self, run: str, now: datetime, inventory: dict, done: dict, *, nothing_new: bool = False) -> str:
        from .report import render

        folder = self.folder / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{now:%Y-%m-%d}.md"
        if path.exists():
            path = folder / f"{now:%Y-%m-%d}-{now:%H%M}.md"
        path.write_text(render(self, run, now, inventory, done, nothing_new=nothing_new), encoding="utf-8")
        return f"review {run} done; report in {path}"
