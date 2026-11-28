"""The app's windows (tkinter): the state of each check, the quick capture, «Qué recojo y qué envío», and the settings."""
import os
import tkinter as tk
import webbrowser
from tkinter import ttk
from typing import Callable, List, Optional, Tuple

from . import NAME, VERSION
from .capture import KINDS
from .checks import Check, WORDS

DOTS = {"ok": "#34c759", "warn": "#ffb020", "fail": "#e63946", "unknown": "#9696a0"}
COBALT = "#2a44d4"
WIDTH = 540


def open_link(link: str) -> None:
    if link.startswith(("http://", "https://")):
        webbrowser.open(link)
    elif link.startswith("file:"):
        os.startfile(link)  # noqa: S606  # a folder of INFINITO on this PC, opened in the Explorer


def icon(root: tk.Tk) -> tk.PhotoImage:
    """INFINITO's mark for every window (a cobalt dot), instead of Tk's feather."""
    size, image = 32, tk.PhotoImage(width=32, height=32)
    middle, radius = (size - 1) / 2, size / 2 - 2
    for y in range(size):
        row = []
        for x in range(size):
            inside = (x - middle) ** 2 + (y - middle) ** 2 <= radius ** 2
            row.append(COBALT if inside else "#ffffff")
        image.put("{" + " ".join(row) + "}", to=(0, y))
    for y in range(size):  # the corners transparent, so it is round on any background
        for x in range(size):
            if (x - middle) ** 2 + (y - middle) ** 2 > radius ** 2:
                image.transparency_set(x, y, True)
    root.iconphoto(True, image)
    return image


class Windows:
    def __init__(self, root: tk.Tk):
        self.root, self.open = root, {}
        self.mark = icon(root)
        self.capture_said: Optional[Callable[[str, bool], None]] = None

    def _window(self, key: str, title: str) -> tk.Toplevel:
        old = self.open.get(key)
        if old is not None and old.winfo_exists():
            old.destroy()
        window = tk.Toplevel(self.root)
        window.title(f"{title} · {NAME}")
        window.resizable(False, False)
        window.attributes("-topmost", True)
        window.after(300, lambda: window.attributes("-topmost", False))
        self.open[key] = window
        return window

    def status(self, checks: List[Check], when: str, again: Callable[[], None]) -> None:
        window = self._window("status", "Estado")
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both")
        ttk.Label(frame, text=f"Así está INFINITO en este PC ({when})", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        for check in checks:
            row = ttk.Frame(frame, padding=(0, 8, 0, 0))
            row.pack(fill="x")
            tk.Label(row, text="●", fg=DOTS[check.state], font=("Segoe UI", 14)).pack(side="left", anchor="n")
            text = ttk.Frame(row)
            text.pack(side="left", fill="x", padx=6)
            ttk.Label(text, text=f"{check.title} — {WORDS[check.state]}", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            ttk.Label(text, text=check.sentence, wraplength=WIDTH - 60, justify="left").pack(anchor="w")
            if check.link:
                link = ttk.Label(text, text="Abrir", foreground=COBALT, cursor="hand2")
                link.pack(anchor="w")
                link.bind("<Button-1>", lambda _event, where=check.link: open_link(where))
        buttons = ttk.Frame(frame, padding=(0, 12, 0, 0))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Comprobar ahora", command=lambda: (window.destroy(), again())).pack(side="left")
        ttk.Button(buttons, text="Cerrar", command=window.destroy).pack(side="right")

    def capture(self, projects: List[str], shortcut: str, may_send: bool,
                on_save: Callable[[str, Optional[str], str], Tuple[bool, str]], sign_in: Callable[[], None]) -> None:
        """A small window: what to keep, its kind and project, and Ctrl+Enter to keep it. Win+H dictates into it."""
        window = self._window("capture", "Captura rápida")
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both")
        ttk.Label(frame, text="¿Qué quieres que recuerde tu memoria?", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        text = tk.Text(frame, width=60, height=6, wrap="word", font=("Segoe UI", 10), undo=True)
        text.pack(fill="both", pady=(6, 6))
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text="Tipo").pack(side="left")
        kind = ttk.Combobox(row, values=list(KINDS), state="readonly", width=14)
        kind.set("")
        kind.pack(side="left", padx=(4, 12))
        ttk.Label(row, text="Proyecto").pack(side="left")
        project = ttk.Combobox(row, values=projects, width=24)
        project.pack(side="left", padx=(4, 0))
        said = ttk.Label(frame, text=("" if may_send else "La app tiene que entrar otra vez en tu cuenta para guardar "
                                      "notas: la nota esperará en la cola hasta entonces."),
                         wraplength=WIDTH - 60, justify="left")
        said.pack(anchor="w", pady=(8, 0))
        buttons = ttk.Frame(frame, padding=(0, 8, 0, 0))
        buttons.pack(fill="x")

        def keep(_event=None):
            done, sentence = on_save(text.get("1.0", "end"), KINDS.get(kind.get()), project.get())
            said.configure(text=sentence, foreground="" if done else "#b3261e")
            if done:
                text.configure(state="disabled")
            return "break"

        def report(sentence: str, finished: bool) -> None:
            if not window.winfo_exists():
                return
            said.configure(text=sentence, foreground="")
            if finished:
                window.after(1500, window.destroy)

        self.capture_said = report
        if not may_send:
            ttk.Button(buttons, text="Entrar en mi cuenta", command=lambda: (window.destroy(), sign_in())).pack(
                side="left")
        ttk.Button(buttons, text=f"Guardar (Ctrl+Intro)", command=keep).pack(side="right")
        ttk.Button(buttons, text="Cancelar (Esc)", command=window.destroy).pack(side="right", padx=(0, 6))
        ttk.Label(buttons, text=f"Atajo: {shortcut} · dictar: Win+H", foreground="#6b6b78").pack(side="left", padx=6)
        window.bind("<Control-Return>", keep)
        window.bind("<Escape>", lambda _event: window.destroy())
        window.after(50, lambda: (window.focus_force(), text.focus_set()))

    def panel(self, paused: bool, waiting: int, last_sent: str, hotkey_taken: bool, shortcut: str) -> None:
        window = self._window("panel", "Qué recojo y qué envío")
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both")
        lines = [
            ("Sensores", "Ninguno todavía: cada uno llegará con su interruptor, apagado (fase 4)."),
            ("Qué lee en este PC", "Solo archivos de INFINITO: el estado del motor cognitivo, el registro de las "
                                   "copias, las tareas programadas «INFINITO», si los hooks de Claude Code están en "
                                   "~/.claude/settings.json y si tienen sesión (sin abrirla)."),
            ("Qué envía", "Solo lo que escribes o dictas en la captura rápida, como nota tuya, y solo a tu servidor de "
                          "INFINITO. Antes de salir del PC pasa por el filtro de secretos: lo que parezca una "
                          "contraseña, una clave o un número de tarjeta no se guarda ni se envía."),
            ("Captura rápida", f"Ctrl+Alt+{shortcut} desde cualquier sitio" + (
                " (ese atajo lo tiene otro programa: elige otra letra en Ajustes)." if hotkey_taken else ".")),
            ("Cola", f"{waiting} nota{'s' if waiting != 1 else ''} esperando a enviarse, cifrada{'s' if waiting != 1 else ''} "
                     "en este PC." if waiting else "Vacía."),
            ("Último envío", last_sent),
            ("Pausa general", "Activada: no comprueba ni avisa." if paused else "Desactivada."),
        ]
        for title, text in lines:
            ttk.Label(frame, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
            ttk.Label(frame, text=text, wraplength=WIDTH - 40, justify="left").pack(anchor="w")
        ttk.Button(frame, text="Cerrar", command=window.destroy).pack(anchor="e", pady=(12, 0))

    def settings(self, state, signed_in: bool, server: str, on_start: Callable[[bool], None],
                 sign_in: Callable[[], None], sign_out: Callable[[], None], on_shortcut: Callable[[str], bool]) -> None:
        window = self._window("settings", "Ajustes")
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both")
        ttk.Label(frame, text=f"{NAME} {VERSION}", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(frame, text=f"Servidor: {server}").pack(anchor="w", pady=(4, 0))
        start = tk.BooleanVar(value=bool(state["start_with_windows"]))
        ttk.Checkbutton(frame, text="Arrancar con Windows", variable=start,
                        command=lambda: on_start(start.get())).pack(anchor="w", pady=(10, 0))

        ttk.Label(frame, text="Captura rápida", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 0))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Atajo: Ctrl+Alt+").pack(side="left")
        letter = tk.StringVar(value=state["hotkey"])
        ttk.Entry(row, textvariable=letter, width=3).pack(side="left")
        ttk.Label(frame, text="Proyectos para elegir (separados por comas)").pack(anchor="w", pady=(6, 0))
        projects = tk.StringVar(value=", ".join(state["projects"]))
        ttk.Entry(frame, textvariable=projects, width=60).pack(anchor="w")
        said = ttk.Label(frame, text="", foreground="#b3261e")
        said.pack(anchor="w")

        ttk.Label(frame, text="Recursos para el trabajo de fondo (los usará el nodo cognitivo, fase 5)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 0))
        budget = dict(state["budget"])
        fields = {}
        for name, label in (("cpu_percent", "CPU máxima (%)"), ("ram_gb", "RAM máxima (GB)"),
                            ("gpu_percent", "GPU máxima (%)")):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=22).pack(side="left")
            fields[name] = tk.StringVar(value=str(budget.get(name, "")))
            ttk.Entry(row, textvariable=fields[name], width=8).pack(side="left")
        battery = tk.BooleanVar(value=bool(budget.get("never_on_battery", True)))
        ttk.Checkbutton(frame, text="Nunca con batería", variable=battery).pack(anchor="w")

        def keep() -> bool:
            for name, value in fields.items():
                try:
                    budget[name] = max(0, min(100, int(value.get())))
                except ValueError:
                    pass
            budget["never_on_battery"] = bool(battery.get())
            state["budget"] = budget
            state["projects"] = [name.strip()[:80] for name in projects.get().split(",") if name.strip()][:50]
            chosen = letter.get().strip().upper()[:1]
            if not ("A" <= chosen <= "Z"):
                said.configure(text="El atajo es Ctrl+Alt y una letra, de la A a la Z.")
                return False
            if chosen != state["hotkey"] and not on_shortcut(chosen):
                said.configure(text=f"Ctrl+Alt+{chosen} lo tiene otro programa: elige otra letra.")
                return False
            return True

        account = ttk.Frame(frame, padding=(0, 12, 0, 0))
        account.pack(fill="x")
        if signed_in:
            ttk.Button(account, text="Cerrar sesión", command=lambda: (window.destroy(), sign_out())).pack(side="left")
            ttk.Button(account, text="Volver a entrar", command=lambda: (window.destroy(), sign_in())).pack(
                side="left", padx=(6, 0))
        else:
            ttk.Button(account, text="Entrar en mi cuenta", command=lambda: (window.destroy(), sign_in())).pack(
                side="left")
        ttk.Button(account, text="Guardar", command=lambda: keep() and window.destroy()).pack(side="right")
