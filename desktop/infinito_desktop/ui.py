"""The app's windows (tkinter): the state of each check, «Qué recojo y qué envío», and the settings."""
import os
import tkinter as tk
import webbrowser
from tkinter import ttk
from typing import Callable, List

from . import NAME, VERSION
from .checks import Check, WORDS

DOTS = {"ok": "#34c759", "warn": "#ffb020", "fail": "#e63946", "unknown": "#9696a0"}
WIDTH = 540


def open_link(link: str) -> None:
    if link.startswith(("http://", "https://")):
        webbrowser.open(link)
    elif link.startswith("file:"):
        os.startfile(link)  # noqa: S606  # a folder of INFINITO on this PC, opened in the Explorer


class Windows:
    def __init__(self, root: tk.Tk):
        self.root, self.open = root, {}

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
                link = ttk.Label(text, text="Abrir", foreground="#2a44d4", cursor="hand2")
                link.pack(anchor="w")
                link.bind("<Button-1>", lambda _event, where=check.link: open_link(where))
        buttons = ttk.Frame(frame, padding=(0, 12, 0, 0))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Comprobar ahora", command=lambda: (window.destroy(), again())).pack(side="left")
        ttk.Button(buttons, text="Cerrar", command=window.destroy).pack(side="right")

    def panel(self, paused: bool, last_sent: str) -> None:
        window = self._window("panel", "Qué recojo y qué envío")
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both")
        lines = [
            ("Sensores", "Ninguno todavía: cada uno llegará con su interruptor, apagado (fase 4)."),
            ("Qué lee en este PC", "Solo archivos de INFINITO: el estado del motor cognitivo, el registro de las "
                                   "copias, las tareas programadas «INFINITO», si los hooks de Claude Code están en "
                                   "~/.claude/settings.json y si tienen sesión (sin abrirla)."),
            ("Qué envía", "Nada a tu memoria. Al servidor solo le pregunta si responde y renueva su propia sesión, "
                          "que no abre tu memoria."),
            ("Cola", "Vacía: la usará la captura rápida (fase 2)."),
            ("Último envío", last_sent),
            ("Pausa general", "Activada: no comprueba ni avisa." if paused else "Desactivada."),
        ]
        for title, text in lines:
            ttk.Label(frame, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
            ttk.Label(frame, text=text, wraplength=WIDTH - 40, justify="left").pack(anchor="w")
        ttk.Button(frame, text="Cerrar", command=window.destroy).pack(anchor="e", pady=(12, 0))

    def settings(self, state, signed_in: bool, server: str, on_start: Callable[[bool], None],
                 sign_in: Callable[[], None], sign_out: Callable[[], None]) -> None:
        window = self._window("settings", "Ajustes")
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both")
        ttk.Label(frame, text=f"{NAME} {VERSION}", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(frame, text=f"Servidor: {server}").pack(anchor="w", pady=(4, 0))
        start = tk.BooleanVar(value=bool(state["start_with_windows"]))
        ttk.Checkbutton(frame, text="Arrancar con Windows", variable=start,
                        command=lambda: on_start(start.get())).pack(anchor="w", pady=(10, 0))
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

        def keep() -> None:
            for name, value in fields.items():
                try:
                    budget[name] = max(0, min(100, int(value.get())))
                except ValueError:
                    pass
            budget["never_on_battery"] = bool(battery.get())
            state["budget"] = budget

        account = ttk.Frame(frame, padding=(0, 12, 0, 0))
        account.pack(fill="x")
        if signed_in:
            ttk.Button(account, text="Cerrar sesión", command=lambda: (window.destroy(), sign_out())).pack(side="left")
        else:
            ttk.Button(account, text="Entrar en mi cuenta", command=lambda: (window.destroy(), sign_in())).pack(
                side="left")
        ttk.Button(account, text="Guardar", command=lambda: (keep(), window.destroy())).pack(side="right")
