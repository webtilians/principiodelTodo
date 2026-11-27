"""The app: one per Windows user, an icon in the tray, a look at everything every few minutes, and a notification only
when something gets worse (alerts.py). Its timers are of minutes; between them it does nothing."""
import ctypes
import queue
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from typing import List

from . import NAME, alerts, checks, look, places, startup, ui
from .session import Store
from .state import State

LOOK_MINUTES = 15
# The session is renewed at each look; the hour-long access token is only renewed once it is near its end.
ALREADY_RUNNING = 183


def _one_at_a_time():
    """A mutex named for this user: a second copy sees it and leaves (and so renewals never race)."""
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\INFINITO-Desktop")
    return handle if ctypes.windll.kernel32.GetLastError() != ALREADY_RUNNING else None


def _notify(check: checks.Check) -> None:
    from cognition.notify import toast

    toast(f"{NAME}: {check.title}", check.sentence, check.link or f"{places.SERVER}/memory", "Ver")


class App:
    def __init__(self):
        self.folder = places.home()
        self.state = State(self.folder)
        self.store = Store(self.folder)
        self.root = tk.Tk()
        self.root.withdraw()
        self.windows = ui.Windows(self.root)
        self.checks: List[checks.Check] = []
        self.when = "aún no"
        self.results: "queue.Queue" = queue.Queue()  # each look's checks, from its worker thread
        self.signed: "queue.Queue" = queue.Queue()  # a sign-in that finished, from its worker thread
        self.looking = False
        from .tray import Tray

        self.tray = Tray(NAME, self.menu, self.show_status)

    def address(self) -> str:
        return self.store.server() or places.SERVER

    def menu(self):
        paused = bool(self.state["paused"])
        return [("Estado", self.show_status),
                ("Captura rápida (llega en la fase 2)", None),
                ("Qué recojo y qué envío", lambda: self.windows.panel(paused, "Nada todavía.")),
                None,
                (("*" if paused else "") + "Pausa", self.toggle_pause),
                ("Abrir mi memoria", lambda: webbrowser.open(f"{self.address()}/memory")),
                ("Ajustes", self.show_settings),
                None,
                ("Salir de INFINITO Desktop", self.quit)]

    def show_status(self) -> None:
        self.windows.status(self.checks, self.when, self.look_now)

    def show_settings(self) -> None:
        self.windows.settings(self.state, self.store.signed_in(), self.address(), self.set_start, self.sign_in,
                              self.sign_out)

    def set_start(self, on: bool) -> None:
        startup.set_on(on)
        self.state["start_with_windows"] = on

    def toggle_pause(self) -> None:
        self.state["paused"] = not self.state["paused"]
        self.tray.show("unknown" if self.state["paused"] else checks.worst(self.checks), self.tip())

    def sign_in(self) -> None:
        def work():
            try:
                self.store.sign_in(places.SERVER)
            except (SystemExit, OSError, ValueError, KeyError):
                pass  # the next look says whether it is signed in
            self.signed.put(True)

        threading.Thread(target=work, daemon=True).start()
        self.root.after(1000, self._after_sign_in)

    def _after_sign_in(self) -> None:
        try:
            self.signed.get_nowait()
        except queue.Empty:
            self.root.after(1000, self._after_sign_in)
            return
        self.look_now()

    def sign_out(self) -> None:
        self.store.sign_out()
        self.look_now()

    def tip(self) -> str:
        if self.state["paused"]:
            return f"{NAME}: en pausa"
        bad = [check.title for check in self.checks if check.state in (checks.WARN, checks.FAIL)]
        return f"{NAME}: " + ("todo bien" if not bad else ", ".join(bad))

    def look_now(self) -> None:
        if self.looking:
            return
        self.looking = True
        threading.Thread(target=lambda: self.results.put(look.everything(self.store, self.address())),
                         daemon=True).start()
        self.root.after(500, self._looked)

    def _looked(self) -> None:
        try:
            found = self.results.get_nowait()
        except queue.Empty:
            self.root.after(500, self._looked)
            return
        self.looking = False
        self.checks, self.when = found, f"{datetime.now():%H:%M}"
        today = datetime.now().date().isoformat()
        seen, told = dict(self.state["seen"]), dict(self.state["told"])
        news = [] if self.state["paused"] else alerts.due(found, seen, told, today)
        for check in news:
            _notify(check)
        alerts.remember(found, seen, told, news, today)
        self.state["seen"], self.state["told"] = seen, told
        self.tray.show("unknown" if self.state["paused"] else checks.worst(found), self.tip())

    def every_few_minutes(self) -> None:
        if not self.state["paused"]:
            self.look_now()
        self.root.after(LOOK_MINUTES * 60 * 1000, self.every_few_minutes)

    def quit(self) -> None:
        self.tray.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.after(2000, self.every_few_minutes)
        self.root.mainloop()


def main() -> int:
    if _one_at_a_time() is None:
        return 0  # already running: its icon is in the tray
    App().run()
    return 0
