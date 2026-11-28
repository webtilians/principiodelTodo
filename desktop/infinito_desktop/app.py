"""The app: one per Windows user, an icon in the tray, a look at everything every few minutes, a notification only
when something gets worse (alerts.py), and since phase 2 the quick capture: Ctrl+Alt+I from anywhere, a note to the
memory in two seconds. Its timers are of minutes; between them it does nothing."""
import ctypes
import queue
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from typing import List, Optional, Tuple

from . import NAME, alerts, capture, checks, look, outbox, places, startup, ui
from .session import Store
from .state import State
from .waiting import Full, Queue

LOOK_MINUTES = 15
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
        self.queue = Queue(self.folder)
        self.root = tk.Tk()
        self.root.withdraw()
        self.windows = ui.Windows(self.root)
        self.checks: List[checks.Check] = []
        self.when = "aún no"
        self.results: "queue.Queue" = queue.Queue()  # each look's checks, from its worker thread
        self.signed: "queue.Queue" = queue.Queue()  # a sign-in that finished, from its worker thread
        self.sent: "queue.Queue" = queue.Queue()  # what happened to the notes sent, from the sending thread
        self.looking = self.sending = False
        from .tray import Tray

        self.tray = Tray(NAME, self.menu, self.show_status)
        self.hotkey_taken = not self.tray.hotkey(self.state["hotkey"], self.show_capture)

    def address(self) -> str:
        return self.store.server() or places.SERVER

    def shortcut(self) -> str:
        return f"Ctrl+Alt+{self.state['hotkey']}"

    def menu(self):
        paused = bool(self.state["paused"])
        return [("Estado", self.show_status),
                (f"Captura rápida ({self.shortcut()})", self.show_capture),
                ("Qué recojo y qué envío", self.show_panel),
                None,
                (("*" if paused else "") + "Pausa", self.toggle_pause),
                ("Abrir mi memoria", lambda: webbrowser.open(f"{self.address()}/memory")),
                ("Ajustes", self.show_settings),
                None,
                ("Salir de INFINITO Desktop", self.quit)]

    def show_status(self) -> None:
        self.windows.status(self.checks, self.when, self.look_now)

    def show_panel(self) -> None:
        self.windows.panel(bool(self.state["paused"]), len(self.queue.items()), self.state["last_sent"],
                           self.hotkey_taken, self.state["hotkey"])

    def show_capture(self) -> None:
        self.windows.capture(list(self.state["projects"]), self.shortcut(), self.store.may_send_notes(), self.save,
                             self.sign_in)

    def show_settings(self) -> None:
        self.windows.settings(self.state, self.store.signed_in(), self.address(), self.set_start, self.sign_in,
                              self.sign_out, self.set_shortcut)

    def set_start(self, on: bool) -> None:
        startup.set_on(on)
        self.state["start_with_windows"] = on

    def set_shortcut(self, letter: str) -> bool:
        if not self.tray.hotkey(letter, self.show_capture):
            self.tray.hotkey(self.state["hotkey"], self.show_capture)  # back to the one that worked
            return False
        self.state["hotkey"], self.hotkey_taken = letter, False
        return True

    def toggle_pause(self) -> None:
        self.state["paused"] = not self.state["paused"]
        self.tray.show("unknown" if self.state["paused"] else checks.worst(self.checks), self.tip())

    # The quick capture.

    def save(self, text: str, kind: Optional[str], project: str) -> Tuple[bool, str]:
        """Called by the capture window: checks the note here, queues it and sends it. (done, what to say)."""
        event, why = capture.note(text, kind, project, self.state["device_id"])
        if why:
            return False, why
        try:
            self.queue.add(event)
        except Full:
            return False, "La cola está llena: no se puede guardar hasta que se envíe lo que espera."
        if project.strip() and project.strip() not in self.state["projects"]:
            self.state["projects"] = (list(self.state["projects"]) + [project.strip()])[-50:]
        self.send_waiting(event["event_id"])
        return True, "Guardando…"

    def send_waiting(self, watching: Optional[str] = None) -> None:
        """Sends what waits, in a thread; the capture window hears what happened to its own note."""
        if self.sending:
            return
        self.sending = True
        self.watching = watching

        def work():
            try:
                self.sent.put(outbox.deliver(self.queue, self.store.send))
            except Exception:  # noqa: BLE001  # whatever happened, what waits stays and is tried again
                self.sent.put([])

        threading.Thread(target=work, daemon=True).start()
        self.root.after(300, self._sent)

    def _sent(self) -> None:
        try:
            happened = self.sent.get_nowait()
        except queue.Empty:
            self.root.after(300, self._sent)
            return
        self.sending = False
        for item in happened:
            if item["outcome"] != "later":
                self.state["last_sent"] = f"{datetime.now():%d/%m %H:%M} · {item['said']}"
        mine = next((item for item in happened if item["event_id"] == getattr(self, "watching", None)), None)
        if self.windows.capture_said and getattr(self, "watching", None):
            if mine is None:
                self.windows.capture_said("Guardada en la cola: se enviará en cuanto se pueda.", True)
            else:
                self.windows.capture_said(mine["said"], mine["outcome"] != "refused")
        self.watching = None
        if self.queue.items():  # something still waits: look again in a minute, only while it does
            self.root.after(60 * 1000, lambda: self.queue.due() and self.send_waiting())

    # The account.

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
        self.send_waiting()

    def sign_out(self) -> None:
        """Ends the session here and on the server, and empties the queue with it."""
        self.store.sign_out()
        self.queue.clear()
        self.look_now()

    # The looks.

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
        if self.queue.due():
            self.send_waiting()
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
