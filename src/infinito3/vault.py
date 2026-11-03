"""INFINITO's vault: secrets the agent can name but never read.

Passwords, PINs and keys live in a KeePass database (KDBX) encrypted with a
master password and, if the user wants, a key file kept apart (on a USB stick,
for example), so the same file opens on any computer or phone with KeePassXC or
KeePassDX, with or without INFINITO. The agent only sees entry names. Opening
the vault, typing a new secret and showing one all happen in small windows on
the user's own screen: the master password and the secret values never pass
through the agent, its model or the chat, so nobody who tricks the agent can get
them. Every unlock, addition and reveal is logged with its time and entry name,
never with a value. The vault locks itself after a while without use.

Needs pykeepass (pip install "infinito-memory[vault]"); the windows use tkinter.
"""
from __future__ import annotations

import json
import locale
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

HOME = Path.home() / ".infinito"


@dataclass
class Credentials:
    password: str
    keyfile: Optional[str] = None


class VaultUI(Protocol):
    def unlock(self, vault: Path, keyfile: Optional[str], creating: bool) -> Optional[Credentials]: ...

    def new_secret(self, title: str) -> Optional[Tuple[str, str]]: ...

    def show(self, title: str, username: str, password: str) -> None: ...


class Vault:
    """The KeePass file, opened only through the user's own screen."""

    def __init__(self, path: Optional[str] = None, ui: Optional[VaultUI] = None, *,
                 home: Optional[Path] = None, idle_minutes: Optional[float] = None):
        self.path = Path(path or os.environ.get("INFINITO_VAULT") or (Path(home) if home else HOME) / "boveda.kdbx")
        # Settings and the log live next to the vault file.
        self.home = Path(home) if home else self.path.parent
        self.ui = ui or TkVaultUI()
        self.idle_seconds = 60 * float(idle_minutes if idle_minutes is not None
                                       else os.environ.get("INFINITO_VAULT_IDLE_MINUTES", 15))
        self._db = None
        self._used = 0.0
        self._lock = threading.Lock()

    # Settings and log: paths, names and times only, never a secret.

    def _settings_path(self) -> Path:
        return self.home / "vault.json"

    def _keyfile(self) -> Optional[str]:
        try:
            return json.loads(self._settings_path().read_text(encoding="utf-8")).get("keyfile")
        except (OSError, ValueError):
            return None

    def _remember_keyfile(self, keyfile: Optional[str]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self._settings_path().write_text(json.dumps({"keyfile": keyfile}, ensure_ascii=False), encoding="utf-8")

    def _log(self, action: str, entry: str = "") -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        line = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "action": action}
        if entry:
            line["entry"] = entry
        with (self.home / "vault-log.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _open_db(self):
        if self._db is not None and time.monotonic() - self._used > self.idle_seconds:
            self._db = None  # locked after a while without use
        if self._db is not None:
            self._used = time.monotonic()
        return self._db

    # What the agent can do.

    def status(self) -> str:
        with self._lock:
            if not self.path.exists():
                return f"No vault yet. vault_unlock creates one at {self.path}."
            db = self._open_db()
            if db is None:
                return f"The vault at {self.path} is locked. vault_unlock asks the user for the master password."
            return f"The vault is open with {len(db.entries)} entries."

    def unlock(self) -> str:
        from pykeepass import PyKeePass, create_database
        from pykeepass.exceptions import CredentialsError

        with self._lock:
            if self._open_db() is not None:
                return "The vault is already open."
            creating = not self.path.exists()
            credentials = self.ui.unlock(self.path, self._keyfile(), creating)
            if credentials is None:
                return "The user closed the window: the vault stays locked."
            try:
                if creating:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    db = create_database(str(self.path), password=credentials.password, keyfile=credentials.keyfile)
                else:
                    db = PyKeePass(str(self.path), password=credentials.password, keyfile=credentials.keyfile)
            except CredentialsError:
                self._log("unlock_failed")
                return "Wrong master password or key file: the vault stays locked."
            except FileNotFoundError:
                return "The key file was not found (is the USB stick plugged in?): the vault stays locked."
            self._remember_keyfile(credentials.keyfile)
            self._db, self._used = db, time.monotonic()
            self._log("created" if creating else "unlocked")
            return (f"Vault created at {self.path}; it is open and empty." if creating
                    else f"The vault is open with {len(db.entries)} entries.")

    def lock(self) -> str:
        with self._lock:
            self._db = None
            return "The vault is locked."

    def names(self) -> str:
        with self._lock:
            db = self._open_db()
            if db is None:
                return "The vault is locked: call vault_unlock first."
            titles = sorted((e.title or "(untitled)") for e in db.entries)
            if not titles:
                return "The vault is empty."
            return "Entries in the vault (names only; values are never shown to you):\n" + "\n".join(
                f"- {t}" for t in titles)

    def _find(self, db, name: str):
        wanted = (name or "").strip().casefold()
        entries = [e for e in db.entries if (e.title or "").casefold() == wanted]
        if not entries:
            entries = [e for e in db.entries if wanted and wanted in (e.title or "").casefold()]
        return entries

    def reveal(self, name: str) -> str:
        with self._lock:
            db = self._open_db()
            if db is None:
                return "The vault is locked: call vault_unlock first."
            entries = self._find(db, name)
            if len(entries) != 1:
                titles = ", ".join(sorted(e.title or "" for e in (entries or db.entries)))
                what = "Several entries match" if entries else f"No entry matches {name!r}"
                return f"{what}. Ask the user which one; entries: {titles}."
            entry = entries[0]
            self.ui.show(entry.title or "", entry.username or "", entry.password or "")
            self._log("revealed", entry.title or "")
            return (f"Shown to the user in a window on their screen: {entry.title}. "
                    "The value was not sent to you; do not ask the user to paste it into the chat.")

    def add(self, name: str) -> str:
        with self._lock:
            db = self._open_db()
            if db is None:
                return "The vault is locked: call vault_unlock first."
            title = (name or "").strip()
            if not title:
                return "Give the entry a name, such as 'Router de casa'."
            typed = self.ui.new_secret(title)
            if typed is None:
                return "The user closed the window: nothing was saved."
            username, password = typed
            existing = self._find(db, title)
            if len(existing) == 1 and (existing[0].title or "").casefold() == title.casefold():
                existing[0].username, existing[0].password = username, password
                action, result = "replaced", f"Replaced the secret of {title}."
            else:
                db.add_entry(db.root_group, title, username, password)
                action, result = "added", f"Saved {title} in the vault."
            db.save()
            self._log(action, title)
            return result + " Its value was typed by the user and never sent to you."


# The user's screen.

_TEXT = {
    "es": {
        "title": "INFINITO · bóveda",
        "open": "Abrir la bóveda", "create_head": "Crear tu bóveda",
        "open_note": "Escribe tu contraseña maestra. Hilo no la verá nunca.",
        "create_note": ("Elige una contraseña maestra. Apúntala en papel y guárdala en casa: "
                        "si la olvidas, nadie podrá recuperar tus claves, ni siquiera tú."),
        "password": "Contraseña maestra", "repeat": "Repítela",
        "keyfile": "Archivo-llave (opcional)", "choose": "Elegir…",
        "make_keyfile": "Crear un archivo-llave en otra carpeta (por ejemplo, un pendrive)",
        "open_button": "Abrir", "create_button": "Crear", "cancel": "Cancelar",
        "short": "Usa al menos 8 caracteres.", "differ": "Las dos contraseñas no coinciden.",
        "new_head": "Guardar una clave", "user": "Usuario (opcional)", "secret": "Contraseña o clave",
        "show": "Mostrar", "save": "Guardar", "empty": "Escribe la clave.",
        "user_label": "Usuario", "secret_label": "Clave", "copy": "Copiar la clave", "close": "Cerrar",
        "copied": "Copiada. Se borrará del portapapeles en {n} s.",
        "closing": "Esta ventana se cierra sola en {n} s.",
    },
    "en": {
        "title": "INFINITO · vault",
        "open": "Open the vault", "create_head": "Create your vault",
        "open_note": "Type your master password. Your agent never sees it.",
        "create_note": ("Choose a master password. Write it down and keep it at home: if you forget it, "
                        "nobody can recover your secrets, not even you."),
        "password": "Master password", "repeat": "Repeat it",
        "keyfile": "Key file (optional)", "choose": "Choose…",
        "make_keyfile": "Create a key file in another folder (a USB stick, for example)",
        "open_button": "Open", "create_button": "Create", "cancel": "Cancel",
        "short": "Use at least 8 characters.", "differ": "The two passwords differ.",
        "new_head": "Save a secret", "user": "Username (optional)", "secret": "Password or key",
        "show": "Show", "save": "Save", "empty": "Type the secret.",
        "user_label": "Username", "secret_label": "Secret", "copy": "Copy the secret", "close": "Close",
        "copied": "Copied. It will be cleared from the clipboard in {n} s.",
        "closing": "This window closes by itself in {n} s.",
    },
}


def _language() -> str:
    chosen = os.environ.get("INFINITO_LANG", "")
    if not chosen:
        try:
            chosen = locale.getlocale()[0] or ""
        except ValueError:
            chosen = ""
    return "es" if chosen.lower().startswith(("es", "spanish")) else "en"


class TkVaultUI:
    """Small windows on the user's screen, one at a time; nothing typed here leaves the computer."""

    COPY_SECONDS = 20
    SHOW_SECONDS = 60
    _one_at_a_time = threading.Lock()

    def __init__(self, language: Optional[str] = None):
        self.t = _TEXT[language or _language()]

    def _window(self, heading: str):
        import tkinter as tk

        root = tk.Tk()
        root.title(self.t["title"])
        root.resizable(False, False)
        root.attributes("-topmost", True)
        frame = tk.Frame(root, padx=18, pady=16)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=heading, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        return root, frame

    @staticmethod
    def _finish(root) -> None:
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
        y = (root.winfo_screenheight() - root.winfo_reqheight()) // 3
        root.geometry(f"+{x}+{y}")
        root.lift()
        root.focus_force()

    def unlock(self, vault: Path, keyfile: Optional[str], creating: bool) -> Optional[Credentials]:
        import tkinter as tk
        from tkinter import filedialog

        with self._one_at_a_time:
            t = self.t
            root, frame = self._window(t["create_head"] if creating else t["open"])
            tk.Label(frame, text=t["create_note"] if creating else t["open_note"], wraplength=380,
                     justify="left").pack(anchor="w", pady=(4, 10))
            tk.Label(frame, text=str(vault), fg="#5a6476").pack(anchor="w", pady=(0, 8))
            tk.Label(frame, text=t["password"]).pack(anchor="w")
            password = tk.Entry(frame, show="•", width=40)
            password.pack(anchor="w", fill="x")
            repeat = None
            if creating:
                tk.Label(frame, text=t["repeat"]).pack(anchor="w", pady=(8, 0))
                repeat = tk.Entry(frame, show="•", width=40)
                repeat.pack(anchor="w", fill="x")
            key_value = tk.StringVar(value=keyfile or "")
            make_key = tk.BooleanVar(value=False)
            row = tk.Frame(frame)
            if creating:
                tk.Checkbutton(frame, text=t["make_keyfile"], variable=make_key).pack(anchor="w", pady=(10, 0))
            else:
                tk.Label(frame, text=t["keyfile"]).pack(anchor="w", pady=(8, 0))
            row.pack(anchor="w", fill="x")
            tk.Entry(row, textvariable=key_value, width=32).pack(side="left", fill="x", expand=True)

            def choose():
                chosen = (filedialog.askdirectory(parent=root) if creating
                          else filedialog.askopenfilename(parent=root))
                if chosen:
                    key_value.set(chosen)
                    make_key.set(True)

            tk.Button(row, text=t["choose"], command=choose).pack(side="left", padx=(6, 0))
            message = tk.Label(frame, text="", fg="#b42318")
            message.pack(anchor="w", pady=(8, 0))
            result: List[Credentials] = []

            def submit(_event=None):
                secret = password.get()
                if creating:
                    if len(secret) < 8:
                        message.config(text=t["short"])
                        return
                    if secret != repeat.get():
                        message.config(text=t["differ"])
                        return
                    keypath = None
                    if make_key.get() and key_value.get():
                        keypath = str(Path(key_value.get()) / "infinito-llave.key")
                        Path(keypath).write_bytes(os.urandom(128))
                    result.append(Credentials(secret, keypath))
                else:
                    result.append(Credentials(secret, key_value.get().strip() or None))
                root.destroy()

            buttons = tk.Frame(frame)
            buttons.pack(anchor="e", pady=(10, 0))
            tk.Button(buttons, text=t["cancel"], command=root.destroy).pack(side="right", padx=(6, 0))
            tk.Button(buttons, text=t["create_button"] if creating else t["open_button"],
                      command=submit, default="active").pack(side="right")
            root.bind("<Return>", submit)
            root.bind("<Escape>", lambda _e: root.destroy())
            self._finish(root)
            password.focus_set()
            root.mainloop()
            return result[0] if result else None

    def new_secret(self, title: str) -> Optional[Tuple[str, str]]:
        import tkinter as tk

        with self._one_at_a_time:
            t = self.t
            root, frame = self._window(f"{t['new_head']}: {title}")
            tk.Label(frame, text=t["user"]).pack(anchor="w", pady=(10, 0))
            user = tk.Entry(frame, width=40)
            user.pack(anchor="w", fill="x")
            tk.Label(frame, text=t["secret"]).pack(anchor="w", pady=(8, 0))
            secret = tk.Entry(frame, show="•", width=40)
            secret.pack(anchor="w", fill="x")
            visible = tk.BooleanVar(value=False)
            tk.Checkbutton(frame, text=t["show"], variable=visible,
                           command=lambda: secret.config(show="" if visible.get() else "•")).pack(anchor="w")
            message = tk.Label(frame, text="", fg="#b42318")
            message.pack(anchor="w")
            result: List[Tuple[str, str]] = []

            def submit(_event=None):
                if not secret.get():
                    message.config(text=t["empty"])
                    return
                result.append((user.get().strip(), secret.get()))
                root.destroy()

            buttons = tk.Frame(frame)
            buttons.pack(anchor="e", pady=(10, 0))
            tk.Button(buttons, text=t["cancel"], command=root.destroy).pack(side="right", padx=(6, 0))
            tk.Button(buttons, text=t["save"], command=submit, default="active").pack(side="right")
            root.bind("<Return>", submit)
            root.bind("<Escape>", lambda _e: root.destroy())
            self._finish(root)
            user.focus_set()
            root.mainloop()
            return result[0] if result else None

    def show(self, title: str, username: str, password: str) -> None:
        import tkinter as tk

        with self._one_at_a_time:
            t = self.t
            root, frame = self._window(title)
            if username:
                tk.Label(frame, text=t["user_label"]).pack(anchor="w", pady=(10, 0))
                shown_user = tk.Entry(frame, width=40)
                shown_user.insert(0, username)
                shown_user.config(state="readonly")
                shown_user.pack(anchor="w", fill="x")
            tk.Label(frame, text=t["secret_label"]).pack(anchor="w", pady=(8, 0))
            shown = tk.Entry(frame, width=40, show="•")
            shown.insert(0, password)
            shown.config(state="readonly")
            shown.pack(anchor="w", fill="x")
            visible = tk.BooleanVar(value=False)
            tk.Checkbutton(frame, text=t["show"], variable=visible,
                           command=lambda: shown.config(show="" if visible.get() else "•")).pack(anchor="w")
            note = tk.Label(frame, text=t["closing"].format(n=self.SHOW_SECONDS), fg="#5a6476")
            note.pack(anchor="w", pady=(6, 0))
            state = {"left": self.SHOW_SECONDS, "copied": False}

            def clear_and_close():
                if state["copied"]:
                    try:
                        if root.clipboard_get() == password:
                            root.clipboard_clear()
                            root.clipboard_append("")
                    except tk.TclError:
                        pass
                root.destroy()

            def tick():
                state["left"] -= 1
                if state["left"] <= 0:
                    clear_and_close()
                    return
                key = "copied" if state["copied"] else "closing"
                note.config(text=t[key].format(n=state["left"]))
                root.after(1000, tick)

            def copy():
                root.clipboard_clear()
                root.clipboard_append(password)
                state["copied"] = True
                state["left"] = min(state["left"], self.COPY_SECONDS)
                note.config(text=t["copied"].format(n=state["left"]))

            buttons = tk.Frame(frame)
            buttons.pack(anchor="e", pady=(10, 0))
            tk.Button(buttons, text=t["close"], command=clear_and_close).pack(side="right", padx=(6, 0))
            tk.Button(buttons, text=t["copy"], command=copy, default="active").pack(side="right")
            root.protocol("WM_DELETE_WINDOW", clear_and_close)
            root.bind("<Escape>", lambda _e: clear_and_close())
            self._finish(root)
            root.after(1000, tick)
            root.mainloop()
