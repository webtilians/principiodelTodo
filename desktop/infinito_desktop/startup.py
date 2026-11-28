"""Starting with Windows: one value under the user's Run key, off unless the person turns it on in the settings."""
import sys
from pathlib import Path

KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE = "INFINITO Desktop"


def command() -> str:
    """What Windows runs at sign-in: the built executable, or pythonw with this checkout's launcher."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    launcher = Path(__file__).resolve().parents[1] / "infinito-desktop.pyw"
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    return f'"{pythonw if pythonw.exists() else sys.executable}" "{launcher}"'


def set_on(on: bool) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY, 0, winreg.KEY_SET_VALUE) as key:
        if on:
            winreg.SetValueEx(key, VALUE, 0, winreg.REG_SZ, command())
        else:
            try:
                winreg.DeleteValue(key, VALUE)
            except FileNotFoundError:
                pass
