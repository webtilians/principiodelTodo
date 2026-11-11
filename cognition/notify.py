"""A notification on the person's computer when a review reaches their page: a click opens it, to read and decide.

On Windows, a toast through PowerShell (no module to install), whose click opens the page in the browser. Elsewhere,
or if anything fails, nothing happens: the memory page shows its own notice anyway.
"""
import base64
import os
import subprocess
from xml.sax.saxutils import escape, quoteattr

# PowerShell's own identity, which Windows lets any script show toasts under.
APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
NO_WINDOW = 0x08000000


def _toast_xml(title: str, text: str, url: str, button: str) -> str:
    return (f'<toast activationType="protocol" launch={quoteattr(url)} scenario="reminder">'
            f'<visual><binding template="ToastGeneric"><text>{escape(title)}</text><text>{escape(text)}</text>'
            f'</binding></visual><actions><action content={quoteattr(button)} activationType="protocol" '
            f'arguments={quoteattr(url)}/><action content="Luego" activationType="system" arguments="dismiss"/>'
            f'</actions></toast>')


def toast(title: str, text: str, url: str, button: str = "Leer y decidir") -> bool:
    if os.name != "nt":
        return False
    xml = _toast_xml(title, text, url, button).replace("'", "''")
    script = ("[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = "
              "WindowsRuntime] > $null; [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
              "ContentType = WindowsRuntime] > $null; $x = New-Object Windows.Data.Xml.Dom.XmlDocument; "
              f"$x.LoadXml('{xml}'); $t = New-Object Windows.UI.Notifications.ToastNotification $x; "
              f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{APP_ID}').Show($t)")
    try:
        done = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
                               "-EncodedCommand", base64.b64encode(script.encode("utf-16-le")).decode()],
                              capture_output=True, timeout=30, creationflags=NO_WINDOW)
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def review_waiting(server: str, proposals: int, reflection: bool) -> bool:
    parts = []
    if reflection:
        parts.append("ha dejado su reflexión")
    if proposals:
        parts.append(f"propone {proposals} pauta{'s' if proposals != 1 else ''} que esperan tu decisión")
    text = "El supervisor " + (" y ".join(parts) if parts else "ha terminado la revisión del día") + "."
    return toast("Motor cognitivo de INFINITO", text, f"{server.rstrip('/')}/memory?view=cognition")
