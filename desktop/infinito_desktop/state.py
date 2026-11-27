"""What the app keeps besides its session, in plain JSON in its folder because none of it is secret: its settings, what
each check last said and the day each was last notified, and its device id (src/infinito3/remote_device_api.py)."""
import json
import uuid
from pathlib import Path

FILE = "state.json"
DEFAULTS = {
    "paused": False,
    "start_with_windows": False,
    # For the local cognitive node (phase 5), which only reads them: nothing heavy runs in phase 1.
    "budget": {"cpu_percent": 25, "ram_gb": 4, "gpu_percent": 50, "never_on_battery": True},
    "seen": {},
    "told": {},
}


class State:
    def __init__(self, folder: Path):
        self.path = Path(folder) / FILE
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = {}
        self.data = {**json.loads(json.dumps(DEFAULTS)), **(loaded if isinstance(loaded, dict) else {})}
        if not self.data.get("device_id"):
            self.data["device_id"] = str(uuid.uuid4())
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        temporary.replace(self.path)

    def __getitem__(self, name):
        return self.data[name]

    def __setitem__(self, name, value) -> None:
        self.data[name] = value
        self.save()
