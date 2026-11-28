"""python -m infinito_desktop [--check | --login | --logout | --version]

    (nothing)   the app: an icon in the tray
    --check     one look at everything, printed as JSON, and out (for the CI and for finding out why)
    --login     sign this app in as a device: the browser opens INFINITO's page and you type your password there
    --logout    end its session, here and on the server
"""
import argparse
import json
import sys

from . import NAME, VERSION, look, places
from .session import Store


def _say(text: str) -> None:
    if sys.stdout is not None:  # pythonw and the built app have no console
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="infinito-desktop", description=NAME)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="print the state as JSON and exit")
    group.add_argument("--login", action="store_true", help="sign this app in as a device")
    group.add_argument("--logout", action="store_true", help="end this app's session")
    group.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    store = Store(places.home())
    if args.version:
        _say(f"{NAME} {VERSION}")
    elif args.check:
        found = look.everything(store, store.server() or places.SERVER)
        _say(json.dumps({"version": VERSION, "checks": [check.as_dict() for check in found]}, ensure_ascii=False,
                        indent=1))
    elif args.login:
        store.sign_in(places.SERVER)
        _say(f"{NAME} ha entrado en {places.SERVER} como dispositivo. Lo verás en tu página, en Apps.")
    elif args.logout:
        store.sign_out()
        _say("Sesión cerrada, aquí y en el servidor.")
    else:
        from .app import main as run

        return run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
