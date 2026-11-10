"""Independent, fail-closed verifier for ONE repair recipe; never eval model code."""
import ast
import inspect
import logging
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from src.infinito3 import remote_apps

EXPECTED_ATTACKS = {"A10", "A11", "A13", "A14"}


def approved_replacement(source):
    """Only the exact trusted function AST is accepted, including signature/decorators.

    Formatting may differ. Imports, calls, extra statements, altered defaults and
    even semantically equivalent alternative implementations fail closed.
    """
    try:
        actual = ast.dump(ast.parse(source), include_attributes=False)
        expected = ast.dump(ast.parse(inspect.getsource(remote_apps.allowed)), include_attributes=False)
        return actual == expected
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return False


def verify(answer):
    if answer["code_path"] != "src/infinito3/remote_apps.py" or not approved_replacement(answer["replacement"]):
        return {"status": "rejected", "reason": "outside trusted restoration recipe; no code executed"}
    from security_lab.arena import recorder

    # This runs only reviewed repository code with synthetic accounts. It is not
    # an OS sandbox for arbitrary generated patches (which are never executed).
    previous = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with recorder.loopback_only(), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            broken = recorder.play("levels-off")
            restored = recorder.play(None)
    finally:
        logging.disable(previous)
    through = {e["id"] for e in broken if not e["held"]}
    fixed_ids = {e["id"] for e in restored}
    controls = {
        "read_recall": remote_apps.allowed("read", "recall"),
        "add_remember": remote_apps.allowed("add", "remember"),
        "all_forget": remote_apps.allowed("all", "forget"),
        "read_cannot_remember": not remote_apps.allowed("read", "remember"),
        "add_cannot_forget": not remote_apps.allowed("add", "forget"),
        "add_cannot_vault": not remote_apps.allowed("add", "vault_list"),
    }
    passed = (EXPECTED_ATTACKS <= through and EXPECTED_ATTACKS <= fixed_ids
              and all(e["held"] for e in restored) and all(controls.values()))
    return {"status": "verified" if passed else "failed", "scope": "synthetic levels-off restoration only",
            "faulty_through": sorted(through), "restored_attacks": len(restored),
            "restored_through": [e["id"] for e in restored if not e["held"]],
            "controls": controls, "faulty_events": broken, "restored_events": restored,
            "generated_tests_executed": False, "arbitrary_patch_executed": False}
