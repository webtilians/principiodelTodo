"""Windows' data protection (DPAPI, CryptProtectData): what only this Windows user, on this PC, can open again.

The app's session is kept this way, never in plain text: another user of the PC, or the file copied elsewhere, opens
nothing. Only Windows has it; anywhere else these functions refuse.
"""
import ctypes
import os
from ctypes import wintypes

# Tied to this app, so another program of the same user cannot open the file by accident with an empty entropy.
ENTROPY = b"INFINITO Desktop session v1"
UI_FORBIDDEN = 0x1


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes):
    buffer = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _run(name: str, data: bytes) -> bytes:
    if os.name != "nt":
        raise OSError("DPAPI is only on Windows")
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    given, _keep = _blob(data)
    entropy, _keep_too = _blob(ENTROPY)
    out = _Blob()
    call = getattr(crypt32, name)
    if name == "CryptProtectData":
        done = call(ctypes.byref(given), "INFINITO Desktop", ctypes.byref(entropy), None, None, UI_FORBIDDEN,
                    ctypes.byref(out))
    else:
        done = call(ctypes.byref(given), None, ctypes.byref(entropy), None, None, UI_FORBIDDEN, ctypes.byref(out))
    if not done:
        raise OSError(f"{name} failed ({ctypes.GetLastError()})")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def protect(data: bytes) -> bytes:
    return _run("CryptProtectData", data)


def unprotect(data: bytes) -> bytes:
    return _run("CryptUnprotectData", data)
