"""The icon in the tray, with Windows' own calls (Shell_NotifyIconW) and no library: a coloured dot and a menu.

It lives in the same thread as the app's windows (tkinter): Tk's event loop on Windows hands every message of the
thread to its window, so this hidden window gets the clicks on the icon with no loop or thread of its own.
"""
import ctypes
from ctypes import wintypes
from typing import Callable, Dict, List, Optional, Tuple

user32, shell32, kernel32, gdi32 = ctypes.windll.user32, ctypes.windll.shell32, ctypes.windll.kernel32, ctypes.windll.gdi32

WM_APP = 0x8000
WM_TRAY = WM_APP + 7
WM_DESTROY = 0x0002
WM_LBUTTONUP, WM_RBUTTONUP, WM_CONTEXTMENU = 0x0202, 0x0205, 0x007B
NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP = 0x1, 0x2, 0x4
MF_STRING, MF_GRAYED, MF_CHECKED, MF_SEPARATOR = 0x0, 0x1, 0x8, 0x800
TPM_RIGHTBUTTON, TPM_RETURNCMD, TPM_NONOTIFY = 0x2, 0x100, 0x80
# The quick capture's shortcut (P-4): Ctrl+Alt and a letter, I unless the person chooses another.
WM_HOTKEY, HOTKEY_ID = 0x0312, 0x494E  # «IN»
MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x1, 0x2, 0x4000
SIZE = 32
# The dot's colour for each state: green, amber, red, and grey while nothing has been looked at yet.
COLOURS = {"ok": (52, 199, 89), "warn": (255, 176, 32), "fail": (230, 57, 70), "unknown": (150, 150, 160)}

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
def _types(function, restype, *argtypes) -> None:
    """Handles are 64-bit: without its types, ctypes would squeeze them into an int."""
    function.restype, function.argtypes = restype, list(argtypes)


_types(user32.DefWindowProcW, LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
_types(kernel32.GetModuleHandleW, wintypes.HMODULE, wintypes.LPCWSTR)
_types(user32.CreateIcon, wintypes.HICON, wintypes.HINSTANCE, ctypes.c_int, ctypes.c_int, wintypes.BYTE, wintypes.BYTE,
       ctypes.c_char_p, ctypes.c_char_p)
_types(user32.DestroyIcon, wintypes.BOOL, wintypes.HICON)
_types(user32.CreateWindowExW, wintypes.HWND, wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE,
       wintypes.LPVOID)
_types(user32.DestroyWindow, wintypes.BOOL, wintypes.HWND)
_types(user32.CreatePopupMenu, wintypes.HMENU)
_types(user32.AppendMenuW, wintypes.BOOL, wintypes.HMENU, wintypes.UINT, ctypes.c_size_t, wintypes.LPCWSTR)
_types(user32.DestroyMenu, wintypes.BOOL, wintypes.HMENU)
_types(user32.SetForegroundWindow, wintypes.BOOL, wintypes.HWND)
_types(user32.GetCursorPos, wintypes.BOOL, ctypes.POINTER(wintypes.POINT))
_types(user32.TrackPopupMenu, wintypes.BOOL, wintypes.HMENU, wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int,
       wintypes.HWND, ctypes.c_void_p)
_types(user32.RegisterHotKey, wintypes.BOOL, wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT)
_types(user32.UnregisterHotKey, wintypes.BOOL, wintypes.HWND, ctypes.c_int)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR)]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND), ("uID", wintypes.UINT),
                ("uFlags", wintypes.UINT), ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                ("szTip", wintypes.WCHAR * 128), ("dwState", wintypes.DWORD), ("dwStateMask", wintypes.DWORD),
                ("szInfo", wintypes.WCHAR * 256), ("uVersion", wintypes.UINT), ("szInfoTitle", wintypes.WCHAR * 64),
                ("dwInfoFlags", wintypes.DWORD), ("guidItem", ctypes.c_ubyte * 16), ("hBalloonIcon", wintypes.HICON)]


_types(user32.RegisterClassW, wintypes.ATOM, ctypes.POINTER(WNDCLASSW))
_types(shell32.Shell_NotifyIconW, wintypes.BOOL, wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW))


def dot(colour: Tuple[int, int, int]) -> int:
    """An icon: a round dot of that colour with a white ring, transparent around it."""
    blue_green_red = bytearray()
    mask = bytearray()
    middle, outer, ring = (SIZE - 1) / 2, SIZE / 2 - 1, SIZE / 2 - 4
    for y in range(SIZE):
        bits = 0
        for x in range(SIZE):
            distance = ((x - middle) ** 2 + (y - middle) ** 2) ** 0.5
            if distance <= ring:
                blue_green_red += bytes((colour[2], colour[1], colour[0], 255))
            elif distance <= outer:
                blue_green_red += bytes((255, 255, 255, 255))
            else:
                blue_green_red += bytes((0, 0, 0, 0))
                bits |= 1 << (SIZE - 1 - x)
        mask += bits.to_bytes(SIZE // 8, "big")
    return user32.CreateIcon(kernel32.GetModuleHandleW(None), SIZE, SIZE, 1, 32, bytes(mask), bytes(blue_green_red))


class Tray:
    """menu() gives the entries, as (label, action) with None for a separator and a leading «*» for a ticked one;
    an action of None is shown greyed out. on_click runs when the icon is clicked."""

    def __init__(self, tip: str, menu: Callable[[], List[Optional[Tuple[str, Optional[Callable]]]]],
                 on_click: Callable[[], None]):
        self.menu, self.on_click = menu, on_click
        self.on_hotkey: Optional[Callable[[], None]] = None
        self.icons: Dict[str, int] = {state: dot(colour) for state, colour in COLOURS.items()}
        self._procedure = WNDPROC(self._message)  # kept, or Windows would call freed memory
        instance = kernel32.GetModuleHandleW(None)
        kind = WNDCLASSW(lpfnWndProc=self._procedure, hInstance=instance, lpszClassName="INFINITODesktopTray")
        user32.RegisterClassW(ctypes.byref(kind))
        self.window = user32.CreateWindowExW(0, "INFINITODesktopTray", "INFINITO Desktop", 0, 0, 0, 0, 0, None, None,
                                             instance, None)
        self.data = NOTIFYICONDATAW(cbSize=ctypes.sizeof(NOTIFYICONDATAW), hWnd=self.window, uID=1,
                                    uFlags=NIF_MESSAGE | NIF_ICON | NIF_TIP, uCallbackMessage=WM_TRAY,
                                    hIcon=self.icons["unknown"], szTip=tip[:127])
        shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(self.data))

    def show(self, state: str, tip: str) -> None:
        self.data.hIcon, self.data.szTip = self.icons.get(state, self.icons["unknown"]), tip[:127]
        shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(self.data))

    def hotkey(self, key: str, action: Callable[[], None]) -> bool:
        """Ctrl+Alt+key anywhere in Windows runs action. False when another program already has it: then the app
        says so and the person chooses another letter in the settings."""
        user32.UnregisterHotKey(self.window, HOTKEY_ID)
        self.on_hotkey = action
        return bool(user32.RegisterHotKey(self.window, HOTKEY_ID, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT,
                                          ord(key.upper()[:1] or "I")))

    def close(self) -> None:
        user32.UnregisterHotKey(self.window, HOTKEY_ID)
        shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(self.data))
        user32.DestroyWindow(self.window)
        for icon in self.icons.values():
            user32.DestroyIcon(icon)

    def _popup(self) -> None:
        entries = self.menu()
        handle = user32.CreatePopupMenu()
        actions = {}
        for number, entry in enumerate(entries, start=1):
            if entry is None:
                user32.AppendMenuW(handle, MF_SEPARATOR, 0, None)
                continue
            label, action = entry
            flags = MF_STRING | (MF_CHECKED if label.startswith("*") else 0) | (MF_GRAYED if action is None else 0)
            user32.AppendMenuW(handle, flags, number, label.lstrip("*"))
            actions[number] = action
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        user32.SetForegroundWindow(self.window)  # or the menu would not close when clicking elsewhere
        chosen = user32.TrackPopupMenu(handle, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY, point.x, point.y, 0,
                                       self.window, None)
        user32.DestroyMenu(handle)
        if actions.get(chosen):
            actions[chosen]()

    def _message(self, window, message, wparam, lparam):
        if message == WM_HOTKEY and wparam == HOTKEY_ID and self.on_hotkey:
            self.on_hotkey()
            return 0
        if message == WM_TRAY:
            if lparam in (WM_RBUTTONUP, WM_CONTEXTMENU):
                self._popup()
            elif lparam == WM_LBUTTONUP:
                self.on_click()
            return 0
        return user32.DefWindowProcW(window, message, wparam, lparam)
