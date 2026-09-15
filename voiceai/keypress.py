"""
Cross-platform single-keypress reading (no Enter required).

Uses msvcrt on Windows, raw termios mode on macOS/Linux, and falls back to
a plain input() if neither is available (e.g. some non-interactive shells).
"""
import sys


def _read_key_windows():
    import msvcrt
    ch = msvcrt.getch()
    try:
        return ch.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _read_key_posix():
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def read_key(prompt=None):
    """Prints `prompt` (if given) and blocks until a single key is pressed."""
    if prompt:
        print(prompt, end="", flush=True)

    try:
        if sys.platform == "win32":
            key = _read_key_windows()
        else:
            key = _read_key_posix()
    except Exception:
        # Fallback for environments without a real terminal (e.g. some IDEs).
        key = (input() or " ")[:1]

    if prompt:
        print()  # move to a new line since the keypress itself doesn't echo

    return key
