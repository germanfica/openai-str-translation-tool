#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cli_colors.py
-------------
Constantes de color para CLI con autodeteccion:
- usa colorama si esta disponible (en Windows traduce ANSI)
- desactiva si NO_COLOR=1 o si la salida no es TTY
- permite forzar con FORCE_COLOR=1 o CLICOLOR_FORCE=1
"""

from __future__ import annotations
import os
import sys

def _truthy(env_name: str) -> bool:
    v = os.getenv(env_name)
    if v is None:
        return False
    return v not in ("0", "false", "False", "")

def _want_color(stream) -> bool:
    # fuerzan color
    if _truthy("FORCE_COLOR") or _truthy("CLICOLOR_FORCE"):
        return True
    # deshabilitan color
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("CLICOLOR") == "0":
        return False
    # TTY?
    try:
        return stream.isatty()
    except Exception:
        return False

# intenta colorama (mejor soporte en windows)
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()  # autoreset no necesario: exponemos RESET manual
    _has_colorama = True
except Exception:
    Fore = Style = None
    _has_colorama = False

_use_color = _want_color(sys.stderr)

if _use_color and _has_colorama:
    RED = Fore.RED
    YELLOW = Fore.YELLOW
    GREEN = Fore.GREEN
    CYAN = Fore.CYAN
    RESET = Style.RESET_ALL
elif _use_color and not _has_colorama:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
else:
    # sin color (p. ej. cuando se redirige a archivo)
    RED = ""
    YELLOW = ""
    GREEN = ""
    CYAN = ""
    RESET = ""

def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if _use_color else text
