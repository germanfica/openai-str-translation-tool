#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
precheck_core.py
----------------
Pre-cálculo de costo para traducir .str (y .srt) usando EXACTAMENTE la misma
lógica de selección de segmentos que usa translate_core (should_translate, parse_*).

No define CLI. Se importa y ejecuta desde srt_translate.py, por ejemplo:

    from precheck_core import run_cost_precheck_for_core
    run_cost_precheck_for_core(
        in_path=args.input,
        model=args.model,
        fmt_forced=args.format,          # 'auto'|'str'|'srt'
        batch_size=max(1, args.batch_size),
        completion_multiplier=1.0,       # salida ≈ entrada
        section='Text tokens',
        tier='Standard',
        pricing_json_path='openai_pricing.json',
        pricing_md_path='openai_pricing.md',
        assume_yes=False                 # siempre preguntar y/N
    )

Si el usuario cancela, el proceso sale con código 130.
Si aprueba (o assume_yes=True), retorna el resumen (dict) y continúa.
"""

from __future__ import annotations
import os
import re
import sys
import json
import math
from typing import List, Tuple, Dict, Optional

# Reutilizamos tu core para NO desalinearnos de cómo elegís segmentos
from translate_core import (
    detect_format,
    parse_str,
    parse_srt,
    should_translate,
    _TRANSLATOR_SYS_PROMPT,
    _BATCH_SYS_PROMPT,
)

# Tu reporte/catálogo local de precios
from cost_report import estimate_operation_cost

# ===== Colores (colorama si está; fallback ANSI) =====
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
    RED = Fore.RED
    YELLOW = Fore.YELLOW
    GREEN = Fore.GREEN
    CYAN = Fore.CYAN
    RESET = Style.RESET_ALL
except Exception:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    RESET = "\033[0m"

# ===== Tokenizador (tiktoken opcional; si no, heurística 1 token ~ 4 chars) =====
def _choose_encoding_name(model: str) -> str:
    m = model.lower()
    if "4o" in m or "gpt-4.1" in m or "gpt-4.2" in m or "o3" in m:
        return "o200k_base"
    return "cl100k_base"

def _load_encoder(model: str):
    try:
        import tiktoken
        return tiktoken.get_encoding(_choose_encoding_name(model))
    except Exception:
        return None

def _count_tokens(text: str, model: str, enc=None) -> int:
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / 4))

# ===== E/S mínimo =====
def _read_text_keep_bom(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8", errors="replace")

# ===== Recolección de segmentos que efectivamente se envían al LLM =====
def _collect_items_str(text: str) -> List[str]:
    items: List[str] = []
    for typ, obj in parse_str(text):
        if typ == "kv":
            val = obj.value.strip()
            if val and should_translate(val):
                items.append(obj.value)
    return items

def _collect_items_srt(text: str) -> List[str]:
    # Cada item es el bloque completo (lineas unidas con \n), igual que en translate_core
    items: List[str] = []
    for b in parse_srt(text):
        if any(line.strip() and should_translate(line) for line in b.text_lines):
            items.append("\n".join(b.text_lines))
    return items

def _group_for_batch(items: List[str], batch_size: int) -> List[str]:
    if batch_size <= 1:
        return items
    grouped: List[str] = []
    for i in range(0, len(items), batch_size):
        group = items[i : i + batch_size]
        grouped.append(json.dumps(group, ensure_ascii=False))
    return grouped

# ===== Métricas auxiliares (palabras y chars) solo de lo que se traduce =====
_WORD_RE = re.compile(r"[A-Za-z]+")

def _count_words(items: List[str]) -> int:
    return sum(len(_WORD_RE.findall(s)) for s in items)

def _count_chars(items: List[str]) -> int:
    return sum(len(s) for s in items)

# ===== Render del cuadro WARNING =====
def _print_warning_box(summary: Dict) -> None:
    title = f"WARNING: costo estimado para traducir '{os.path.basename(summary['file'])}'"
    line = "-" * max(60, len(title) + 4)
    print()
    print(f"{YELLOW}{line}{RESET}")
    print(f"{YELLOW}| {RED}{title}{YELLOW} |{RESET}")
    print(f"{YELLOW}{line}{RESET}")
    print(f"{CYAN}Modelo:{RESET} {summary['model']}  {CYAN}Tier:{RESET} {summary['tier']}  {CYAN}Seccion:{RESET} {summary['section']}")
    print(f"{CYAN}Formato:{RESET} {summary['format']}  {CYAN}Batch size:{RESET} {summary['batch_size']}  {CYAN}Requests:{RESET} {summary['requests']}")
    print(f"{CYAN}Items traducibles:{RESET} {summary['items_translated']}  {CYAN}Palabras (est.):{RESET} {summary['word_count']}  {CYAN}Chars:{RESET} {summary['char_count']}")
    print(f"{CYAN}Prompt tokens (est.):{RESET} {summary['prompt_tokens']}")
    print(f"{CYAN}Completion tokens (est.):{RESET} {summary['completion_tokens']}")
    print(f"{CYAN}Total tokens (est.):{RESET} {summary['total_tokens']}")
    print(f"{CYAN}Overhead/req:{RESET} {summary['overhead_per_request']}  {CYAN}Incluye system:{RESET} {summary['include_system']}")
    print(f"{YELLOW}===> Costo aproximado (USD): {RED}${summary['estimated_cost_usd']:.6f}{YELLOW}{RESET}")
    print()

def _ask_confirmation(default_no: bool = True) -> bool:
    prompt = f"{YELLOW}Proceder con la traduccion? {RESET}[y/N]: " if default_no else f"{YELLOW}Proceder con la traduccion? {RESET}[Y/n]: "
    try:
        ans = input(prompt).strip()
    except EOFError:
        ans = ""
    except KeyboardInterrupt:
        print(f"\n{RED}Cancelado por el usuario.{RESET}", file=sys.stderr)
        sys.exit(130)
    if default_no:
        return ans.lower() == "y"
    else:
        return ans == "" or ans.lower() == "y"

# ===== Cálculo principal (sin I/O de red) =====
def compute_precheck_summary_for_core(
    *,
    in_path: str,
    model: str,
    fmt_forced: str = "auto",          # 'auto'|'str'|'srt'
    batch_size: int = 1,
    completion_multiplier: float = 1.0, # salida ≈ entrada
    section: str = "Text tokens",
    tier: str = "Standard",
    pricing_json_path: str = "openai_pricing.json",
    pricing_md_path: str = "openai_pricing.md",
) -> Dict:
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"No existe el archivo: {in_path}")

    text = _read_text_keep_bom(in_path)
    fmt = detect_format(text) if fmt_forced == "auto" else fmt_forced.lower()
    if fmt not in ("str", "srt"):
        fmt = "str"

    # Segmentos que realmente va a ver el modelo (alinea con translate_core)
    originals = _collect_items_srt(text) if fmt == "srt" else _collect_items_str(text)
    grouped = _group_for_batch(originals, max(1, batch_size))

    # Token count
    enc = _load_encoder(model)
    sys_prompt = _BATCH_SYS_PROMPT if batch_size > 1 else _TRANSLATOR_SYS_PROMPT
    sys_tokens_per_req = _count_tokens(sys_prompt, model, enc)
    overhead_per_req = 6  # mismo valor que usamos en otros prechecks

    reqs = len(grouped)
    user_tokens_per_req = [_count_tokens(s, model, enc) for s in grouped]

    prompt_tokens = sum(user_tokens_per_req) + reqs * (sys_tokens_per_req + overhead_per_req)
    completion_tokens = max(0, int(round(sum(user_tokens_per_req) * completion_multiplier)))

    # Costo usando tu catálogo local (via cost_report)
    details = estimate_operation_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_input_tokens=0,
        preferred_section=section,
        preferred_tier=tier,
        pricing_json_path=pricing_json_path,
        pricing_md_path=pricing_md_path,
    )

    summary: Dict = {
        "file": in_path,
        "format": fmt,
        "batch_size": max(1, batch_size),
        "requests": reqs,
        "items_translated": len(originals),
        "word_count": _count_words(originals),
        "char_count": _count_chars(originals),
        "model": details.model,          # puede normalizar nombre segun catálogo
        "section": details.section,
        "tier": details.tier,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "include_system": True,
        "overhead_per_request": overhead_per_req,
        "estimated_cost_usd": details.cost_total,
        "pricing_source": details.pricing_source,
        "pricing_file": details.pricing_file,
        "timestamp_iso": details.timestamp_iso,
    }
    return summary

# ===== Orquestador: imprime WARNING, pregunta y/N y decide seguir o abortar =====
def run_cost_precheck_for_core(
    *,
    in_path: str,
    model: str,
    fmt_forced: str = "auto",
    batch_size: int = 1,
    completion_multiplier: float = 1.0,
    section: str = "Text tokens",
    tier: str = "Standard",
    pricing_json_path: str = "openai_pricing.json",
    pricing_md_path: str = "openai_pricing.md",
    assume_yes: bool = False
) -> Dict:
    """
    Calcula, muestra el WARNING y solicita confirmación. Si el usuario cancela,
    escribe un mensaje y termina el proceso con exit code 130.
    Devuelve el resumen (dict) si se aprueba (o assume_yes=True).
    """
    summary = compute_precheck_summary_for_core(
        in_path=in_path,
        model=model,
        fmt_forced=fmt_forced,
        batch_size=batch_size,
        completion_multiplier=completion_multiplier,
        section=section,
        tier=tier,
        pricing_json_path=pricing_json_path,
        pricing_md_path=pricing_md_path,
    )

    _print_warning_box(summary)

    if assume_yes:
        print(f"{GREEN}Aprobado automaticamente (--assume-yes interno).{RESET}")
        return summary

    if not _ask_confirmation(default_no=True):
        print(f"{RED}Cancelado por el usuario.{RESET}", file=sys.stderr)
        sys.exit(130)

    print(f"{GREEN}Aprobado por el usuario.{RESET}")
    return summary
