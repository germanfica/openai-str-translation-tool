#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
from openai import OpenAI, APIError, RateLimitError  # pip install openai>=1.0.0

# ====================== Utils: archivo y BOM ======================

def _read_text_keep_bom(path: str) -> Tuple[str, bool, str]:
    with open(path, 'rb') as f:
        raw = f.read()
    bom = raw.startswith(b'\xef\xbb\xbf')
    text = raw.decode('utf-8-sig' if bom else 'utf-8', errors='replace')
    newline = '\r\n' if '\r\n' in text and text.count('\r\n') >= text.count('\n') else '\n'
    return text, bom, newline

def _write_text_with_bom(path: str, text: str, bom: bool) -> None:
    data = text.encode('utf-8')
    if bom:
        data = b'\xef\xbb\xbf' + data
    with open(path, 'wb') as f:
        f.write(data)

def _derive_out_path(in_path: str, suffix: str, forced_ext: Optional[str] = None) -> str:
    base, ext = os.path.splitext(in_path)
    if forced_ext:
        ext = forced_ext if forced_ext.startswith('.') else f'.{forced_ext}'
    elif not ext:
        ext = '.str'
    return f'{base}{suffix}{ext}'

# ====================== Placeholders a proteger ======================

_PLACEHOLDER_PATTERNS = [
    re.compile(r'%(?:\d+\$)?[+\-#0 ]*(?:\d+|\*)?(?:\.(?:\d+|\*))?(?:hh|h|l|ll|z|j|t|L)?[diuoxXfFeEgGaAcsp%]'),  # printf
    re.compile(r'\{[A-Za-z0-9_:\.\-]+\}'),     # {0} {name} {0:N2}
    re.compile(r'\$[A-Za-z0-9_]+\$'),          # $NAME$
    re.compile(r'%[A-Za-z0-9_]+%'),            # %PLAYER%
    re.compile(r'</?[^>\s]+(?:\s+[^>]*?)?>'),  # etiquetas <b>, </b>, <i>...
    re.compile(r'\\[nrt"\\]'),                 # escapes \n, \t, \", \\
]

def freeze_placeholders(s: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    idx = 0
    def repl(m):
        nonlocal idx
        token = m.group(0)
        key = f'__PH_{idx}__'
        mapping[key] = token
        idx += 1
        return key
    frozen = s
    for rx in _PLACEHOLDER_PATTERNS:
        frozen = rx.sub(repl, frozen)
    return frozen, mapping

def unfreeze_placeholders(s: str, mapping: Dict[str, str]) -> str:
    out = s
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out

def validate_placeholders(orig: str, translated: str) -> bool:
    counts: Dict[str, int] = {}
    for rx in _PLACEHOLDER_PATTERNS:
        for m in rx.finditer(orig):
            counts[m.group(0)] = counts.get(m.group(0), 0) + 1
    for tok, n in counts.items():
        if translated.count(tok) != n:
            return False
    return True

# ====================== OpenAI ======================

def get_client(explicit_key: Optional[str] = None) -> OpenAI:
    api_key = explicit_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: seteá la variable de entorno OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)

_TRANSLATOR_SYS_PROMPT = (
    "Sos un traductor EN->ES (español latino neutro). "
    "Devolvé solo el texto traducido, sin comillas ni comentarios. "
    "Respetá exactamente los placeholders (por ej. __PH_0__, %s, {0}, $NAME$, etiquetas <b>...</b>, secuencias \\n). "
    "No agregues ni quites espacios alrededor de placeholders. "
    "Usá español latino neutro, claro y natural."
)

def translate_segment(client: OpenAI, model: str, text_en: str) -> str:
    frozen, mapping = freeze_placeholders(text_en)
    delay = 1.0
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _TRANSLATOR_SYS_PROMPT},
                    {"role": "user", "content": frozen},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            out = (resp.choices[0].message.content or "").strip()
            out_unfrozen = unfreeze_placeholders(out, mapping)
            if not validate_placeholders(text_en, out_unfrozen):
                # última red: si rompió placeholders, devolvé el original
                out_unfrozen = text_en
            return out_unfrozen
        except (RateLimitError, APIError):
            if attempt == 4:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    return text_en

# ====================== Detección de formato ======================

_TIME_RE = re.compile(r'^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*$')

def detect_format(text: str) -> str:
    # muy simple: si vemos una línea de tiempos SRT y bloques con índices, lo tratamos como SRT
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        if lines[i].strip().isdigit() and _TIME_RE.match(lines[i+1].strip()):
            return 'srt'
    # caso contrario, asumimos 'str' (clave=valor)
    return 'str'

# ====================== STR (clave=valor) ======================

_COMMENT_RE = re.compile(r'^\s*(//|#|;)\s?.*$')
_EMPTY_RE   = re.compile(r'^\s*$')
_QUOTED_PAIR_RE = re.compile(r'^\s*"(?P<key>(?:\\"|[^"])*)"\s*"(?P<val>(?:\\"|[^"])*)"\s*$')
_EQUALS_PAIR_RE = re.compile(r'^\s*(?P<key>[A-Za-z0-9_.\-]+)\s*=\s*(?P<val>".*?"|.+?)\s*$')
_COLON_PAIR_RE  = re.compile(r'^\s*(?P<key>[A-Za-z0-9_.\-]+)\s*:\s*(?P<val>".*?"|.+?)\s*$')

@dataclass
class KVLine:
    key: str
    value: str
    trailing_comment: str = ''

@dataclass
class RawLine:
    content: str

ParsedLine = Tuple[str, Union[KVLine, RawLine]]

def _split_trailing_comment(s: str) -> Tuple[str, str]:
    in_q = False
    for i, ch in enumerate(s):
        if ch == '"':
            in_q = not in_q
        elif not in_q and i+1 < len(s) and s[i:i+2] == '//':
            return s[:i].rstrip(), s[i:]
        elif not in_q and ch in ['#', ';']:
            if i == 0 or s[i-1].isspace():
                return s[:i].rstrip(), s[i:]
    return s.rstrip(), ''

def parse_str(text: str) -> List[ParsedLine]:
    out: List[ParsedLine] = []
    for line in text.splitlines():
        if _COMMENT_RE.match(line) or _EMPTY_RE.match(line):
            out.append(('raw', RawLine(line)))
            continue
        core, trailing = _split_trailing_comment(line)

        m = _QUOTED_PAIR_RE.match(core)
        if m:
            key = m.group('key').replace('\\"', '"')
            val = m.group('val').replace('\\"', '"')
            out.append(('kv', KVLine(key=key, value=val, trailing_comment=trailing)))
            continue

        matched = False
        for rx in (_EQUALS_PAIR_RE, _COLON_PAIR_RE):
            m = rx.match(core)
            if m:
                key = m.group('key')
                val = m.group('val')
                if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
                    val = val[1:-1].replace('\\"', '"')
                out.append(('kv', KVLine(key=key, value=val, trailing_comment=trailing)))
                matched = True
                break
        if not matched:
            out.append(('raw', RawLine(line)))
    return out

def render_str(lines: List[ParsedLine], newline: str = '\n') -> str:
    rendered: List[str] = []
    for typ, obj in lines:
        if typ == 'raw':
            rendered.append(obj.content)
        else:
            v = obj.value.replace('"', '\\"')
            line = f'{obj.key} = "{v}"'
            if obj.trailing_comment:
                line += ' ' + obj.trailing_comment
            rendered.append(line)
    return newline.join(rendered) + newline

# ====================== SRT ======================

@dataclass
class SrtBlock:
    index: int
    start: str
    end: str
    text_lines: List[str]

def parse_srt(text: str) -> List[SrtBlock]:
    lines = text.splitlines()
    n = len(lines)
    i = 0
    blocks: List[SrtBlock] = []

    while i < n:
        # saltar blanks
        while i < n and lines[i].strip() == '':
            i += 1
        if i >= n:
            break

        # índice
        if not lines[i].strip().isdigit():
            # línea suelta fuera de bloque; ignoramos
            i += 1
            continue
        idx = int(lines[i].strip()); i += 1
        if i >= n:
            break

        # tiempos
        if not _TIME_RE.match(lines[i].strip()):
            # formato inesperado; intentamos saltar bloque
            i += 1
            continue
        time_line = lines[i].strip(); i += 1
        start, end = [s.strip() for s in time_line.split('-->')]

        # texto (hasta blank)
        text_lines: List[str] = []
        while i < n and lines[i].strip() != '':
            text_lines.append(lines[i])
            i += 1

        blocks.append(SrtBlock(index=idx, start=start, end=end, text_lines=text_lines))

        # saltar blank que separa bloques
        while i < n and lines[i].strip() == '':
            i += 1

    return blocks

def render_srt(blocks: List[SrtBlock], newline: str = '\n') -> str:
    parts: List[str] = []
    for b in blocks:
        parts.append(str(b.index))
        parts.append(f'{b.start} --> {b.end}')
        parts.extend(b.text_lines if b.text_lines else [''])
        parts.append('')  # separador entre bloques
    text = newline.join(parts)
    # quitar un newline extra final si quedó doble
    return text.rstrip('\r\n') + newline

# ====================== Progreso ======================

def print_progress(i: int, total: int):
    width = 30
    filled = int(width * i / max(total, 1))
    bar = '#' * filled + '-' * (width - filled)
    msg = f'\r[{bar}] {i}/{total}'
    print(msg, end='', file=sys.stderr, flush=True)
    if i == total:
        print(file=sys.stderr)

# ====================== Orquestación principal ======================

def should_translate(line: str) -> bool:
    # traduce solo si hay letras ASCII; deja pasar signos y números sin traducir
    has_ascii_letters = re.search(r'[A-Za-z]', line) is not None
    return has_ascii_letters

def process_str_file(text: str, newline: str, client: OpenAI, model: str,
                     save_map: Optional[str], show_progress: bool) -> Tuple[str, List[Dict]]:
    parsed = parse_str(text)

    # construir lista de segmentos a traducir
    segments: List[Tuple[int, str]] = []  # (posicion en parsed, texto)
    for pos, (typ, obj) in enumerate(parsed):
        if typ == 'kv':
            val = obj.value.strip()
            if val and should_translate(val):
                segments.append((pos, obj.value))

    total = len(segments)
    mapping_out: List[Dict] = []
    done = 0

    for pos, src in segments:
        tgt = translate_segment(client, model, src)
        # reinyectar
        typ, obj = parsed[pos]
        assert typ == 'kv'
        parsed[pos] = ('kv', KVLine(key=obj.key, value=tgt, trailing_comment=obj.trailing_comment))
        mapping_out.append({'type': 'str', 'index_in_parsed': pos, 'source': src, 'translated': tgt})
        done += 1
        if show_progress:
            print_progress(done, total)

    if save_map:
        with open(save_map, 'w', encoding='utf-8') as f:
            json.dump(mapping_out, f, ensure_ascii=False, indent=2)

    return render_str(parsed, newline=newline), mapping_out

def process_srt_file(text: str, newline: str, client: OpenAI, model: str,
                     save_map: Optional[str], show_progress: bool) -> Tuple[str, List[Dict]]:
    blocks = parse_srt(text)
    segments: List[Tuple[int, int, str]] = []  # (block_idx, line_idx, text)

    for bi, b in enumerate(blocks):
        for li, line in enumerate(b.text_lines):
            if line.strip() and should_translate(line):
                segments.append((bi, li, line))

    total = len(segments)
    mapping_out: List[Dict] = []
    done = 0

    for bi, li, src in segments:
        tgt = translate_segment(client, model, src)
        blocks[bi].text_lines[li] = tgt
        mapping_out.append({'type': 'srt', 'block': bi, 'line': li, 'source': src, 'translated': tgt})
        done += 1
        if show_progress:
            print_progress(done, total)

    if save_map:
        with open(save_map, 'w', encoding='utf-8') as f:
            json.dump(mapping_out, f, ensure_ascii=False, indent=2)

    return render_srt(blocks, newline=newline), mapping_out

def process_file(in_path: str, out_suffix: str, dry_run: bool, model: str, api_key: Optional[str],
                 force_format: str, save_map: Optional[str], show_progress: bool) -> str:
    text, had_bom, newline = _read_text_keep_bom(in_path)
    fmt = detect_format(text) if force_format == 'auto' else force_format.lower()
    client = get_client(api_key)

    if fmt not in ('str', 'srt', 'auto'):
        raise ValueError("Formato inválido. Usá --format auto|str|srt")

    if fmt == 'srt' or (fmt == 'auto' and detect_format(text) == 'srt'):
        out_text, _ = process_srt_file(text, newline, client, model, save_map, show_progress)
        out_path = _derive_out_path(in_path, out_suffix, forced_ext='.srt')
    else:
        out_text, _ = process_str_file(text, newline, client, model, save_map, show_progress)
        out_path = _derive_out_path(in_path, out_suffix, forced_ext=None)

    if dry_run:
        print(out_text)
        return out_path

    _write_text_with_bom(out_path, out_text, had_bom)
    return out_path

# ====================== CLI ======================

def main():
    parser = argparse.ArgumentParser(
        description="Traductor de archivos .srt y .str EN->ES (latino) preservando estructura y placeholders.")
    parser.add_argument('input', help="Ruta al archivo .srt/.str origen en inglés.")
    parser.add_argument('--suffix', default='_es', help="Sufijo para el archivo de salida (default: _es).")
    parser.add_argument('--model', default='gpt-4o-mini',
                        help="Modelo de OpenAI a usar (default: gpt-4o-mini; usá el que tengas disponible).")
    parser.add_argument('--format', default='auto', choices=['auto', 'srt', 'str'],
                        help="Forzar formato de entrada (por defecto auto).")
    parser.add_argument('--api-key', default=None, help="Clave de API (si no, usa OPENAI_API_KEY).")
    parser.add_argument('--dry-run', action='store_true', help="Imprime el resultado en stdout sin escribir archivo.")
    parser.add_argument('--map-json', default=None, help="Guardar un JSON con el mapeo de traducciones.")
    parser.add_argument('--progress', action='store_true', help="Mostrar barra de progreso.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: no se encontró el archivo: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        out_path = process_file(
            in_path=args.input,
            out_suffix=args.suffix,
            dry_run=args.dry_run,
            model=args.model,
            api_key=args.api_key,
            force_format=args.format,
            save_map=args.map_json,
            show_progress=args.progress
        )
        if args.dry_run:
            print("\n--- fin (dry-run) ---")
        else:
            print(f"OK: guardado '{out_path}'")
            if args.map_json:
                print(f"OK: mapeo guardado en '{args.map_json}'")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
