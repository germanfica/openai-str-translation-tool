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
from cost_report import estimate_operation_cost, render_cost_summary

total_prompt_tokens = 0
total_completion_tokens = 0

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
    "Si el texto tiene varios renglones, devolvé la misma cantidad de renglones y en el mismo orden. "
    "No elimines guiones de diálogo al inicio de línea. "
    "Usá signos de apertura ¿ y ¡ cuando corresponda. "
    "Usá español latino neutro, claro y natural."
)

_BATCH_SYS_PROMPT = (
    "Sos un traductor EN->ES (español latino neutro). "
    "Vas a recibir un array JSON de strings; cada string es un bloque completo de subtítulo. "
    "Debés responder EXCLUSIVAMENTE con un array JSON de strings de la MISMA longitud. "
    "Cada item traducido debe preservar exactamente la cantidad de renglones (\\n) y su orden respecto del item original. "
    "Respetá exactamente los placeholders (por ej. __PH_0__, %s, {0}, $NAME$, etiquetas <b>...</b>, secuencias \\n). "
    "No agregues ni quites espacios alrededor de placeholders ni cambies el número de elementos. "
    "No incluyas texto adicional, ni bloques de código, ni comentarios. "
    "Usá signos de apertura ¿ y ¡ cuando corresponda y español latino neutro, claro y natural."
)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith('```'):
        # Intentar remover fences tipo ```json ... ```
        s = s.strip('`')
        # a veces el primer renglón es 'json' o 'JSON'
        if '\n' in s:
            first, rest = s.split('\n', 1)
            if first.lower() in ('json', '```json'):
                return rest.strip()
    return s

def _add_usage_from_response(resp) -> None:
    """
    Extrae prompt_tokens y completion_tokens de la respuesta 'resp' de forma tolerante
    y los suma a los acumuladores globales.
    """
    global total_prompt_tokens, total_completion_tokens
    try:
        usage = None
        # varios SDKs/formatos: puede ser atributo o dict
        if hasattr(resp, "usage"):
            usage = resp.usage
        elif isinstance(resp, dict):
            usage = resp.get("usage")
        # si no hay usage, salir
        if not usage:
            return
        # obtener valores de forma tolerante (attr o key)
        def _get(u, key):
            if hasattr(u, key):
                return getattr(u, key)
            if isinstance(u, dict):
                return u.get(key)
            return None

        p = _get(usage, "prompt_tokens")
        c = _get(usage, "completion_tokens")
        # fallback: algunos SDKs usan 'total_tokens' o nombres distintos; no lo tomamos aquí
        p = int(p) if p is not None else 0
        c = int(c) if c is not None else 0

        total_prompt_tokens += p
        total_completion_tokens += c
    except Exception:
        # Nunca queremos que fallos en extracción rompan la traducción.
        return

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
            # intentar acumular usage (si lo tiene)
            _add_usage_from_response(resp)

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

def translate_segments_batch(client: OpenAI, model: str, texts_en: List[str]) -> List[str]:
    # Congela placeholders por item
    frozens: List[str] = []
    mappings: List[Dict[str, str]] = []
    for t in texts_en:
        f, m = freeze_placeholders(t)
        frozens.append(f)
        mappings.append(m)

    payload = json.dumps(frozens, ensure_ascii=False)
    delay = 1.0
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _BATCH_SYS_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=0.2,
                max_tokens=4096,
            )

            # acumular usage de la llamada batch (si existe)
            _add_usage_from_response(resp)

            raw = (resp.choices[0].message.content or "").strip()
            raw = _strip_code_fences(raw)
            # a veces el modelo devuelve texto extra, tratamos de aislar el primer array
            start = raw.find('[')
            end = raw.rfind(']')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("respuesta sin JSON válido")
            arr = json.loads(raw[start:end+1])
            if not isinstance(arr, list) or len(arr) != len(frozens):
                raise ValueError("longitud de array inválida")

            # Descongelar y validar placeholders por item
            out: List[str] = []
            for orig, tr_frozen, mapping in zip(texts_en, arr, mappings):
                tr = tr_frozen if isinstance(tr_frozen, str) else str(tr_frozen)
                tr = unfreeze_placeholders(tr, mapping)
                if not validate_placeholders(orig, tr):
                    tr = orig
                out.append(tr)
            return out
        except (RateLimitError, APIError, json.JSONDecodeError, ValueError):
            if attempt == 4:
                # fallback: item por item
                return [translate_segment(client, model, t) for t in texts_en]
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    return texts_en

# helper: traducir un bloque completo preservando cantidad de saltos de línea
def translate_block_preserving_newlines(client: OpenAI, model: str, lines: List[str]) -> List[str]:
    joined = '\n'.join(lines)
    translated = translate_segment(client, model, joined)
    out_lines = translated.split('\n')
    return out_lines

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

def _render_srt_block(block: SrtBlock, newline: str) -> str:
    parts: List[str] = [str(block.index), f'{block.start} --> {block.end}']
    parts.extend(block.text_lines if block.text_lines else [''])
    parts.append('')  # separador de bloque
    return newline.join(parts)

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

def _chunk_indices(idxs: List[int], batch_size: int) -> List[List[int]]:
    return [idxs[i:i+batch_size] for i in range(0, len(idxs), max(1, batch_size))]

def process_str_file(text: str, newline: str, client: OpenAI, model: str,
                     save_map: Optional[str], show_progress: bool,
                     show_preview: bool = False, batch_size: int = 1) -> Tuple[str, List[Dict]]:
    parsed = parse_str(text)

    # construir lista de segmentos a traducir
    segments: List[Tuple[int, str]] = []
    for pos, (typ, obj) in enumerate(parsed):
        if typ == 'kv':
            val = obj.value.strip()
            if val and should_translate(val):
                segments.append((pos, obj.value))

    total = len(segments)
    mapping_out: List[Dict] = []
    done = 0

    if batch_size <= 1:
        for pos, src in segments:
            tgt = translate_segment(client, model, src)
            typ, obj = parsed[pos]
            assert typ == 'kv'
            parsed[pos] = ('kv', KVLine(key=obj.key, value=tgt, trailing_comment=obj.trailing_comment))
            mapping_out.append({'type': 'str', 'index_in_parsed': pos, 'source': src, 'translated': tgt})
            done += 1
            if show_progress:
                print_progress(done, total)
            if show_preview:
                v_escaped = tgt.replace('"', '\\"')
                preview_line = f'{obj.key} = "{v_escaped}"'
                if obj.trailing_comment:
                    preview_line += ' ' + obj.trailing_comment
                print(preview_line, flush=True)
    else:
        # procesar en lotes de claves/valores
        positions = [pos for pos, _ in segments]
        texts_en = [src for _, src in segments]
        for chunk_start in range(0, len(texts_en), batch_size):
            chunk_positions = positions[chunk_start:chunk_start+batch_size]
            chunk_texts = texts_en[chunk_start:chunk_start+batch_size]
            chunk_translated = translate_segments_batch(client, model, chunk_texts)
            for pos, src, tgt in zip(chunk_positions, chunk_texts, chunk_translated):
                typ, obj = parsed[pos]
                assert typ == 'kv'
                parsed[pos] = ('kv', KVLine(key=obj.key, value=tgt, trailing_comment=obj.trailing_comment))
                mapping_out.append({'type': 'str', 'index_in_parsed': pos, 'source': src, 'translated': tgt})
                done += 1
                if show_progress:
                    print_progress(done, total)
                if show_preview:
                    v_escaped = tgt.replace('"', '\\"')
                    preview_line = f'{obj.key} = "{v_escaped}"'
                    if obj.trailing_comment:
                        preview_line += ' ' + obj.trailing_comment
                    print(preview_line, flush=True)

    if save_map:
        with open(save_map, 'w', encoding='utf-8') as f:
            json.dump(mapping_out, f, ensure_ascii=False, indent=2)

    return render_str(parsed, newline=newline), mapping_out

def process_srt_file(text: str, newline: str, client: OpenAI, model: str,
                     save_map: Optional[str], show_progress: bool,
                     show_preview: bool = False, batch_size: int = 1) -> Tuple[str, List[Dict]]:
    blocks = parse_srt(text)

    # primero contamos cuantos bloques requieren traducción para la barra
    translatable_block_idxs: List[int] = []
    for bi, b in enumerate(blocks):
        if any(line.strip() and should_translate(line) for line in b.text_lines):
            translatable_block_idxs.append(bi)

    total = len(translatable_block_idxs)
    mapping_out: List[Dict] = []
    done = 0

    if batch_size <= 1:
        # bloque a bloque como antes
        for bi in translatable_block_idxs:
            b = blocks[bi]
            try_block = translate_block_preserving_newlines(client, model, b.text_lines)
            if len(try_block) == len(b.text_lines):
                new_lines = try_block
            else:
                new_lines = []
                for src in b.text_lines:
                    if src.strip() and should_translate(src):
                        new_lines.append(translate_segment(client, model, src))
                    else:
                        new_lines.append(src)
            for li, (src, tgt) in enumerate(zip(b.text_lines, new_lines)):
                mapping_out.append({'type': 'srt', 'block': bi, 'line': li, 'source': src, 'translated': tgt})
            blocks[bi].text_lines = new_lines
            done += 1
            if show_progress:
                print_progress(done, total)
            if show_preview:
                preview_text = _render_srt_block(blocks[bi], newline)
                print(preview_text, end='', flush=True)
    else:
        # batch por bloques: cada item del batch es un bloque completo
        idx_chunks = _chunk_indices(translatable_block_idxs, batch_size)
        for chunk in idx_chunks:
            originals: List[str] = []
            for bi in chunk:
                b = blocks[bi]
                originals.append('\n'.join(b.text_lines))
            translated_blocks = translate_segments_batch(client, model, originals)
            # validar y, si falla la cantidad de líneas, fallback por bloque
            for bi, orig_joined, tr_joined in zip(chunk, originals, translated_blocks):
                orig_lines = orig_joined.split('\n')
                tr_lines = tr_joined.split('\n')
                if len(tr_lines) != len(orig_lines):
                    # fallback por bloque completo
                    b = blocks[bi]
                    try_block = translate_block_preserving_newlines(client, model, b.text_lines)
                    if len(try_block) == len(b.text_lines):
                        tr_lines = try_block
                    else:
                        # fallback final línea por línea
                        tr_lines = []
                        for src in b.text_lines:
                            if src.strip() and should_translate(src):
                                tr_lines.append(translate_segment(client, model, src))
                            else:
                                tr_lines.append(src)
                # mapping y reinyectar
                b = blocks[bi]
                for li, (src, tgt) in enumerate(zip(b.text_lines, tr_lines)):
                    mapping_out.append({'type': 'srt', 'block': bi, 'line': li, 'source': src, 'translated': tgt})
                blocks[bi].text_lines = tr_lines
                done += 1
                if show_progress:
                    print_progress(done, total)
                if show_preview:
                    preview_text = _render_srt_block(blocks[bi], newline)
                    print(preview_text, end='', flush=True)

    if save_map:
        with open(save_map, 'w', encoding='utf-8') as f:
            json.dump(mapping_out, f, ensure_ascii=False, indent=2)

    return render_srt(blocks, newline=newline), mapping_out

def process_file(in_path: str, out_suffix: str, dry_run: bool, model: str, api_key: Optional[str],
                 force_format: str, save_map: Optional[str], show_progress: bool,
                 show_preview: bool, batch_size: int) -> str:
    text, had_bom, newline = _read_text_keep_bom(in_path)
    fmt = detect_format(text) if force_format == 'auto' else force_format.lower()
    client = get_client(api_key)

    if fmt not in ('str', 'srt', 'auto'):
        raise ValueError("Formato inválido. Usá --format auto|str|srt")

    if fmt == 'srt' or (fmt == 'auto' and detect_format(text) == 'srt'):
        out_text, _ = process_srt_file(
            text, newline, client, model, save_map, show_progress, show_preview, batch_size
        )
        out_path = _derive_out_path(in_path, out_suffix, forced_ext='.srt')
    else:
        out_text, _ = process_str_file(
            text, newline, client, model, save_map, show_progress, show_preview, batch_size
        )
        out_path = _derive_out_path(in_path, out_suffix, forced_ext=None)

    if dry_run:
        print(out_text, end='')
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
    parser.add_argument('--live-preview', dest='live_preview', action='store_true',
                        help='Mostrar en vivo el resultado por segmento: en .srt imprime el bloque completo (index, tiempos y texto) y en .str imprime la linea key = "value".')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Cantidad de items por request. En SRT, cada item es 1 bloque (default: 1).')

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
            show_progress=args.progress,
            show_preview=args.live_preview,
            batch_size=max(1, args.batch_size)
        )
        if args.dry_run:
            print("\n--- fin (dry-run) ---")
        else:
            print(f"OK: guardado '{out_path}'")
            if args.map_json:
                print(f"OK: mapeo guardado en '{args.map_json}'")
        # Reporte de costos
        print("REPORTE:")
        # ahora total_prompt_tokens y total_completion_tokens ya existen y fueron actualizados
        if (total_prompt_tokens + total_completion_tokens) > 0:
            details = estimate_operation_cost(
                model=args.model,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cached_input_tokens=0,                    # si usaste cache input, ponelo acá
                preferred_section="Text tokens",          # o el que corresponda
                preferred_tier="Standard",                # Standard / Batch / Flex / Priority
                pricing_json_path="openai_pricing.json",  # usa JSON si lo tenés
                pricing_md_path="openai_pricing.md",      # fallback a MD si no hay JSON
            )
            print("\n=== Costo estimado ===")
            print(render_cost_summary(details))
        else:
            print("No se registraron tokens (0).")

        #estimate_operation_cost(args.input, out_path, args.map_json, args.model)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
