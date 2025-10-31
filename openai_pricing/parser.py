# openai_pricing/parser.py
from __future__ import annotations
import re
from typing import Dict, Tuple, List
from .models import PriceEntry
import logging

logger = logging.getLogger(__name__)

_RE_TABLE_ROW = re.compile(r'^\s*\|(.+)\|\s*$')
_RE_SEPARATOR = re.compile(r'^\s*\|\s*-{1,}\s*(\|\s*-{1,}\s*)+\|\s*$')
_RE_PRICE = re.compile(r'\$?\s*([0-9,]*\.?[0-9]+|-)\s*')  # captures number or '-'

def _parse_price_cell(cell: str) -> float | None:
    m = _RE_PRICE.search(cell.strip())
    if not m:
        return None
    raw = m.group(1)
    if raw == '-':
        return None
    # remove commas
    raw = raw.replace(',', '')
    try:
        return float(raw)
    except ValueError:
        return None

def parse_markdown_pricing(md_text: str) -> Dict[str, Dict[str, Dict[str, PriceEntry]]]:
    """
    Devuelve estructura:
      { section_name: { tier_name: { model_name: PriceEntry(...) } } }
    Ejemplo: data['Text tokens']['Batch']['gpt-5'].input -> float | None
    """
    lines = md_text.splitlines()
    section = None
    tier = None
    i = 0
    data: Dict[str, Dict[str, Dict[str, PriceEntry]]] = {}

    # Helper to ensure nested dicts
    def ensure(section_name: str, tier_name: str):
        if section_name not in data:
            data[section_name] = {}
        if tier_name not in data[section_name]:
            data[section_name][tier_name] = {}

    while i < len(lines):
        line = lines[i].strip()
        # detect section headings: example "Text tokens" or "Image tokens" etc.
        if line and not line.startswith('|') and not line.startswith('###') and not line.endswith(':'):
            # Avoid accidental capture of paragraph lines; heurística:
            # if next non-empty line contains 'Prices' or 'Prices per' or next line is empty and following is a small title (Batch/Flex)
            next_non_empty = None
            j = i + 1
            while j < len(lines) and not next_non_empty:
                if lines[j].strip():
                    next_non_empty = lines[j].strip()
                j += 1
            if next_non_empty and ('price' in next_non_empty.lower() or next_non_empty.lower().startswith('prices') or next_non_empty.lower().startswith('|model|')):
                section = line
                tier = None
                logger.debug('Seccion detectada: %s (line %d)', section, i+1)
                i += 1
                continue

        # detect tier headings like 'Batch', 'Flex', 'Standard', 'Priority'
        if line and not line.startswith('|') and line.lower() in {'batch', 'flex', 'standard', 'priority', 'text tokens', 'image tokens', 'audio tokens', 'video', 'fine-tuning', 'fine-tuning', 'fine-tuning', 'built-in tools', 'embeddings', 'legacy models'}:
            # common tiers are short words, we treat them as tier if previous section exists
            if section:
                tier = line
                ensure(section, tier)
                logger.debug('Tier detectado: %s en sección %s', tier, section)
                i += 1
                continue

        # detect table header line
        if _RE_TABLE_ROW.match(lines[i]):
            # peek next line for separator
            if i + 1 < len(lines) and _RE_SEPARATOR.match(lines[i + 1]):
                header_cells = [c.strip().lower() for c in lines[i].strip().strip('|').split('|')]
                # normalize header names to expected: model,input,cached input,output or price per second etc.
                headers = [h.replace(' ', '_') for h in header_cells]
                # read rows
                i += 2
                rows: List[str] = []
                while i < len(lines) and _RE_TABLE_ROW.match(lines[i]):
                    rows.append(lines[i])
                    i += 1

                # parse rows
                # If tier is None, try to infer from last non-empty non-table short line above header
                if not tier:
                    # look backward for a short line (<= 50 chars) that is not a section
                    k = i - len(rows) - 3
                    found = None
                    while k >= 0:
                        candidate = lines[k].strip()
                        if candidate and not candidate.startswith('|') and len(candidate) < 60:
                            found = candidate
                            break
                        k -= 1
                    tier = found or 'default'
                    ensure(section or 'default', tier)
                    logger.debug('Inferido tier: %s', tier)

                if section is None:
                    section = 'default'
                    ensure(section, tier)

                for r in rows:
                    # cells
                    raw_cells = [c.strip() for c in r.strip().strip('|').split('|')]
                    if not raw_cells:
                        continue
                    model_name = raw_cells[0]
                    # map other columns by header
                    mapping = {}
                    for idx in range(1, min(len(raw_cells), len(headers))):
                        hdr = headers[idx]
                        mapping[hdr] = raw_cells[idx]
                    # common header keys: input, cached_input, output, price_per_second
                    input_v = None
                    cached_v = None
                    output_v = None
                    if 'input' in mapping:
                        input_v = _parse_price_cell(mapping['input'])
                    elif 'price' in mapping and 'per_second' not in headers:
                        input_v = _parse_price_cell(mapping['price'])
                    if 'cached_input' in mapping:
                        cached_v = _parse_price_cell(mapping['cached_input'])
                    elif 'cached' in mapping:
                        cached_v = _parse_price_cell(mapping['cached'])
                    if 'output' in mapping:
                        output_v = _parse_price_cell(mapping['output'])
                    # fallback: if a table has 4 columns and headers unknown, assume columns [Model, Input, Cached input, Output]
                    if input_v is None and cached_v is None and output_v is None and len(raw_cells) >= 4:
                        input_v = _parse_price_cell(raw_cells[1])
                        cached_v = _parse_price_cell(raw_cells[2])
                        output_v = _parse_price_cell(raw_cells[3])
                    # store
                    entry = PriceEntry(input=input_v, cached_input=cached_v, output=output_v)
                    data.setdefault(section, {}).setdefault(tier, {})[model_name.strip()] = entry
                # after table, continue outer loop
                continue

        i += 1

    logger.debug('Parse completo. Secciones: %s', list(data.keys()))
    return data
