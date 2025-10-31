#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cost_report.py
--------------
Cálculo y formateo de costo por operación usando tokens y tabla de precios local.

Requisitos:
- Tener en el proyecto los módulos del parser previos:
  openai_pricing/models.py, openai_pricing/io.py, openai_pricing/parser.py, openai_pricing/compute.py
- Un archivo de precios local:
  - Preferido: openai_pricing.json (generado por tu pricing_cli)
  - Alternativo: openai_pricing.md (el markdown que pegaste)

Uso programático:
    from cost_report import estimate_operation_cost, render_cost_summary
    details = estimate_operation_cost(
        model="gpt-4o-mini",
        prompt_tokens=150,
        completion_tokens=300,
        cached_input_tokens=0,
        preferred_section="Text tokens",
        preferred_tier="Standard",
        pricing_json_path="openai_pricing.json",
        pricing_md_path="openai_pricing.md",
    )
    print(render_cost_summary(details))

Uso CLI (opcional):
    python cost_report.py --model gpt-4o-mini --prompt 150 --completion 300
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import json
import logging
import datetime

# Reutilizamos tu paquete modular
from openai_pricing.models import PriceEntry
from openai_pricing.io import read_text_file
from openai_pricing.parser import parse_markdown_pricing
from openai_pricing.compute import find_price_entry, compute_cost_for_tokens

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class CostDetails:
    model: str
    section: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    cached_input_tokens: int
    price_input_per_1m: Optional[float]
    price_cached_input_per_1m: Optional[float]
    price_output_per_1m: Optional[float]
    cost_input: float
    cost_cached_input: float
    cost_output: float
    cost_total: float
    pricing_source: str            # json|md
    pricing_file: str
    timestamp_iso: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_catalog_from_json(path: Path) -> Optional[Dict[str, Dict[str, Dict[str, PriceEntry]]]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        catalog: Dict[str, Dict[str, Dict[str, PriceEntry]]] = {}
        for section, tiers in raw.items():
            catalog.setdefault(section, {})
            for tier, models in tiers.items():
                catalog[section].setdefault(tier, {})
                for model_name, entry in models.items():
                    catalog[section][tier][model_name] = PriceEntry(
                        input=entry.get("input"),
                        cached_input=entry.get("cached_input"),
                        output=entry.get("output"),
                    )
        return catalog
    except Exception as e:
        logger.warning("No se pudo leer JSON de precios (%s): %s", path, e)
        return None


def _load_catalog_from_md(path: Path) -> Optional[Dict[str, Dict[str, Dict[str, PriceEntry]]]]:
    if not path.exists():
        return None
    try:
        md = read_text_file(path)
        return parse_markdown_pricing(md)
    except Exception as e:
        logger.warning("No se pudo parsear markdown de precios (%s): %s", path, e)
        return None


def _ensure_catalog(pricing_json_path: str, pricing_md_path: str) -> tuple[Dict[str, Dict[str, Dict[str, PriceEntry]]], str, str]:
    """
    Devuelve (catalog, source_kind, source_file)
    - source_kind: 'json' o 'md'
    """
    json_path = Path(pricing_json_path)
    md_path = Path(pricing_md_path)

    # Preferimos JSON (estable y rápido)
    catalog = _load_catalog_from_json(json_path)
    if catalog:
        return catalog, "json", str(json_path)

    # Fallback: parsear markdown
    catalog = _load_catalog_from_md(md_path)
    if catalog:
        return catalog, "md", str(md_path)

    raise FileNotFoundError(
        f"No encuentro precios en '{json_path}' ni en '{md_path}'. Generá el JSON con tu pricing_cli o coloca el MD."
    )


def estimate_operation_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_input_tokens: int = 0,
    preferred_section: str = "Text tokens",
    preferred_tier: str = "Standard",
    pricing_json_path: str = "openai_pricing.json",
    pricing_md_path: str = "openai_pricing.md",
) -> CostDetails:
    """
    Calcula el costo de la operación a partir de tokens y la tabla de precios local.
    Retorna CostDetails con desglose y total.
    """
    if prompt_tokens < 0 or completion_tokens < 0 or cached_input_tokens < 0:
        raise ValueError("Los tokens no pueden ser negativos.")

    catalog, source_kind, source_file = _ensure_catalog(pricing_json_path, pricing_md_path)

    entry = find_price_entry(catalog, model_name=model, preferred_section=preferred_section, preferred_tier=preferred_tier)
    if not entry:
        # intento adicional: búsqueda relajada por minúsculas
        lowered = {m.lower(): (sec, tr, m) for sec, tiers in catalog.items() for tr, models in tiers.items() for m in models}
        key = model.lower()
        if key in lowered:
            sec, tr, orig_name = lowered[key]
            entry = catalog[sec][tr][orig_name]
            preferred_section, preferred_tier, model = sec, tr, orig_name
        else:
            # sugerencias
            candidates = sorted(list({m for sec in catalog.values() for tr in sec.values() for m in tr.keys()}))
            raise KeyError(f"Modelo '{model}' no encontrado en precios. Algunos disponibles: {', '.join(candidates[:20])} ...")

    # Calcular costo total con la función común
    total_cost = compute_cost_for_tokens(entry, input_tokens=prompt_tokens, output_tokens=completion_tokens, cached_input_tokens=cached_input_tokens)

    # Desglose por tipo
    pi = entry.input if entry.input is not None else entry.output
    po = entry.output if entry.output is not None else entry.input
    pc = entry.cached_input if entry.cached_input is not None else pi

    # Normalizar a precio por token
    pti = (pi or 0.0) / 1_000_000.0
    ptc = (pc or pti) / 1_000_000.0
    pto = (po or pti) / 1_000_000.0

    cached_applied = min(cached_input_tokens, prompt_tokens)
    non_cached = max(0, prompt_tokens - cached_applied)

    cost_input = non_cached * pti
    cost_cached = cached_applied * ptc
    cost_output = completion_tokens * pto

    return CostDetails(
        model=model,
        section=preferred_section,
        tier=preferred_tier,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_input_tokens=cached_input_tokens,
        price_input_per_1m=pi,
        price_cached_input_per_1m=pc,
        price_output_per_1m=po,
        cost_input=round(cost_input, 8),
        cost_cached_input=round(cost_cached, 8),
        cost_output=round(cost_output, 8),
        cost_total=round(total_cost, 8),
        pricing_source=source_kind,
        pricing_file=source_file,
        timestamp_iso=datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    )


def render_cost_summary(details: CostDetails) -> str:
    """
    Devuelve un string listo para imprimir en consola con el desglose.
    """
    parts = [
        f"Modelo: {details.model}  [{details.section} / {details.tier}]",
        f"Tokens -> input: {details.prompt_tokens} (cached: {details.cached_input_tokens}), output: {details.completion_tokens}",
        f"Precios (USD por 1M): input={details.price_input_per_1m}, cached_input={details.price_cached_input_per_1m}, output={details.price_output_per_1m}",
        f"Costo -> input=${details.cost_input:.8f}, cached=${details.cost_cached_input:.8f}, output=${details.cost_output:.8f}",
        f"TOTAL USD: ${details.cost_total:.8f}",
        f"Fuente precios: {details.pricing_source} ({details.pricing_file})",
        f"Fecha (UTC): {details.timestamp_iso}",
    ]
    return "\n".join(parts)


# CLI opcional para pruebas rápidas
def _main_cli():
    ap = argparse.ArgumentParser(description="Calcular costo de operación por tokens usando precios locales.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", type=int, default=0, help="prompt_tokens")
    ap.add_argument("--completion", type=int, default=0, help="completion_tokens")
    ap.add_argument("--cached", type=int, default=0, help="cached_input_tokens")
    ap.add_argument("--section", default="Text tokens")
    ap.add_argument("--tier", default="Standard")
    ap.add_argument("--pricing-json", default="openai_pricing.json")
    ap.add_argument("--pricing-md", default="openai_pricing.md")
    args = ap.parse_args()

    details = estimate_operation_cost(
        model=args.model,
        prompt_tokens=args.prompt,
        completion_tokens=args.completion,
        cached_input_tokens=args.cached,
        preferred_section=args.section,
        preferred_tier=args.tier,
        pricing_json_path=args.pricing_json,
        pricing_md_path=args.pricing_md,
    )
    print(render_cost_summary(details))


if __name__ == "__main__":
    _main_cli()
