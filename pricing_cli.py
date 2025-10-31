#!/usr/bin/env python3
# pricing_cli.py
from pathlib import Path
import argparse
import logging
import json

from openai_pricing.io import read_text_file, write_json_file
from openai_pricing.parser import parse_markdown_pricing
from openai_pricing.compute import find_price_entry, compute_cost_for_tokens

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('pricing_cli')

def main():
    p = argparse.ArgumentParser(description='Parsear archivo markdown de precios OpenAI y calcular costos.')
    p.add_argument('--input-md', type=str, default='openai_pricing.md', help='Archivo markdown con la tabla de precios')
    p.add_argument('--out-json', type=str, default='openai_pricing.json', help='Archivo JSON de salida (cache)')
    p.add_argument('--model', type=str, help='Modelo para calcular costo (ej: gpt-4o-mini)')
    p.add_argument('--section', type=str, help='Seccion preferida (ej: "Text tokens")')
    p.add_argument('--tier', type=str, help='Tier preferido (ej: "Standard")')
    p.add_argument('--input-tokens', type=int, default=0)
    p.add_argument('--output-tokens', type=int, default=0)
    p.add_argument('--cached-input-tokens', type=int, default=0)
    args = p.parse_args()

    md_path = Path(args.input_md)
    if not md_path.exists():
        logger.error('No existe %s', md_path)
        return

    md = read_text_file(md_path)
    catalog = parse_markdown_pricing(md)
    write_json_file(Path(args.out_json), {sec: {tier: {m: catalog[sec][tier][m].to_dict() for m in catalog[sec][tier]} for tier in catalog[sec]} for sec in catalog})

    logger.info('Parse completado y guardado en %s', args.out_json)

    if args.model:
        entry = find_price_entry(catalog, args.model, args.section, args.tier)
        if not entry:
            logger.error('Modelo %s no encontrado en el catalogo', args.model)
            available = []
            for sec in catalog.values():
                for tier in sec.values():
                    available.extend(list(tier.keys()))
            logger.info('Modelos disponibles (ejemplos): %s', ', '.join(sorted(set(available))[:40]))
            return
        cost = compute_cost_for_tokens(entry, args.input_tokens, args.output_tokens, args.cached_input_tokens)
        logger.info('Costo estimado (USD) para modelo %s: $%0.8f', args.model, cost)
        logger.info('Detalle precio (USD por 1M tokens): %s', entry.to_dict())

if __name__ == '__main__':
    main()
