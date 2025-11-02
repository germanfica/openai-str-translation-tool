#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

# importamos SOLO la logica; la CLI queda igual
from translate_core import (
    process_file,
    total_prompt_tokens,
    total_completion_tokens,
)

# tu modulo de costos existente
from cost_report import estimate_operation_cost, render_cost_summary

def main():
    parser = argparse.ArgumentParser(
        description="Traductor de archivos .srt y .str EN->ES (latino) preservando estructura y placeholders."
    )
    parser.add_argument('input', help="Ruta al archivo .srt/.str origen en ingles.")
    parser.add_argument('--suffix', default='_es', help="Sufijo para el archivo de salida (default: _es).")
    parser.add_argument('--model', default='gpt-4o-mini',
                        help="Modelo de OpenAI a usar (default: gpt-4o-mini; usa el que tengas disponible).")
    parser.add_argument('--format', default='auto', choices=['auto', 'srt', 'str'],
                        help="Forzar formato de entrada (por defecto auto).")
    parser.add_argument('--api-key', default=None, help="Clave de API (si no, usa OPENAI_API_KEY).")
    parser.add_argument('--dry-run', action='store_true', help="Imprime el resultado en stdout sin escribir archivo.")
    parser.add_argument('--map-json', default=None, help="Guardar un JSON con el mapeo de traducciones.")
    parser.add_argument('--progress', action='store_true', help="Mostrar barra de progreso.")
    parser.add_argument('--live-preview', dest='live_preview', action='store_true',
                        help='Mostrar en vivo el resultado por segmento.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Cantidad de items por request. En SRT, cada item es 1 bloque (default: 1).')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: no se encontro el archivo: {args.input}", file=sys.stderr)
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
            batch_size=max(1, args.batch_size),
        )

        if args.dry_run:
            print("\n--- fin (dry-run) ---")
        else:
            print(f"OK: guardado '{out_path}'")
            if args.map_json:
                print(f"OK: mapeo guardado en '{args.map_json}'")

        # Reporte de costos (sin nuevos argumentos)
        print("REPORTE:")
        if (total_prompt_tokens + total_completion_tokens) > 0:
            details = estimate_operation_cost(
                model=args.model,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cached_input_tokens=0,
                preferred_section="Text tokens",
                preferred_tier="Standard",
                pricing_json_path="openai_pricing.json",
                pricing_md_path="openai_pricing.md",
            )
            print("\n=== Costo estimado ===")
            print(render_cost_summary(details))
        else:
            print("No se registraron tokens (0).")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
