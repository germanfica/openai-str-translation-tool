#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

from translate_core import run_translation
from precheck_core import run_cost_precheck_for_core

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
                        help='Mostrar en vivo el resultado por segmento: en .srt imprime el bloque y en .str la linea key = "value".')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Cantidad de items por request. En SRT, cada item es 1 bloque (default: 1).')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: no se encontro el archivo: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        # ===================== PRECHECK (inicio) =====================
        # Muestra WARNING con estimación de costo basada SOLO en lo que se enviará al LLM
        # y pide confirmación y/N. Si el usuario cancela, run_cost_precheck_for_core debe hacer sys.exit(130).
        precheck_summary = run_cost_precheck_for_core(
            in_path=args.input,
            model=args.model,
            fmt_forced=args.format,               # 'auto'|'str'|'srt'
            batch_size=max(1, args.batch_size),
            completion_multiplier=1.0,            # asume salida ≈ entrada (ajustar si corresponde)
            section='Text tokens',
            tier='Standard',
            pricing_json_path='openai_pricing.json',
            pricing_md_path='openai_pricing.md',
            assume_yes=False                      # siempre preguntar en este script
        )
        # opcional: podés loguear precheck_summary si necesitas

        # ===================== TRADUCCIÓN =====================
        out_path = run_translation(
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
            report=True,  # el reporte sale desde translate_core
        )

        if args.dry_run:
            print("\n--- fin (dry-run) ---")
        else:
            print(f"OK: guardado '{out_path}'")
            if args.map_json:
                print(f"OK: mapeo guardado en '{args.map_json}'")

    except SystemExit:
        # Si el precheck llamó a sys.exit(130) o similar, respetamos ese comportamiento
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
