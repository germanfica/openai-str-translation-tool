# openai_pricing/compute.py
from __future__ import annotations
from typing import Dict, Optional
from .models import PriceEntry
import math

def find_price_entry(catalog: Dict[str, Dict[str, Dict[str, PriceEntry]]],
                     model_name: str,
                     preferred_section: Optional[str] = None,
                     preferred_tier: Optional[str] = None) -> PriceEntry | None:
    """
    Busca una entrada PriceEntry por nombre de modelo.
    Si preferred_section/tier están dados, intenta usarlos primero.
    Luego hace búsqueda en todas las secciones/tiers y devuelve la primera coincidencia.
    """
    # try direct
    if preferred_section and preferred_tier:
        try:
            return catalog[preferred_section][preferred_tier][model_name]
        except KeyError:
            pass

    # try preferred_section across tiers
    if preferred_section:
        tiers = catalog.get(preferred_section, {})
        for tier_data in tiers.values():
            if model_name in tier_data:
                return tier_data[model_name]

    # global search
    for sec in catalog.values():
        for tier_data in sec.values():
            if model_name in tier_data:
                return tier_data[model_name]

    return None

def compute_cost_for_tokens(entry: PriceEntry,
                            input_tokens: int,
                            output_tokens: int,
                            cached_input_tokens: int = 0) -> float:
    """
    Calcula costo en USD usando los precios que están en entry (USD por 1M tokens).
    cached_input_tokens se aplica sobre input_tokens si entry.cached_input está presente,
    si no, se considera al mismo precio que input.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError('token counts must be non-negative')

    input_price_1m = entry.input
    output_price_1m = entry.output
    cached_price_1m = entry.cached_input

    # if both input and output are None, raise
    if input_price_1m is None and output_price_1m is None:
        raise ValueError('No hay precios disponibles en la entrada para calcular costo.')

    # fallback: if one missing, use the other if available
    if input_price_1m is None:
        input_price_1m = output_price_1m
    if output_price_1m is None:
        output_price_1m = input_price_1m
    if cached_price_1m is None:
        cached_price_1m = input_price_1m

    input_price_per_token = input_price_1m / 1_000_000.0
    output_price_per_token = output_price_1m / 1_000_000.0
    cached_price_per_token = cached_price_1m / 1_000_000.0

    cached = min(cached_input_tokens, input_tokens)
    non_cached = max(0, input_tokens - cached)

    cost = non_cached * input_price_per_token + cached * cached_price_per_token + output_tokens * output_price_per_token
    # redondear a 10^-8 para consistencia
    return float(round(cost, 8))
