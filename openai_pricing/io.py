# openai_pricing/io.py
from pathlib import Path
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

def read_text_file(path: Path) -> str:
    txt = path.read_text(encoding='utf-8')
    logger.debug('Leido archivo %s (%d bytes)', path, len(txt))
    return txt

def write_json_file(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.debug('Escrito JSON a %s', path)
