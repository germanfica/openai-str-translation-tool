# openai-srt-translation-tool

## Prerequisites

- Python 3.10.0

## Install

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

##

```bash
# export OPENAI_API_KEY="sk-...."      # linux / mac
# o en Windows PowerShell:
# setx OPENAI_API_KEY "sk-...."
$env:OPENAI_API_KEY = "sk-...."

python main.py "Hola, dame un ejemplo de un endpoint REST en Nodejs"
```

```bash
python srt_translate.py ".\input.srt" --model "gpt-4o-mini" --progress
python srt_translate.py ".\input.srt" --model "gpt-4o-mini" --batch-size "10" --progress
python srt_translate.py ".\input.srt" --model "gpt-4o-mini" --live-preview
```
