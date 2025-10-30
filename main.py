# main.py
import os
import sys
from openai import OpenAI
from typing import Any

def main(prompt: str) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: setea la variable de entorno OPENAI_API_KEY antes de ejecutar.", file=sys.stderr)
        sys.exit(1)

    # Crear cliente (la libreria leerá api_key si no se lo pasamos explícitamente,
    # pero lo hacemos explícito aqui por claridad)
    client = OpenAI(api_key=api_key)

    try:
        # Ejemplo de completion estilo 'chat'
        resp: Any = client.chat.completions.create(
            model="gpt-3.5-turbo",  # cambialo por el modelo que prefieras / tengas acceso
            messages=[
                {"role": "system", "content": "Eres un asistente util y conciso."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )

        # En la mayoría de versiones la respuesta principal viene en:
        # resp.choices[0].message.content
        text = resp.choices[0].message.content
        print("Respuesta del modelo:\n")
        print(text)

    except Exception as e:
        print("Ocurrió un error al llamar a la API:", str(e), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py \"tu pregunta aqui\"")
        sys.exit(1)
    prompt = " ".join(sys.argv[1:])
    main(prompt)
