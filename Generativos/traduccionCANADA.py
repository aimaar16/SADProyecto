from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import pandas as pd
import argparse

# Ejecuta: ollama pull gemma2:2b  antes de correr el script

parser = argparse.ArgumentParser(description='Traducir reviews con Ollama LLM')
parser.add_argument('--model', type=str, default='gemma2:2b', help='Nombre del modelo en Ollama')
parser.add_argument('--input', type=str, default='reviews_extraidas.csv', help='Archivo CSV de entrada')
parser.add_argument('--limit', type=int, default=-1, help='Número de reviews a traducir')
args = parser.parse_args()

# Crear prompt para traducción
template = "Eres un experto en traducción, la vida de muchas personas depende de tu trabajo.Traduce al inglés el siguiente texto manteniendo el significado original, solamente devolviendome el texto traducido:\nTexto: {review}\nTraducción:"
prompt = PromptTemplate.from_template(template)

# Cargar modelo
model = OllamaLLM(model=args.model, temperature=0)
chain = prompt | model

# Leer CSV con pandas
df = pd.read_csv(args.input)

# Procesar las primeras `limit` filas
for i in range(len(df) if args.limit == -1 else min(args.limit, len(df))):
    original = df.loc[i, 'Review']
    traduccion = chain.invoke({'review': original}).strip()
    df.loc[i, 'Review'] = traduccion
    print(f"[{i+1}] Traducido:\n{traduccion}\n")

# Guardar el resultado (opcional)
df.to_csv('reviews_traducidas.csv', index=False)
