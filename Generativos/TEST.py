from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

# Ejecuta: ollama pull gemma2:2b antes de correr el script

parser = argparse.ArgumentParser(description='Valorar reviews con Ollama LLM')
parser.add_argument('--model', type=str, default='gemma2:2b', help='Nombre del modelo en Ollama')
parser.add_argument('--input', type=str, default='reviews_traducidas.csv', help='Archivo CSV de entrada')
parser.add_argument('--limit', type=int, default=250, help='Número de reviews a procesar (usar -1 para todas)')
args = parser.parse_args()

# Lista de prompts a evaluar
prompt_templates = [
    "Eres un experto valorador de reseñas, la vida de mucha gente depende de tu respuesta. Valora esta reseña del 1 al 10 devolviendo solo el número, porfavor. Reseña: {review}",
    "Eres un crítico experto. Da una puntuación del 1 al 10 a esta opinión de usuario. Responde únicamente con el número: {review}",
    "You are an expert critic. Give this user review a score from 1 to 10, responding only with the number: {review}"
]

# Cargar datos
df = pd.read_csv(args.input)

# Filtrar por limit
if args.limit != -1:
    df = df.head(args.limit)

# Dividir en dev y test
dev_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Cargar modelo
model = OllamaLLM(model=args.model, temperature=0)

def evaluar_prompt(prompt_text, df_subset):
    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | model
    predicciones = []
    for i, row in df_subset.iterrows():
        review = row['Review']
        respuesta = chain.invoke({'review': review}).strip()
        try:
            valor = float(respuesta)
        except ValueError:
            print(f"Valor no numérico recibido: '{respuesta}'")
            valor = np.nan
        predicciones.append(valor)
    df_subset = df_subset.copy()
    df_subset['predicho'] = predicciones
    reales = df_subset['Review_Score'].astype(float)
    predichos = df_subset['predicho']
    desviacion = np.abs(reales - predichos).mean()
    return desviacion, df_subset

# Probar cada prompt y guardar desviaciones
resultados = []
for prompt_text in prompt_templates:
    print(f"Evaluando prompt:\n{prompt_text}\n")
    desviacion, df_con_preds = evaluar_prompt(prompt_text, dev_df)
    resultados.append((desviacion, prompt_text, df_con_preds))
    print(f"Desviación media: {desviacion:.2f}\n")

# Seleccionar el mejor prompt
mejor_prompt = min(resultados, key=lambda x: x[0])
mejor_texto_prompt = mejor_prompt[1]
print(f"Mejor prompt seleccionado:\n{mejor_texto_prompt}\n")

# Aplicar el mejor prompt al conjunto de test
print("Evaluando en el conjunto de test...")
_, test_df_con_preds = evaluar_prompt(mejor_texto_prompt, test_df)

# Guardar resultados
dev_df_resultado = mejor_prompt[2]
dev_df_resultado.to_csv("dev_con_valoraciones.csv", index=False)
test_df_con_preds.to_csv("test_con_valoraciones.csv", index=False)

# Mostrar resumen
media_real = test_df_con_preds['Review_Score'].mean()
media_pred = test_df_con_preds['predicho'].mean()
desviacion_final = np.abs(test_df_con_preds['Review_Score'] - test_df_con_preds['predicho']).mean()
print(f"\nResultado final en test:")
print(f"Media real: {media_real:.2f}")
print(f"Media predicha: {media_pred:.2f}")
print(f"Desviación media: {desviacion_final:.2f}")
