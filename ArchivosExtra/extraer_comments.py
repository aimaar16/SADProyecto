import pandas as pd
import ast
import csv  # Importación añadida para controlar el quoting al guardar el CSV

# Cargar el CSV
df = pd.read_csv("airbnb_CA.csv")

# Crear una lista para almacenar los comentarios y puntuaciones
comentarios_y_scores_totales = []

# Iterar sobre todas las filas del DataFrame
for i in range(len(df)):
    try:
        # Obtener la cadena de la fila en la columna "reviews"
        texto_reviews = df.loc[i, "reviews"]
        
        # Obtener el campo review_scores_value de la columna review_scores
        review_scores_str = df.loc[i, "review_scores"]
        
        # Manejo de errores al convertir la cadena review_scores a un diccionario
        try:
            review_scores_dict = ast.literal_eval(review_scores_str)
            review_value = review_scores_dict.get("review_scores_value", None)
            if review_value is not None:
                # Convertir a string y reemplazar punto por coma
                review_value = str(review_value).replace('.', ',')
        except (ValueError, SyntaxError):
            review_value = None
        
        # Convertir la cadena de "reviews" a una lista de diccionarios
        try:
            lista_reviews = ast.literal_eval(texto_reviews)
        except (ValueError, SyntaxError):
            lista_reviews = []
        
        # Extraer los comentarios y puntuaciones de manera segura
        for r in lista_reviews:
            if "comments" in r:
                comentario = r["comments"]
                comentarios_y_scores_totales.append({
                    "Review": comentario,
                    "Review_Score": review_value
                })
    
    except Exception as e:
        print(f"Error en la fila {i}: {e}")

# Crear un nuevo DataFrame y guardarlo como CSV con los campos entrecomillados
df_comentarios = pd.DataFrame(comentarios_y_scores_totales)
df_comentarios.to_csv("reviews_extraidas.csv", index=False, quoting=csv.QUOTE_ALL)

# Imprimir el número total de reviews
print(f"Se han escrito {len(comentarios_y_scores_totales)} reviews.")

