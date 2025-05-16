import pandas as pd
import ast  # para convertir string a diccionario seguro

# Cargar el CSV
df = pd.read_csv("airbnb.csv")

# Convertir strings tipo dict a diccionarios reales (solo si están en string)
df["address"] = df["address"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Filtrar por países (Canadá y Australia)
df_filtrado = df[df["address"].apply(lambda x: x.get("country") in ["Canada"])]

# Guardar el resultado
df_filtrado.to_csv("airbnb_CA.csv", index=False)

print("¡CSV filtrado guardado como 'airbnb_CA.csv'!")

