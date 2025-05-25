import pandas as pd
import numpy as np
import json
import re
import string
import csv
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product



def load_data(file_path):
    return pd.read_csv(file_path)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def impute_missing_values(df, config):
    if config['preprocessing']['missing_values']['impute']:
        impute_method_numeric = config['preprocessing']['missing_values']['impute_method_numeric']
        impute_method_categorical = config['preprocessing']['missing_values']['impute_method_categorical']
        categorical_columns = config['preprocessing'].get('categorical_columns', [])

        # Primero eliminamos nulos si así lo indican los métodos de imputación
        if impute_method_categorical == "remove":
            for column in categorical_columns:
                df = df[df[column].notna()]

        if impute_method_numeric == "remove":
            for column in df.columns:
                if column not in categorical_columns and pd.api.types.is_numeric_dtype(df[column]):
                    df = df[df[column].notna()]

        # Después, imputamos los valores si no se han eliminado
        for column in df.columns:
            if column in categorical_columns:
                df[column] = df[column].astype('category')
                if impute_method_categorical == "mode":
                    df[column] = df[column].fillna(df[column].mode()[0])
            elif pd.api.types.is_numeric_dtype(df[column]):
                if impute_method_numeric == "mean":
                    df[column] = df[column].fillna(df[column].mean())
                elif impute_method_numeric == "median":
                    df[column] = df[column].fillna(df[column].median())
                elif impute_method_numeric == "mode":
                    df[column] = df[column].fillna(df[column].mode()[0])

    return df




def correct_column_types(df, config):
    if config['preprocessing'].get('correct_column_types', {}).get('enabled', False):
        for column in df.columns:
            num_valid_values = df[column].apply(lambda x: str(x).replace('.', '', 1).isdigit()).sum()
            total_values = df[column].shape[0]

            if total_values > 0 and (num_valid_values / total_values) > 0.8:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif column in config['preprocessing'].get('categorical_columns', []):
                if df[column].nunique() == 1:
                    df[column] = np.nan
                else:
                    mode = df[column].mode()[0]
                    df[column] = df[column].apply(lambda x: x if isinstance(x, str) and not x.isdigit() else mode)
            elif column in config['preprocessing'].get('target', []):
                if df[column].nunique() == 1:
                    df[column] = np.nan
                else:
                    mode = df[column].mode()[0]
                    df[column] = df[column].apply(lambda x: x if isinstance(x, str) and not x.isdigit() else mode)
                    
    return df


def convert_categorical_to_numeric(df, config):
    preprocessing = config.get('preprocessing', {})
    categorical_columns = preprocessing.get('categorical_columns', [])

    categorical_columns = [col for col in categorical_columns if col in df.columns and col!="Academic Pressure"] #and col != "nombre_x"

    for column in categorical_columns:
        df[column] = df[column].astype(str).fillna("Missing")
        df[column] = df[column].astype('category').cat.codes

    return df


def simplify_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS and len(word) > 1])
    else:
        text = ''
    return text


def process_text_data(df, config):
    categorical_columns = config['preprocessing'].get('categorical_columns', [])
    target = config['preprocessing'].get('target', [])
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    text_columns = [col for col in df.columns if col not in categorical_columns and col not in numeric_columns and col not in target]
    
    min_freq = config['preprocessing']['text_processing'].get('tfidf_min_frequency', 1)  # Umbral de frecuencia mínima

    # Intentamos cargar el vectorizador desde un archivo si existe
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        print("TfidfVectorizer cargado exitosamente desde archivo.")
    except FileNotFoundError:
        vectorizer = TfidfVectorizer()
        print("TfidfVectorizer no encontrado, se crea uno nuevo.")

    for column in text_columns:
        df[column] = df[column].fillna('')
        df[column] = df[column].apply(simplify_text)

        # Contar la frecuencia de las palabras en todo el dataset
        all_text = ' '.join(df[column])
        word_counts = {}
        for word in all_text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

        # Filtrar palabras que no alcanzan el umbral
        words_above_threshold = {word for word, count in word_counts.items() if count >= min_freq}

        # Filtrar el texto en cada fila, manteniendo solo las palabras por encima del umbral
        df[column] = df[column].apply(lambda text: ' '.join([word for word in text.split() if word in words_above_threshold]))

        # Aplicar el vectorizador ya cargado o creado
        tfidf_matrix = vectorizer.transform(df[column])  # Usamos transform en vez de fit_transform
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Añadir el dataframe tf-idf a df y eliminar la columna de texto original
        df = pd.concat([df, tfidf_df], axis=1).drop(column, axis=1)

    # Si el vectorizador es nuevo, lo guardamos para futuras ejecuciones
    if not hasattr(vectorizer, 'vocabulary_'):
        with open('tfidf_vectorizer.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)
        print("TfidfVectorizer guardado para futuras ejecuciones.")

    return df


def process_text_data2(df, config):
    categorical_columns = config['preprocessing'].get('categorical_columns', [])
    target = config['preprocessing'].get('target', [])
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    text_columns = [col for col in df.columns if col not in categorical_columns and col not in numeric_columns and col not in target]
    
    min_freq = config['preprocessing']['text_processing'].get('tfidf_min_frequency', 1)  # Umbral de frecuencia mínima

    # Intentamos cargar el vectorizador desde un archivo si existe
    try:
        with open('tfidf_vectorizer2.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        print("TfidfVectorizer cargado exitosamente desde archivo.")
    except FileNotFoundError:
        vectorizer = TfidfVectorizer()
        print("TfidfVectorizer no encontrado, se crea uno nuevo.")

    for column in text_columns:
        df[column] = df[column].fillna('')
        df[column] = df[column].apply(simplify_text)

        # Contar la frecuencia de las palabras en todo el dataset
        all_text = ' '.join(df[column])
        word_counts = {}
        for word in all_text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

        # Filtrar palabras que no alcanzan el umbral
        words_above_threshold = {word for word, count in word_counts.items() if count >= min_freq}

        # Filtrar el texto en cada fila, manteniendo solo las palabras por encima del umbral
        df[column] = df[column].apply(lambda text: ' '.join([word for word in text.split() if word in words_above_threshold]))

        # Aplicar el vectorizador ya cargado o creado
        tfidf_matrix = vectorizer.transform(df[column])  # Usamos transform en vez de fit_transform
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Añadir el dataframe tf-idf a df y eliminar la columna de texto original
        df = pd.concat([df, tfidf_df], axis=1).drop(column, axis=1)

    # Si el vectorizador es nuevo, lo guardamos para futuras ejecuciones
    if not hasattr(vectorizer, 'vocabulary_'):
        with open('tfidf_vectorizer.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)
        print("TfidfVectorizer guardado para futuras ejecuciones.")

    return df

def scale_data(df, config):
    if config.get('preprocessing', {}).get('scaling', {}).get('enabled', False):
        # Obtener las columnas a escalar desde el JSON
        scaling_columns = config.get('preprocessing', {}).get('scaling', {}).get('columns', [])
        
        # Filtrar solo las columnas numéricas indicadas en el JSON
        numeric_columns = [col for col in scaling_columns if col in df.select_dtypes(include=[np.number]).columns]

        if numeric_columns:
            scaling_method = config['preprocessing']['scaling'].get('method', 'minmax').lower()

            if scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif scaling_method == "zscore":
                scaler = StandardScaler()
            elif scaling_method == "maxscale":
                scaler = MaxAbsScaler()
            else:
                raise ValueError(f"Método de escalado desconocido: {scaling_method}")

            # Aplicar el escalado solo a las columnas especificadas en el JSON
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df


def balance_data(df, config):
    sampling_config = config.get('preprocessing', {}).get('sampling', {})
    method = sampling_config.get('method', None)
    target_column = sampling_config.get('target_column', None)

    if method and target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if method == "oversample":
            sampler = RandomOverSampler(random_state=42)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=42)
        else:
            raise ValueError(f"Método de muestreo desconocido: {method}")

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        df = pd.concat([X_resampled, y_resampled], axis=1)

    return df

def impute_unique_values(df):
    for column in df.columns:
        if df[column].nunique() == 1:
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df
    
def handle_outliers(df, config):
    outlier_action = config['preprocessing']['outliers']['handle']
    columns_to_check = config['preprocessing']['outliers'].get('columns', [])
    
    # Filtrar solo las columnas numéricas dentro de las especificadas en el JSON
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    columns_to_process = [col for col in columns_to_check if col in numeric_columns]
    
    for column in columns_to_process:
        # Calcular los cuartiles y el rango intercuartílico (IQR)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir los límites superior e inferior
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        # Acciones basadas en la configuración del JSON
        if outlier_action == "remove":
            df = df[(df[column] >= lower_limit) & (df[column] <= upper_limit) | df[column].isna()]
        elif outlier_action == "round":
            df[column] = df[column].apply(
                lambda x: lower_limit if x < lower_limit and not pd.isna(x) else 
                          (upper_limit if x > upper_limit and not pd.isna(x) else x)
            )
    
    return df

#Función principal que llama a todas las etapas de preprocesamiento
def preproceso(input_file, config_file, output_file):
    df = load_data(input_file)
    df.drop(df.columns[1], axis=1, inplace=True)
    config = load_config(config_file)

    #Preprocesos
    df = correct_column_types(df, config)
    df = impute_unique_values(df)
    df = impute_missing_values(df, config)
    df = handle_outliers(df, config)
    df = convert_categorical_to_numeric(df, config)
    df = process_text_data(df, config)
    #df = scale_data(df, config)
    df = balance_data(df, config)

    df.to_csv(output_file, index=False)
    print(f"Preprocesamiento completado. Archivo guardado en: {output_file}")
    
    return df
    
def preproceso2(input_file, config_file, output_file):
    df = load_data(input_file)
    df.drop(df.columns[1], axis=1, inplace=True)
    config = load_config(config_file)

    #Preprocesos
    df = correct_column_types(df, config)
    df = impute_unique_values(df)
    df = impute_missing_values(df, config)
    df = handle_outliers(df, config)
    df = convert_categorical_to_numeric(df, config)
    df = process_text_data2(df, config)
    #df = scale_data(df, config)
    df = balance_data(df, config)

    df.to_csv(output_file, index=False)
    print(f"Preprocesamiento completado. Archivo guardado en: {output_file}")
    
    return df
    
def classify_instances_from_csv():
    config = load_config('config_A.json')

    model_filenames = ['best_naive_bayes_model.pkl', 'best_naive_bayes_model2.pkl']
    processed_filenames = ['processed_data.csv', 'processed_data2.csv']
    
    original_csv_filename = "reviews_extraidas.csv"
    output_filename = "classified_instances.csv"

    original_df = pd.read_csv(original_csv_filename)

    prediction_columns = []

    for model_filename, processed_filename in zip(model_filenames, processed_filenames):
        with open(model_filename, 'rb') as file:
            model_data = pickle.load(file)

        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
            discretizer = model_data.get("discretizer", None)
        else:
            model = model_data
            discretizer = None

        processed_df = pd.read_csv(processed_filename)

        if discretizer is not None:
            processed_df = discretizer.transform(processed_df)

        predictions = model.predict(processed_df)

        prediction_column_name = f"Prediction_{model_filename.split('.')[0]}"
        original_df[prediction_column_name] = predictions
        prediction_columns.append(prediction_column_name)

    original_df = calculate_final_prediction(original_df, *prediction_columns)

    # Reemplazar puntos por comas en la columna Review_Score
    if 'Review_Score' in original_df.columns:
        original_df['Review_Score'] = original_df['Review_Score'].astype(str).str.replace('.', ',', regex=False)

    original_df.to_csv(output_filename, index=False)
    print(f"Predicciones guardadas en {output_filename}")




def calculate_final_prediction(df, prediction_col_1, prediction_col_2):
    """
    Barrido de hiperparámetros para combinación lineal + clipping.
    Se prueba en dev, se evalúa en test con la mejor combinación.
    """

    # Dividir en dev y test (80%-20%)
    df_dev, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Rango de hiperparámetros
    multipliers = [round(x, 1) for x in np.arange(1.0, 5, 0.2)]  # De 1.0 a 3.0
    biases = [round(x, 1) for x in np.arange(0.0, 5, 0.2)]       # De 0.0 a 3.0
    clip_mins = list(range(1, 8))                                  # De 1 a 7

    
    best_params = None
    best_score = float('inf')  # menor diferencia absoluta media
    
    print("Evaluando combinaciones en dev set:")
    for m, b, cmin in product(multipliers, biases, clip_mins):
        temp_pred = (m * df_dev[prediction_col_1] + df_dev[prediction_col_2] + b).clip(cmin, 10)
        diff = (df_dev['Review_Score'] - temp_pred).abs().mean()
        
        print(f"multiplicador={m}, bias={b}, clip_min={cmin} --> diff_abs_mean={diff:.4f}")
        
        if diff < best_score:
            best_score = diff
            best_params = (m, b, cmin)

    # Aplicar mejor combinación al test
    best_m, best_b, best_clip_min = best_params
    df_test['Final_Prediction'] = (best_m * df_test[prediction_col_1] + df_test[prediction_col_2] + best_b).clip(best_clip_min, 10)
    df_test['Difference'] = df_test['Review_Score'] - df_test['Final_Prediction']

    review_score_mean = df_test['Review_Score'].mean()
    final_prediction_mean = df_test['Final_Prediction'].mean()

    print("\n--- Resultados finales en test set ---")
    print(f"Mejores hiperparámetros encontrados: multiplicador={best_m}, bias={best_b}, clip_min={best_clip_min}")
    print(f"Media de Review_Score (test): {review_score_mean:.2f}")
    print(f"Media de Final_Prediction (test): {final_prediction_mean:.2f}")


    return df_test





    
if __name__ == "__main__":
    df = preproceso('reviews_traducidas.csv', 'config_A.json', 'processed_data.csv')
    df = preproceso2('reviews_traducidas.csv', 'config_A.json', 'processed_data2.csv')
    classify_instances_from_csv()
