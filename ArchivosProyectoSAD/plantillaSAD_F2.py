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

        # Ahora, aplicar el TfidfVectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df[column])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Añadir el dataframe tf-idf a df y eliminar la columna de texto original
        df = pd.concat([df, tfidf_df], axis=1).drop(column, axis=1)

    # Guardar el vectorizador para usarlo más tarde
    with open('tfidf_vectorizer2.pkl', 'wb') as file:
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

def preproceso_por_chunks(input_file, config_file, output_file, chunksize=10000):
    config = load_config(config_file)
    processed_chunks = []

    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk = correct_column_types(chunk, config)
        chunk = impute_unique_values(chunk)
        chunk = impute_missing_values(chunk, config)
        chunk = handle_outliers(chunk, config)
        chunk = convert_categorical_to_numeric(chunk, config)
        chunk = process_text_data(chunk, config)
        #chunk = scale_data(chunk, config)
        #chunk = balance_data(chunk, config)

        processed_chunks.append(chunk)

    # Concatenar todos los chunks procesados
    df = pd.concat(processed_chunks, ignore_index=True)
    df.to_csv(output_file, index=False)
    print(f"Preprocesamiento por chunks completado. Archivo guardado en: {output_file}")
    
    return df


#División de datos en train, dev y test
def split_data(df, config):
    test_size = config.get("test_size", 0.2)
    dev_size = config.get("dev_size", 0.2)

    # Obtener la columna de target del JSON
    target_columns = config.get("preprocessing", {}).get("target", [])
    if len(target_columns) == 0:
        raise ValueError("El parámetro 'target' no está definido en el JSON.")
    
    # Si hay múltiples columnas de target, se toma la primera como y
    y = df[target_columns[0]]

    # Características X (el resto de las columnas excepto la de target)
    X = df.drop(columns=target_columns)

    # Separar los datos en conjunto de entrenamiento, validación y prueba con estratificación
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + dev_size), stratify=y, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (test_size + dev_size)), stratify=y_temp, random_state=42)
    
    #Estratificadas en base al target
    #X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + dev_size), stratify=y, random_state=42)
    #X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (test_size + dev_size)), stratify=y_temp, random_state=42)

    # Asegurarse de que y sea un vector unidimensional
    y_train = y_train.values.ravel()
    y_dev = y_dev.values.ravel()
    y_test = y_test.values.ravel()
    
    y_train = y_train.astype('int32')
    y_dev = y_dev.astype('int32')
    y_test = y_test.astype('int32')


    return X_train, X_dev, X_test, y_train, y_dev, y_test


def evaluate_model(model, X, y, dataset_name, params, algorithm, csv_filename, config):
    y_pred = model.predict(X)

    # 🔹 Obtener qué métricas se deben guardar desde config.json
    selected_metrics = config.get("metrics", ["accuracy", "precision", "recall", "fscore"])
    best_metric = config.get("best_metric", "fscore")  # 🔹 Métrica para elegir el mejor modelo
    average_type = config.get("average", "macro")  # 🔹 Tipo de promedio desde el JSON

    # 🔹 Calcular todas las métricas
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average=average_type, zero_division=0),
        "recall": recall_score(y, y_pred, average=average_type, zero_division=0),
        "fscore": f1_score(y, y_pred, average=average_type, zero_division=0),
    }

    # 🔹 Filtrar solo las métricas seleccionadas
    selected_values = [metrics[m] for m in selected_metrics if m in metrics]

    # 🔹 Escribir en CSV solo las métricas necesarias
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([algorithm, params, dataset_name] + selected_values)

    return metrics.get(best_metric, 0)  # 🔹 Devuelve la métrica especificada en config.json

def export_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Modelo exportado como {filename}")

def knn_pipeline(df, config):
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df, config)
    csv_filename = "knn_results.csv"

    selected_metrics = config.get("metrics", ["accuracy", "precision", "recall", "fscore"])
    best_metric = config.get("best_metric", "fscore")

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Hyperparameters", "Dataset"] + selected_metrics)

    k_min = config["knn"]["k_min"]
    k_max = config["knn"]["k_max"]
    p_values = config["knn"].get("p_values", [1, 2])
    weight_options = config["knn"].get("weights", ["uniform", "distance"])

    best_value = 0
    best_params = None
    best_model = None

    for k in range(k_min, k_max + 1, 2):
        for p in p_values:
            for weights in weight_options:
                model = KNeighborsClassifier(n_neighbors=k, p=p, weights=weights)
                model.fit(X_train, y_train)
                metric_value = evaluate_model(model, X_dev, y_dev, "dev", f"k={k}, p={p}, weights={weights}", "knn", csv_filename, config)

                if metric_value > best_value:
                    best_value = metric_value
                    best_params = f"k={k}, p={p}, weights={weights}"
                    best_model = model

    print(f"Mejor modelo KNN: {best_params} con {best_metric} {best_value}")
    evaluate_model(best_model, X_test, y_test, "test", best_params, "knn", csv_filename, config)
    export_model(best_model, "best_knn_model.pkl")


def decision_tree_pipeline(df, config):
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df, config)
    csv_filename = "decision_tree_results.csv"

    selected_metrics = config.get("metrics", ["accuracy", "precision", "recall", "fscore"])
    best_metric = config.get("best_metric", "fscore")

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Hyperparameters", "Dataset"] + selected_metrics)

    max_depth_values = config["decision_tree"]["max_depth"]
    min_samples_split_values = config["decision_tree"]["min_samples_split"]
    min_samples_leaf_values = config["decision_tree"]["min_samples_leaf"]

    best_value = 0
    best_params = None
    best_model = None

    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for min_samples_leaf in min_samples_leaf_values:
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                model.fit(X_train, y_train)
                metric_value = evaluate_model(model, X_dev, y_dev, "dev", f"max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}", "decision_tree", csv_filename, config)

                if metric_value > best_value:
                    best_value = metric_value
                    best_params = f"max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}"
                    best_model = model

    print(f"Mejor modelo Decision Tree: {best_params} con {best_metric} {best_value}")
    evaluate_model(best_model, X_test, y_test, "test", best_params, "decision_tree", csv_filename, config)
    export_model(best_model, "best_decision_tree_model.pkl")

def random_forest_pipeline(df, config):
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df, config)
    csv_filename = "random_forest_results.csv"

    selected_metrics = config.get("metrics", ["accuracy", "precision", "recall", "fscore"])
    best_metric = config.get("best_metric", "fscore")

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Hyperparameters", "Dataset"] + selected_metrics)

    n_estimators_values = config["random_forest"].get("n_estimators", [100])
    max_features_values = config["random_forest"].get("max_features", ["sqrt"])
    bootstrap_values = config["random_forest"].get("bootstrap", [True])
    max_depth_values = config["random_forest"].get("max_depth", [None])
    min_samples_split_values = config["random_forest"].get("min_samples_split", [2])
    min_samples_leaf_values = config["random_forest"].get("min_samples_leaf", [1])

    best_value = 0
    best_params = None
    best_model = None

    for n_estimators in n_estimators_values:
        for max_features in max_features_values:
            for bootstrap in bootstrap_values:
                for max_depth in max_depth_values:
                    for min_samples_split in min_samples_split_values:
                        for min_samples_leaf in min_samples_leaf_values:
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_features=max_features,
                                bootstrap=bootstrap,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf
                            )
                            model.fit(X_train, y_train)
                            metric_value = evaluate_model(
                                model, X_dev, y_dev, "dev", 
                                f"n_estimators={n_estimators}, max_features={max_features}, bootstrap={bootstrap}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}", 
                                "random_forest", csv_filename, config
                            )

                            if metric_value > best_value:
                                best_value = metric_value
                                best_params = f"n_estimators={n_estimators}, max_features={max_features}, bootstrap={bootstrap}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}"
                                best_model = model

    print(f"Mejor modelo Random Forest: {best_params} con {best_metric} {best_value}")
    evaluate_model(best_model, X_test, y_test, "test", best_params, "random_forest", csv_filename, config)
    export_model(best_model, "best_random_forest_model.pkl")
    
    
def naive_bayes_pipeline(df, config):
    from sklearn.preprocessing import KBinsDiscretizer
    from mixed_naive_bayes import MixedNB  # Asegúrate de instalarlo

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df, config)
    csv_filename = "naive_bayes_results.csv"

    selected_metrics = config.get("metrics", ["accuracy", "precision", "recall", "fscore"])
    best_metric = config.get("best_metric", "fscore")

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Hyperparameters", "Dataset"] + selected_metrics)

    best_value = 0
    best_params = None
    best_model = None
    best_discretizer = None

    nb_config = config.get("naive_bayes", {})
    model_type = nb_config.get("type", "gaussian")
    alpha_values = nb_config.get("alpha", [1.0])
    binarize_values = nb_config.get("binarize", [0.0])
    var_smoothing_values = nb_config.get("var_smoothing", [1e-9])

    use_discretization = nb_config.get("use_discretization", False)
    use_mixed_nb = nb_config.get("use_mixed_nb", False)

    # --------- DISCRETIZATION + CategoricalNB ---------
    if use_discretization:
        discretizer = KBinsDiscretizer(n_bins=nb_config.get("n_bins", 5), encode='ordinal', strategy='uniform')
        discretizer.fit(X_train)

        X_train_disc = discretizer.transform(X_train)
        X_dev_disc = discretizer.transform(X_dev)
        X_test_disc = discretizer.transform(X_test)

        for alpha in alpha_values:
            model = CategoricalNB(alpha=alpha)
            model.fit(X_train_disc, y_train)
            metric_value = evaluate_model(model, X_dev_disc, y_dev, "dev", f"CategoricalNB alpha={alpha}", "naive_bayes (discretized)", csv_filename, config)

            if metric_value > best_value:
                best_value = metric_value
                best_params = f"CategoricalNB alpha={alpha} (discretized)"
                best_model = model
                best_discretizer = discretizer  # Guardamos discretizador

    # --------- Mixed Naive Bayes ---------
    if use_mixed_nb:
        model = MixedNB()
        model.fit(X_train, y_train)
        metric_value = evaluate_model(model, X_dev, y_dev, "dev", "MixedNB default", "naive_bayes (mixed)", csv_filename, config)

        if metric_value > best_value:
            best_value = metric_value
            best_params = "MixedNB default"
            best_model = model
            best_discretizer = None

    # --------- Otros tipos de Naive Bayes clásicos ---------
    """
    for alpha in alpha_values:
        for binarize in binarize_values:
            for var_smoothing in var_smoothing_values:
                if model_type == "gaussian":
                    model = GaussianNB(var_smoothing=var_smoothing)
                    params = f"var_smoothing={var_smoothing}"
                elif model_type == "multinomial":
                    model = MultinomialNB(alpha=alpha)
                    params = f"alpha={alpha}"
                elif model_type == "bernoulli":
                    model = BernoulliNB(alpha=alpha, binarize=binarize)
                    params = f"alpha={alpha}, binarize={binarize}"
                else:
                    raise ValueError("Tipo de Naive Bayes no soportado")
                
                model.fit(X_train, y_train)
                metric_value = evaluate_model(model, X_dev, y_dev, "dev", params, "naive_bayes", csv_filename, config)

                if metric_value > best_value:
                    best_value = metric_value
                    best_params = params
                    best_model = model
                    best_discretizer = None
      """

    print(f"Mejor modelo Naive Bayes: {best_params} con {best_metric}: {best_value}")

    # --------- Evaluar sobre test ---------
    if best_discretizer is not None:
        evaluate_model(best_model, best_discretizer.transform(X_test), y_test, "test", best_params, "naive_bayes", csv_filename, config)
    else:
        evaluate_model(best_model, X_test, y_test, "test", best_params, "naive_bayes", csv_filename, config)

    # --------- Exportar modelo ---------
    # Cuando exportas:
    with open("best_naive_bayes_model2.pkl", 'wb') as file:
        pickle.dump({
            "model": best_model,
            "discretizer": best_discretizer
        }, file)


def classify_instances_from_csv():
    config = load_config('config.json')
    model_filename = config.get("model")
    csv_filename = "processed_data.csv"
    output_filename = "classified_instances.csv"
    
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    
    df = pd.read_csv(csv_filename)
    predictions = model.predict(df)
    
    df["Prediction"] = predictions
    df.to_csv(output_filename, index=False)
    print(f"Predicciones guardadas en {output_filename}")

#Punto de entrada principal
if __name__ == "__main__":
    config = load_config('config_A.json') #Cambiar JSON aquí (T --> TripAdvisor (1-5), A --> AirBnb (0,1))
    dataset_path = config.get("dataset")
    
    df = preproceso(dataset_path, 'config_A.json', 'processed_data.csv') #Cambiar JSON aquí también (T --> TripAdvisor (1-5), A --> AirBnb (0,1))
    mode = config.get("mode", "train")
    
    if mode == "train":
        algorithm = config.get("algorithm", "knn")
        if algorithm == "knn":
            knn_pipeline(df, config)
        elif algorithm == "decision_tree":
            decision_tree_pipeline(df, config)
        elif algorithm == "random_forest":
            random_forest_pipeline(df, config)
        elif algorithm == "naive_bayes":
            naive_bayes_pipeline(df, config)
        else:
            print("Algoritmo no reconocido en config.json")
    elif mode == "test":
        classify_instances_from_csv()
