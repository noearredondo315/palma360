# -*- coding: utf-8 -*-
"""
Predictor para el clasificador con validación cruzada de 5 folds.

Este script realiza predicciones utilizando el modelo entrenado por V2_classifier_5fold.py.
"""

# Bibliotecas estándar
import os
import time
from typing import Dict, List, Tuple, Union, Optional, Any

# Bibliotecas de terceros
import numpy as np
import pandas as pd
import joblib
import torch
from sentence_transformers import SentenceTransformer


# --- Configuración para Predicción Batch ---
MAP_INPUT_PATH = '/content/drive/MyDrive/Colab_PT/v5/kiosko_desglosado_23042025.xlsx'  # Ruta al archivo de entrada en Google Drive
MAP_OUTPUT_PATH = '/content/drive/MyDrive/Colab_PT/v5/test2_kiosko_desglosado_23042025_predicted.xlsx'  # Ruta de salida en Google Drive
MODEL_DIR = '/content/drive/MyDrive/Colab_PT/v5/modelo_v5'  # Directorio en Google Drive para guardar artefactos
EMBEDDING_MODEL_NAME = 'sentence-transformers/stsb-xlm-r-multilingual'  # Modelo más robusto para españolCONFIDENCE_THRESHOLD_PREDICT = 0.6  # Umbral de confianza


# Bibliotecas estándar
import os
import re
import time
from typing import Dict, List, Tuple, Union, Optional, Any

# Bibliotecas de terceros
import numpy as np
import pandas as pd
import unicodedata
import torch
from scipy.sparse import hstack as sparse_hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: Union[str, float, None]) -> str:
    """Limpia texto: lowercase, normaliza unicode, elimina caracteres no deseados, quita espacios extra.
    
    Args:
        text: Texto a limpiar, puede ser None, un valor flotante o una cadena.
        
    Returns:
        Texto limpio como cadena.
    """
    if pd.isna(text):
        return ''
    text = str(text).lower()
    # Normalizar para manejar acentos y ñ correctamente
    text = unicodedata.normalize('NFC', text)
    # Eliminar códigos extraños como _x000d_ primero
    text = re.sub(r'_x000d_', ' ', text)
    # Mantener letras (incluyendo ñ y acentuadas), números, espacios y caracteres útiles como -, /, .
    # Se eliminan otros símbolos como (), [], {}, #, $, %, &, etc.
    text = re.sub(r'[^\w\s\-\/\.ñáéíóúü]', '', text)  # \w incluye letras, números y _
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios extra y saltos de línea remanentes
    return text


def generate_features(
    df: pd.DataFrame, 
    provider_giros: Dict[str, str], 
    embedding_model: SentenceTransformer, 
    fit_vectorizers: bool = True,
    existing_vectorizers: Optional[Tuple[TfidfVectorizer, TfidfVectorizer, TfidfVectorizer]] = None
) -> Tuple[sparse_hstack, Tuple[TfidfVectorizer, TfidfVectorizer, TfidfVectorizer]]:
    """Genera embeddings y características TF-IDF.
    
    Args:
        df: DataFrame con las columnas 'desc', 'prov', y 'sat'.
        provider_giros: Diccionario que mapea proveedores a sus giros/categorías.
        embedding_model: Modelo pre-entrenado de SentenceTransformer.
        fit_vectorizers: Si es True, entrena nuevos vectorizadores. Si es False, usa los proporcionados.
        existing_vectorizers: Tupla de vectorizadores existentes (desc, prov, sat) si fit_vectorizers es False.
        
    Returns:
        Tupla con las características combinadas (X_features) y los vectorizadores utilizados.
        
    Raises:
        ValueError: Si fit_vectorizers es False pero existing_vectorizers es None.
    """
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo para embeddings: {device}")
    embedding_model.to(device)

    # 1. Embeddings de descripciones
    print("Generando embeddings para 'desc'...")
    desc_embeddings = embedding_model.encode(df['desc'].tolist(), batch_size=128, show_progress_bar=True, device=device)

    # 2. Embeddings enriquecidos de proveedores
    print("Generando embeddings para 'prov' (enriquecido)...")
    enriched_prov_texts = [
        f"{prov} proveedor de {provider_giros.get(prov, 'desconocido')}" if provider_giros.get(prov) else prov
        for prov in df['prov']
    ]
    prov_embeddings = embedding_model.encode(enriched_prov_texts, batch_size=128, show_progress_bar=True, device=device)

    # 3. Características TF-IDF
    print("Generando características TF-IDF...")
    if fit_vectorizers:
        print("Ajustando nuevos vectorizadores TF-IDF...")
        vectorizer_desc = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
        vectorizer_prov = TfidfVectorizer(max_features=400)
        vectorizer_sat = TfidfVectorizer(max_features=400)

        desc_tfidf = vectorizer_desc.fit_transform(df['desc'])
        prov_tfidf = vectorizer_prov.fit_transform(df['prov'])
        sat_tfidf = vectorizer_sat.fit_transform(df['sat'])

        vectorizers = (vectorizer_desc, vectorizer_prov, vectorizer_sat)
    else:
        print("Usando vectorizadores TF-IDF existentes...")
        if existing_vectorizers is None:
            raise ValueError("Se requieren existing_vectorizers si fit_vectorizers es False.")
        vectorizer_desc, vectorizer_prov, vectorizer_sat = existing_vectorizers
        desc_tfidf = vectorizer_desc.transform(df['desc'])
        prov_tfidf = vectorizer_prov.transform(df['prov'])
        sat_tfidf = vectorizer_sat.transform(df['sat'])
        vectorizers = existing_vectorizers

    # Combinar características: Embeddings (densos) + TF-IDF (dispersos)
    print("Combinando características...")
    # Convertir embeddings a float32 por si acaso
    desc_embeddings = desc_embeddings.astype(np.float32)
    prov_embeddings = prov_embeddings.astype(np.float32)
    X_features = sparse_hstack((desc_embeddings, prov_embeddings, desc_tfidf, prov_tfidf, sat_tfidf), format='csr')

    end_time = time.time()
    print(f"Generación de features completada en {end_time - start_time:.2f} segundos.")
    return X_features, vectorizers


# --- Función de predicción (actualizada desde el notebook de Colab) ---
def predict_new_data(new_data_df, model_dir, embedding_model_name, confidence_threshold=0.6):
    """Predice subcategorías, confianza y origen (dict/model) para nuevos datos."""
    print("\n--- Iniciando Proceso de Predicción ---")
    start_time = time.time()
    try:
        # Cargar artefactos necesarios
        print("Cargando artefactos desde:", model_dir)
        known_combinations = joblib.load(os.path.join(model_dir, 'known_combinations.joblib'))
        provider_giros = joblib.load(os.path.join(model_dir, 'provider_giros.joblib'))
        le_subcat_original = joblib.load(os.path.join(model_dir, 'label_encoder_subcat_original.joblib'))
        unique_original_labels = joblib.load(os.path.join(model_dir, 'unique_original_labels_after_filter.joblib'))
        vectorizer_desc = joblib.load(os.path.join(model_dir, 'vectorizer_desc.joblib'))
        vectorizer_prov = joblib.load(os.path.join(model_dir, 'vectorizer_prov.joblib'))
        vectorizer_sat = joblib.load(os.path.join(model_dir, 'vectorizer_sat.joblib'))
        vectorizers = (vectorizer_desc, vectorizer_prov, vectorizer_sat)

        # Cargar el mapeo de formato original si existe
        try:
            original_format_mapping = joblib.load(os.path.join(model_dir, 'original_format_mapping.joblib'))
            print(f"Mapeo de formato original cargado con {len(original_format_mapping)} entradas")
        except FileNotFoundError:
            print("Advertencia: No se encontró el mapeo de formato original. Se usarán las etiquetas limpias.")
            original_format_mapping = {}

        # Crear mapeo inverso new_label (remapeado) -> old_label (original codificado)
        inverse_label_mapping = {new_label: old_label for new_label, old_label in enumerate(unique_original_labels)}

        # Cargar modelos de los folds
        models = []
        fold_files = [f for f in os.listdir(model_dir) if f.startswith('lgbm_model_fold_') and f.endswith('.joblib')]
        if not fold_files:
            raise FileNotFoundError("No se encontraron archivos de modelo de folds (.joblib).")
        for fold_file in sorted(fold_files):
            models.append(joblib.load(os.path.join(model_dir, fold_file)))
        if not models:
            raise FileNotFoundError("No se pudieron cargar modelos entrenados.")
        print(f"Cargados {len(models)} modelos de folds.")

        # Cargar modelo de embeddings
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding_model = SentenceTransformer(embedding_model_name, device=device)

    except FileNotFoundError as e:
        print(f"Error al cargar artefactos: {e}")
        print(f"Asegúrate de que el directorio '{model_dir}' existe y contiene los archivos del entrenamiento.")
        return None
    except Exception as e:
        print(f"Error inesperado durante la carga de artefactos: {e}")
        return None

    predictions = []
    predictions_confidence = []
    found_in_dict_list = []
    original_format_predictions = []
    categoria_id_predictions = []

    # Preprocesar nuevos datos
    print("Preprocesando nuevos datos (limpieza de texto)...")
    new_data_processed = new_data_df.copy()
    for col in ['desc', 'prov', 'sat']:
        if col not in new_data_processed.columns:
            raise ValueError(f"La columna '{col}' falta en los nuevos datos.")
        new_data_processed[col] = new_data_processed[col].astype(str)
        new_data_processed[col] = new_data_processed[col].apply(clean_text)

    # Generar características
    print("Generando características para nuevos datos...")
    try:
        X_new_features, _ = generate_features(
            new_data_processed,
            provider_giros,
            embedding_model,
            fit_vectorizers=False,
            existing_vectorizers=vectorizers
        )
    except Exception as e:
        print(f"Error al generar características para nuevos datos: {e}")
        return None

    print(f"Prediciendo {len(new_data_processed)} filas...")
    for idx, row in new_data_processed.iterrows():
        key = (row['desc'], row['prov'], row['sat'])
        if key in known_combinations:
            original_encoded_label_id = known_combinations[key]
            try:
                pred_label = le_subcat_original.inverse_transform([original_encoded_label_id])[0]
                predictions.append(pred_label)
                predictions_confidence.append(1.0)
                found_in_dict_list.append(True)

                # Agregar formato original y categoria_id si existe en el mapeo
                if pred_label in original_format_mapping:
                    original_format_predictions.append(original_format_mapping[pred_label]['original_format'])
                    categoria_id_predictions.append(original_format_mapping[pred_label]['categoria_id'])
                else:
                    # Si no está en el mapeo, usar el valor limpio
                    original_format_predictions.append(pred_label)
                    categoria_id_predictions.append('')
            except ValueError as e:
                print(f"Advertencia: ID {original_encoded_label_id} del dict no encontrado en LabelEncoder original. Fila {idx}. {e}")
                predictions.append("ERROR_DICT_DECODE")
                predictions_confidence.append(1.0)
                found_in_dict_list.append(True)
                original_format_predictions.append("ERROR_DICT_DECODE")
                categoria_id_predictions.append('')
        else:
            X_row_features = X_new_features[idx]
            all_fold_preds_proba = []
            try:
                for model in models:
                    fold_pred_proba = model.predict(X_row_features)[0]
                    all_fold_preds_proba.append(fold_pred_proba)
            except Exception as e:
                print(f"Error al predecir con modelo en fila {idx}: {e}")
                predictions.append("ERROR_PREDICCION")
                predictions_confidence.append(0.0)
                found_in_dict_list.append(False)
                continue

            avg_proba = np.mean(all_fold_preds_proba, axis=0)
            max_prob = np.max(avg_proba)
            pred_id_remapped = np.argmax(avg_proba)

            found_in_dict_list.append(False)

            if max_prob >= confidence_threshold:
                try:
                    original_encoded_label = inverse_label_mapping.get(pred_id_remapped)
                    if original_encoded_label is not None:
                        pred_label = le_subcat_original.inverse_transform([original_encoded_label])[0]
                        predictions.append(pred_label)
                        predictions_confidence.append(max_prob)

                        # Agregar formato original y categoria_id si existe en el mapeo
                        if pred_label in original_format_mapping:
                            original_format_predictions.append(original_format_mapping[pred_label]['original_format'])
                            categoria_id_predictions.append(original_format_mapping[pred_label]['categoria_id'])
                        else:
                            # Si no está en el mapeo, usar el valor limpio
                            original_format_predictions.append(pred_label)
                            categoria_id_predictions.append('')
                    else:
                        print(f"Advertencia: pred_id remapeado {pred_id_remapped} no encontrado en mapeo inverso. Fila {idx}.")
                        predictions.append("ERROR_MAPEADO")
                        predictions_confidence.append(max_prob)
                        original_format_predictions.append("ERROR_MAPEADO")
                        categoria_id_predictions.append('')
                except Exception as e:
                    print(f"Error decodificando etiqueta original para pred_id remapeado {pred_id_remapped}. Fila {idx}. {e}")
                    predictions.append("ERROR_DECODIFICACION")
                    predictions_confidence.append(max_prob)
                    original_format_predictions.append("ERROR_DECODIFICACION")
                    categoria_id_predictions.append('')
            else:
                predictions.append("REVISION_MANUAL")
                predictions_confidence.append(max_prob)
                original_format_predictions.append("REVISION_MANUAL")
                categoria_id_predictions.append('')

    new_data_df['predicted_subcategoria'] = predictions
    new_data_df['prediction_confidence'] = predictions_confidence
    new_data_df['found_in_dictionary'] = found_in_dict_list
    new_data_df['subcategoria_original_format'] = original_format_predictions
    new_data_df['categoria_id'] = categoria_id_predictions

    end_time = time.time()
    print(f"Predicción completada en {end_time - start_time:.2f} segundos.")
    return new_data_df

# --- Ejecución Principal ---
def main():
    """Función principal para ejecutar el pipeline de predicción."""
    batch_start_time = time.time()
    print(f"Iniciando predicción batch para el archivo: {MAP_INPUT_PATH}")
    print(f"Usando modelos y artefactos de: {MODEL_DIR}")

    # 1. Cargar datos de entrada
    try:
        df_map = pd.read_excel(MAP_INPUT_PATH)
        print(f"Datos cargados: {df_map.shape[0]} filas.")
        print("Columnas encontradas:", df_map.columns.tolist())

        # Cargar el catálogo SAT
        sat_path = '/content/drive/MyDrive/Colab_PT/catalogoCFDI4_sat.xlsx'  # Ajusta esta ruta
        primeras_dos_columnas = pd.read_excel(sat_path, usecols=[0, 1])
        sat_descriptions = dict(zip(primeras_dos_columnas.iloc[:, 0], primeras_dos_columnas.iloc[:, 1]))
        df_map['sat'] = df_map['CLAVE PROD.'].map(sat_descriptions).str.lower()

        # Mapeo de columnas
        column_mapping = {
            'DESCRIPCION': 'desc',
            'PROVEEDOR': 'prov',
            'sat': 'sat'
        }
        cols_to_rename = {k: v for k, v in column_mapping.items() if k in df_map.columns and k != v}
        if cols_to_rename:
            print(f"Renombrando columnas: {cols_to_rename}")
            df_map.rename(columns=cols_to_rename, inplace=True)
        else:
            print("No se requiere renombrar columnas (o los nombres ya coinciden).")

        required_cols = ['desc', 'prov', 'sat']
        if not all(col in df_map.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_map.columns]
            raise ValueError(f"Faltan columnas requeridas: {missing}. Verifica el archivo de entrada y el 'column_mapping'.")

    except FileNotFoundError:
        print(f"Error Crítico: No se encontró el archivo en {MAP_INPUT_PATH}")
        exit()
    except ValueError as ve:
        print(ve)
        exit()
    except Exception as e:
        print(f"Error Crítico al cargar o preparar el archivo de entrada: {e}")
        exit()

    # 2. Realizar predicciones
    print("Aplicando modelo para predicciones...")
    try:
        df_predicted = predict_new_data(df_map, MODEL_DIR, EMBEDDING_MODEL_NAME, confidence_threshold=CONFIDENCE_THRESHOLD_PREDICT)
        if df_predicted is None:
            print("La predicción falló. Revisa los mensajes de error anteriores.")
            exit()
    except Exception as e:
        print(f"Error Crítico durante el proceso de predicción: {e}")
        exit()

    # 3. Guardar resultados
    print(f"Guardando resultados en: {MAP_OUTPUT_PATH}")
    try:
        cols_original = [col for col in df_map.columns if col not in ['predicted_subcategoria', 'prediction_confidence', 'found_in_dictionary']]
        cols_new = ['predicted_subcategoria', 'prediction_confidence', 'found_in_dictionary']
        final_cols = cols_original + cols_new
        df_predicted_output = df_predicted[final_cols]
        df_predicted_output.to_excel(MAP_OUTPUT_PATH, index=False, engine='openpyxl')
        print("Archivo de resultados guardado exitosamente.")
    except Exception as e:
        print(f"Error Crítico al guardar el archivo de resultados: {e}")
        print("Asegúrate de tener permisos de escritura y la librería 'openpyxl' instalada (`!pip install openpyxl`).")
        exit()

    batch_end_time = time.time()
    print(f"\nPredicción batch completada en {batch_end_time - batch_start_time:.2f} segundos.")


if __name__ == "__main__":
    main()