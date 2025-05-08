# -*- coding: utf-8 -*-
"""
Predictor para el clasificador con validación cruzada de 5 folds.

Este módulo realiza predicciones utilizando el modelo entrenado para clasificar facturas.
Se integra con el flujo principal de la aplicación.
"""

# Bibliotecas estándar
import os
import time
import re
from typing import Dict, List, Tuple, Union, Optional, Any

# Bibliotecas de terceros
import numpy as np
import pandas as pd
import joblib
import torch
import unicodedata
from scipy.sparse import hstack as sparse_hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class Predictor:
    """
    Clase para realizar predicciones de subcategorías en datos de facturas.
    """
    def __init__(self, downloads_dir, results_dir, model_dir, embedding_model_name='sentence-transformers/stsb-xlm-r-multilingual'):
        """
        Inicializa el predictor con las rutas de directorios y el modelo de embeddings.
        
        Args:
            downloads_dir: Directorio base para descargas
            results_dir: Directorio para resultados
            model_dir: Directorio donde se encuentran los archivos del modelo
            embedding_model_name: Nombre del modelo de embeddings a utilizar
        """
        self.downloads_dir = downloads_dir
        self.results_dir = results_dir
        self.model_dir = model_dir
        self.embedding_model_name = embedding_model_name
        self.confidence_threshold = 0.6  # Umbral de confianza por defecto
        
    def clean_text(self, text: Union[str, float, None]) -> str:
        """
        Limpia texto: lowercase, normaliza unicode, elimina caracteres no deseados, quita espacios extra.
        
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
        self,
        df: pd.DataFrame, 
        provider_giros: Dict[str, str], 
        embedding_model: SentenceTransformer, 
        fit_vectorizers: bool = True,
        existing_vectorizers: Optional[Tuple[TfidfVectorizer, TfidfVectorizer, TfidfVectorizer]] = None
    ) -> Tuple[sparse_hstack, Tuple[TfidfVectorizer, TfidfVectorizer, TfidfVectorizer]]:
        """
        Genera embeddings y características TF-IDF.
        
        Args:
            df: DataFrame con las columnas 'descripcion', 'proveedor', y 'sat'.
            provider_giros: Diccionario que mapea proveedores a sus giros/categorías.
            embedding_model: Modelo pre-entrenado de SentenceTransformer.
            fit_vectorizers: Si es True, entrena nuevos vectorizadores. Si es False, usa los proporcionados.
            existing_vectorizers: Tupla de vectorizadores existentes (descripcion, proveedor, sat) si fit_vectorizers es False.
            
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
        print("Generando embeddings para 'descripcion'...")
        desc_embeddings = embedding_model.encode(df['descripcion'].tolist(), batch_size=128, show_progress_bar=True, device=device)

        # 2. Embeddings enriquecidos de proveedores
        print("Generando embeddings para 'proveedor' (enriquecido)...")
        enriched_prov_texts = [
            f"{prov} proveedor de {provider_giros.get(prov, 'desconocido')}" if provider_giros.get(prov) else prov
            for prov in df['proveedor']
        ]
        prov_embeddings = embedding_model.encode(enriched_prov_texts, batch_size=128, show_progress_bar=True, device=device)

        # 3. Características TF-IDF
        print("Generando características TF-IDF...")
        if fit_vectorizers:
            print("Ajustando nuevos vectorizadores TF-IDF...")
            vectorizer_desc = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
            vectorizer_prov = TfidfVectorizer(max_features=400)
            vectorizer_sat = TfidfVectorizer(max_features=400)

            desc_tfidf = vectorizer_desc.fit_transform(df['descripcion'])
            prov_tfidf = vectorizer_prov.fit_transform(df['proveedor'])
            sat_tfidf = vectorizer_sat.fit_transform(df['sat'])

            vectorizers = (vectorizer_desc, vectorizer_prov, vectorizer_sat)
        else:
            print("Usando vectorizadores TF-IDF existentes...")
            if existing_vectorizers is None:
                raise ValueError("Se requieren existing_vectorizers si fit_vectorizers es False.")
            vectorizer_desc, vectorizer_prov, vectorizer_sat = existing_vectorizers
            desc_tfidf = vectorizer_desc.transform(df['descripcion'])
            prov_tfidf = vectorizer_prov.transform(df['proveedor'])
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

    def predict(self, df_to_predict, confidence_threshold=None):
        """
        Predice subcategorías, confianza y origen (dict/model) para datos de facturas.
        
        Args:
            df_to_predict: DataFrame con los datos a predecir
            confidence_threshold: Umbral de confianza opcional (usa el predeterminado si es None)
            
        Returns:
            DataFrame con los datos originales y las predicciones añadidas
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        print("\n--- Iniciando Proceso de Predicción ---")
        start_time = time.time()
        try:
            # Cargar artefactos necesarios
            print("Cargando artefactos desde:", self.model_dir)
            known_combinations = joblib.load(os.path.join(self.model_dir, 'modelo_v5/known_combinations.joblib'))
            provider_giros = joblib.load(os.path.join(self.model_dir, 'modelo_v5/provider_giros.joblib'))
            le_subcat_original = joblib.load(os.path.join(self.model_dir, 'modelo_v5/label_encoder_subcat_original.joblib'))
            unique_original_labels = joblib.load(os.path.join(self.model_dir, 'modelo_v5/unique_original_labels_after_filter.joblib'))
            vectorizer_desc = joblib.load(os.path.join(self.model_dir, 'modelo_v5/vectorizer_desc.joblib'))
            vectorizer_prov = joblib.load(os.path.join(self.model_dir, 'modelo_v5/vectorizer_prov.joblib'))
            vectorizer_sat = joblib.load(os.path.join(self.model_dir, 'modelo_v5/vectorizer_sat.joblib'))
            vectorizers = (vectorizer_desc, vectorizer_prov, vectorizer_sat)

            # Cargar el mapeo de formato original si existe
            try:
                original_format_mapping = joblib.load(os.path.join(self.model_dir, 'modelo_v5/original_format_mapping.joblib'))
                print(f"Mapeo de formato original cargado con {len(original_format_mapping)} entradas")
            except FileNotFoundError:
                print("Advertencia: No se encontró el mapeo de formato original. Se usarán las etiquetas limpias.")
                original_format_mapping = {}

            # Crear mapeo inverso new_label (remapeado) -> old_label (original codificado)
            inverse_label_mapping = {new_label: old_label for new_label, old_label in enumerate(unique_original_labels)}

            # Cargar modelos de los folds
            models = []
            fold_files = [f for f in os.listdir(os.path.join(self.model_dir, 'modelo_v5')) 
                         if f.startswith('lgbm_model_fold_') and f.endswith('.joblib')]
            if not fold_files:
                raise FileNotFoundError("No se encontraron archivos de modelo de folds (.joblib).")
            for fold_file in sorted(fold_files):
                models.append(joblib.load(os.path.join(self.model_dir, 'modelo_v5', fold_file)))
            if not models:
                raise FileNotFoundError("No se pudieron cargar modelos entrenados.")
            print(f"Cargados {len(models)} modelos de folds.")

            # Cargar modelo de embeddings
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

        except FileNotFoundError as e:
            print(f"Error al cargar artefactos: {e}")
            print(f"Asegúrate de que el directorio '{self.model_dir}/modelo_v5' existe y contiene los archivos del entrenamiento.")
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
        new_data_processed = df_to_predict.copy()
        for col in ['descripcion', 'proveedor', 'sat']:
            if col not in new_data_processed.columns:
                raise ValueError(f"La columna '{col}' falta en los nuevos datos.")
            new_data_processed[col] = new_data_processed[col].astype(str)
            new_data_processed[col] = new_data_processed[col].apply(self.clean_text)

        # Generar características
        print("Generando características para nuevos datos...")
        try:
            X_new_features, _ = self.generate_features(
                new_data_processed,
                provider_giros,
                embedding_model,
                fit_vectorizers=False,
                existing_vectorizers=vectorizers
            )
        except Exception as e:
            print(f"Error al generar características: {e}")
            return None

        # Verificar si hay coincidencias exactas en el diccionario
        print("Verificando coincidencias exactas en el diccionario...")
        for idx, row in new_data_processed.iterrows():
            desc = row['descripcion']
            prov = row['proveedor']
            sat_code = row['sat']

            # Buscar coincidencia exacta en el diccionario
            key = (desc, prov, sat_code)
            if key in known_combinations:
                # Coincidencia exacta encontrada
                original_encoded_label_id = known_combinations[key]
                try:
                    # Decodificar el valor numérico al nombre de la subcategoría
                    pred_label = le_subcat_original.inverse_transform([original_encoded_label_id])[0]
                    found_in_dict = True
                    confidence = 1.0  # Máxima confianza para coincidencias exactas
                except ValueError as e:
                    print(f"Advertencia: ID {original_encoded_label_id} del dict no encontrado en LabelEncoder original. Fila {idx}. {e}")
                    predictions.append("ERROR_DICT_DECODE")
                    predictions_confidence.append(1.0)
                    found_in_dict_list.append(True)
                    original_format_predictions.append("ERROR_DICT_DECODE")
                    categoria_id_predictions.append(None)
                    continue
            else:
                # No hay coincidencia exacta, usar modelo
                # Obtener predicciones de todos los modelos de fold
                all_fold_preds_proba = []

                for model in models:
                    # Para LightGBM, el método predict devuelve directamente las probabilidades
                    single_sample_features = X_new_features[idx]
                    fold_pred_proba = model.predict(single_sample_features)[0]
                    all_fold_preds_proba.append(fold_pred_proba)
                    
                # Combinar probabilidades de todos los folds y obtener la predicción del ensemble
                avg_proba = np.mean(all_fold_preds_proba, axis=0)
                max_prob = np.max(avg_proba)
                pred_id_remapped = np.argmax(avg_proba)
                
                # Usar la predicción del ensemble
                ensemble_pred = pred_id_remapped
                confidence = max_prob

                # Usar la predicción del ensemble
                original_encoded_label = inverse_label_mapping.get(ensemble_pred)
                if original_encoded_label is None:
                    # Error: la etiqueta mapeada no existe en el mapeo inverso
                    print(f"Advertencia: pred_id remapeado {ensemble_pred} no encontrado en mapeo inverso. Fila {idx}.")
                    predictions.append("ERROR_MAPEADO")
                    predictions_confidence.append(confidence)
                    found_in_dict_list.append(False)
                    original_format_predictions.append("ERROR_MAPEADO")
                    categoria_id_predictions.append(None)
                    continue
                    
                # Decodificar la etiqueta original
                try:
                    pred_label = le_subcat_original.inverse_transform([original_encoded_label])[0]
                    found_in_dict = False
                except Exception as e:
                    print(f"Error decodificando etiqueta original para pred_id remapeado {ensemble_pred}. Fila {idx}. {e}")
                    predictions.append("ERROR_DECODIFICACION")
                    predictions_confidence.append(confidence)
                    found_in_dict_list.append(False)
                    original_format_predictions.append("ERROR_DECODIFICACION")
                    categoria_id_predictions.append(None)
                    continue

            # Obtener formato original si está disponible
            if pred_label in original_format_mapping:
                if isinstance(original_format_mapping[pred_label], dict):
                    # Si es un diccionario con estructura {original_format, categoria_id}
                    original_format = original_format_mapping[pred_label].get('original_format', pred_label)
                    categoria_id = original_format_mapping[pred_label].get('categoria_id', '')
                else:
                    # Si es simplemente el valor de formato original
                    original_format = original_format_mapping[pred_label]
                    # Extraer ID de categoría si está en formato "ID - Nombre"
                    categoria_id = ''
                    if isinstance(original_format, str) and ' - ' in original_format:
                        parts = original_format.split(' - ', 1)
                        if parts[0].isdigit():
                            categoria_id = parts[0]  # Mantener como string para consistencia
            else:
                # Si no está en el mapeo, usar el valor limpio
                original_format = pred_label
                categoria_id = ''
                
            # Almacenar resultados
            predictions.append(pred_label)
            predictions_confidence.append(confidence)
            found_in_dict_list.append(found_in_dict)
            original_format_predictions.append(original_format)
            categoria_id_predictions.append(categoria_id)

        # Preparar el DataFrame de salida
        result_df = df_to_predict.copy()
        
        # Añadir las columnas de predicción manteniendo las columnas originales
        # Solo se agregan 4 columnas adicionales como se requiere
        result_df['subcategoria_predicha'] = original_format_predictions  # Subcategoria en formato original
        result_df['categoria_id'] = categoria_id_predictions           # Categoria correspondiente
        result_df['encontrado_en_diccionario'] = found_in_dict_list     # Si estaba en el diccionario
        result_df['confianza_prediccion'] = predictions_confidence      # Porcentaje de confianza
        
        # Marcar filas como REVISION_MANUAL si la confianza es menor que el umbral
        revision_mask = result_df['confianza_prediccion'] < confidence_threshold
        if revision_mask.any():
            result_df.loc[revision_mask, 'subcategoria_predicha'] = 'REVISION_MANUAL'

        # Guardar resultados
        output_filename = f"facturas_predichas_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        output_path = os.path.join(self.results_dir, output_filename)
        result_df.to_excel(output_path, index=False)
        print(f"\nPredicciones guardadas en: {output_path}")
        
        # Estadísticas de predicción
        total_samples = len(result_df)
        dict_matches = sum(found_in_dict_list)
        model_predictions = total_samples - dict_matches
        confident_predictions = sum(result_df['confianza_prediccion'] >= confidence_threshold)
        
        print(f"\nEstadísticas de predicción:")
        print(f"Total de muestras procesadas: {total_samples}")
        print(f"Coincidencias exactas en diccionario: {dict_matches} ({dict_matches/total_samples*100:.1f}%)")
        print(f"Predicciones del modelo: {model_predictions} ({model_predictions/total_samples*100:.1f}%)")
        print(f"Predicciones confiables (confianza >= {confidence_threshold}): {confident_predictions} ({confident_predictions/total_samples*100:.1f}%)")
        
        end_time = time.time()
        print(f"\nProceso de predicción completado en {end_time - start_time:.2f} segundos.")
        
        return result_df
