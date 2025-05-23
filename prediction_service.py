# -*- coding: utf-8 -*-
"""
Servicio de predicción para integrar el predictor con el flujo principal.

Este módulo proporciona funciones para ejecutar predicciones sobre los datos
procesados de facturas, integrando el flujo de trabajo principal con el predictor.
"""

import os
import logging
import pandas as pd
from datetime import datetime
from predictor import Predictor

# Configurar logging
logger = logging.getLogger('prediction_service')

class PredictionService:
    """
    Servicio para gestionar la integración entre el procesamiento de facturas y la predicción.
    """
    
    def __init__(self, downloads_dir, results_dir, model_dir):
        """
        Inicializa el servicio de predicción.
        
        Args:
            downloads_dir: Directorio base para descargas
            results_dir: Directorio para resultados
            model_dir: Directorio donde se encuentran los archivos del modelo
        """
        self.downloads_dir = downloads_dir
        self.results_dir = results_dir
        self.model_dir = model_dir
        self.predictor = Predictor(downloads_dir, results_dir, model_dir)
        
    def run_prediction_from_dataframe(self, df, confidence_threshold=0.6):
        """
        Ejecuta la predicción directamente desde un DataFrame.
        
        Args:
            df: DataFrame con los datos procesados de facturas
            confidence_threshold: Umbral de confianza para las predicciones
            
        Returns:
            DataFrame con las predicciones añadidas o None si hay error
        """
        logger.info(f"Iniciando predicción con {len(df)} registros")
        
        try:
            # Verificar que el DataFrame tiene las columnas necesarias
            required_columns = ['descripcion', 'proveedor', 'sat']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Faltan columnas requeridas en el DataFrame: {missing_columns}")
                print(f"Error: Faltan columnas requeridas para la predicción: {missing_columns}")
                return None
                
            # Ejecutar predicción
            print(f"\nEjecutando predicción con {len(df)} registros...")
            result_df = self.predictor.predict(df, confidence_threshold)
            
            if result_df is not None:
                logger.info(f"Predicción completada exitosamente para {len(result_df)} registros")
                return result_df
            else:
                logger.error("La predicción falló y devolvió None")
                return None
                
        except Exception as e:
            logger.error(f"Error durante la predicción: {str(e)}")
            print(f"Error durante la predicción: {str(e)}")
            import traceback
            logger.error(f"Traceback completo: {traceback.format_exc()}")
            return None
    
    def run_prediction_from_file(self, pickle_path, confidence_threshold=0.6):
        """
        Carga un archivo pickle y ejecuta la predicción sobre los datos.
        
        Args:
            pickle_path: Ruta al archivo pickle con los datos procesados
            confidence_threshold: Umbral de confianza para las predicciones
            
        Returns:
            DataFrame con las predicciones añadidas o None si hay error
        """
        logger.info(f"Cargando datos desde {pickle_path}")
        
        try:
            # Verificar que el archivo existe
            if not os.path.exists(pickle_path):
                logger.error(f"El archivo {pickle_path} no existe")
                print(f"Error: El archivo {pickle_path} no existe")
                return None
                
            # Cargar el DataFrame
            df = pd.read_pickle(pickle_path)
            print(f"Datos cargados desde {pickle_path}: {len(df)} registros")
            
            # Ejecutar predicción
            return self.run_prediction_from_dataframe(df, confidence_threshold)
                
        except Exception as e:
            logger.error(f"Error al cargar o procesar el archivo {pickle_path}: {str(e)}")
            print(f"Error al cargar o procesar el archivo {pickle_path}: {str(e)}")
            import traceback
            logger.error(f"Traceback completo: {traceback.format_exc()}")
            return None

# Función de utilidad para ejecutar la predicción desde el flujo principal
def run_prediction_pipeline(processed_df, downloads_dir, results_dir, model_dir, confidence_threshold=0.6):
    """
    Función de utilidad para ejecutar la predicción desde el flujo principal.
    
    Args:
        processed_df: DataFrame con los datos procesados de facturas
        downloads_dir: Directorio base para descargas
        results_dir: Directorio para resultados
        model_dir: Directorio donde se encuentran los archivos del modelo
        confidence_threshold: Umbral de confianza para las predicciones
        
    Returns:
        DataFrame con las predicciones añadidas o None si hay error
    """
    print("\n===== Iniciando proceso de predicción =====\n")
    
    # Crear el servicio de predicción
    prediction_service = PredictionService(downloads_dir, results_dir, model_dir)
    
    # Ejecutar la predicción
    result_df = prediction_service.run_prediction_from_dataframe(processed_df, confidence_threshold)
    
    if result_df is not None:
        print("\n===== Proceso de predicción completado exitosamente =====\n")
    else:
        print("\n===== Proceso de predicción falló =====\n")
        
    return result_df
