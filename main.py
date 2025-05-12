import os
import sys
import logging
import asyncio
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from auth_manager import AuthManager
from invoice_manager import InvoiceManager
from xml_downloader import XMLDownloader
from xml_processor import XMLProcessor
from local_storage_manager import LocalStorageManager
from prediction_service import run_prediction_pipeline
from collections import defaultdict
from supabase_uploader import SupabaseUploader
from utils import limpiar_uuid_dataframe

# Configurar logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Configurar loggers específicos
logger = logging.getLogger('main')
# Configurar el logger de xml_downloader específicamente para reducir mensajes de descarga exitosa
xml_logger = logging.getLogger('xml_downloader')
xml_logger.setLevel(logging.WARNING)  # Solo mostrará WARNING, ERROR y CRITICAL

# --- Directorios Configurables ---
DOWNLOADS_DIR = os.path.expanduser("data")
APP_DATA_DIR = os.path.abspath("data") # Directorio 'data' relativo a la ubicación del script
RESULTS_DIR = os.path.join(DOWNLOADS_DIR, "resultados") # Directorio para los archivos .pkl
MODEL_DIR   = os.path.join(DOWNLOADS_DIR, "model") # Directorio para los archivos .pkl
# ----------------------------------

def main():
    """
    Función principal que orquesta todo el proceso de consulta, descarga y almacenamiento.
    """
    print("\n===== Iniciando proceso de descarga de facturas y XMLs =====\n")
    
    # Asegurar que el directorio de descargas exista
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True) # Asegurar que el directorio de resultados exista

    # 1. Autenticación en el sistema
    auth_manager = AuthManager()
    auth_result = auth_manager.iniciar_sesion()
    
    if not auth_result or not auth_result.get('cookies'):
        logger.error("No se pudo iniciar sesión. Verifique credenciales.")
        return
    
    # Crear sesión de requests con las cookies obtenidas
    import requests
    session = requests.Session()
    session.cookies.update(auth_result.get('cookies', {}))
    session.headers.update({'Content-Type': 'application/json; charset=utf-8'})
    
    # Guardar lista de obras para consultas
    obras_lista = auth_result.get('obras', [])
    # print (obras_lista)
    
    # Limitar a 3 obras para la prueba
    obras_prueba = obras_lista[20] if len(obras_lista) > 20 else obras_lista
    # obras_prueba = obras_lista
    # 2. Inicializar gestor de facturas
    invoice_manager = InvoiceManager(session, base_data_path=APP_DATA_DIR)
    
    try:
        # 3. Consultar obras y obtener facturas
        print("\nConsultando facturas disponibles...")
        resultados = invoice_manager.consultar_facturas_sync(obras_prueba)
        
        # 4. Filtrar facturas pagadas
        print("\nFiltrando facturas pagadas...")
        df_pagadas = invoice_manager.filtrar_facturas_pagadas(resultados["facturas"])
        
        # 4.1 Cargar facturas pagadas a Supabase
        print("\nCargando facturas pagadas a Supabase...")
        try:
            # Inicializar el uploader de Supabase con las credenciales del entorno
            supabase_uploader = SupabaseUploader()
            # Cargar las facturas pagadas a Supabase
            df_pagadas = limpiar_uuid_dataframe(df_pagadas)
            df_pagadas = df_pagadas.drop_duplicates(subset='xml_uuid')
            # df_pagadas.to_excel(os.path.join(RESULTS_DIR, "duplicados_facturas_pagadas.xlsx"))
            resultado_supabase = supabase_uploader.cargar_facturas_pagadas(df_pagadas)
            print(f"Facturas cargadas a Supabase: {resultado_supabase['count']}")
        except Exception as e:
            print(f"Error al cargar facturas a Supabase: {str(e)}")
            logging.error(f"Error al cargar facturas a Supabase: {str(e)}")
        
        # 5. Verificar facturas nuevas
        print("\nComprobando facturas nuevas...")
        df_nuevas = invoice_manager.filtrar_nuevas_facturas(
            df_pagadas
        )
        
        print(f"\nFacturas encontradas: {len(resultados['facturas'])}")
        print(f"Notas de crédito encontradas: {len(resultados['notas'])}")
        print(f"Errores: {len(resultados['errores'])}")
        print(f"\nFacturas pagadas: {len(df_pagadas)}")
        print(f"Facturas nuevas: {len(df_nuevas)}")
        
        # # Guardar a Pickle las facturas nuevas
        # if not df_nuevas.empty:
        #     filename_nuevas = f"facturas_nuevas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        #     pickle_path_nuevas = os.path.join(RESULTS_DIR, filename_nuevas)
        #     df_nuevas.to_pickle(pickle_path_nuevas)
        #     print(f"Facturas nuevas guardadas en: {pickle_path_nuevas}")
            
        # # Guardar a Pickle las facturas pagadas con columnas específicas
        # if not df_pagadas.empty:
        #     df_pagadas_export = df_pagadas.copy()
        #     filename_pagadas = f"facturas_concentrado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        #     pickle_path_pagadas = os.path.join(RESULTS_DIR, filename_pagadas)
        #     df_pagadas_export.to_pickle(pickle_path_pagadas)
        #     print(f"Facturas pagadas guardadas en: {pickle_path_pagadas}")
            
        # 6. Descargar XMLs de facturas nuevas y almacenarlos localmente
        if len(df_nuevas) > 0:
            print(f"\nDescargando {len(df_nuevas)} archivos XML...")
            # Pasar la sesión autenticada al XMLDownloader
            xml_downloader = XMLDownloader(base_data_path=APP_DATA_DIR, session=session)
            # La función sync ahora devuelve información de los archivos descargados, no el DF procesado
            resultados_descarga = xml_downloader.download_all_xmls_sync(df_nuevas)
            
            print(f"\nResultados de la fase de descarga XML:")
            print(f"Total archivos intentados para descarga: {resultados_descarga.get('total_files', 'N/A')}")
            print(f"Éxitos en descarga: {resultados_descarga.get('success_count', 'N/A')}")
            print(f"Errores en descarga: {resultados_descarga.get('error_count', 'N/A')}")
            print(f"Archivos XML descargados en: {resultados_descarga.get('download_folder_path', 'N/A')}")

            downloaded_files_info = resultados_descarga.get('downloaded_xml_files_info', [])

            if downloaded_files_info:
                print(f"\nIniciando procesamiento de {len(downloaded_files_info)} archivos XML descargados...")
                xml_processor = XMLProcessor() # Instanciar el procesador
                all_processed_dataframes = []
                
                # Inicializar el gestor de almacenamiento para registrar XMLs con errores
                storage_manager = LocalStorageManager(base_data_path=APP_DATA_DIR)
                
                # Contadores para agrupar errores por tipo
                error_counters = defaultdict(int)
                error_types = {
                    'sin_rfc': 'No se encontró RFC del Emisor',
                    'sin_conceptos': 'No contiene conceptos',
                    'error_parseo': 'Error al parsear XML',
                    'error_indice': 'Error de índice',
                    'otro_error': 'Error inesperado'
                }
                
                # Crear un mapeo de row_idx a row_data para fácil acceso
                # Esto asume que df_nuevas no ha sido reindexado desde que se pasaron los índices a _download_and_store_xml
                # Si df_nuevas tiene un índice por defecto (0, 1, 2...), iloc[info['row_idx']] funcionará.
                # Si tiene un índice custom, necesitaríamos un mapeo más robusto o pasar el UUID para buscar en df_nuevas.
                # Por simplicidad, asumimos que row_idx es el índice posicional de df_nuevas.

                num_processed_successfully = 0
                num_processing_errors = 0

                for file_info in tqdm(downloaded_files_info, desc="Procesando XMLs", unit="archivo"):
                    local_path = file_info.get('local_path')
                    row_idx = file_info.get('row_idx') # Este es el índice original de df_nuevas
                    xml_uuid_for_log = file_info.get('xml_uuid', 'N/A')

                    if local_path and row_idx is not None:
                        try:
                            # Obtener la 'row_data' correspondiente del DataFrame original 'df_nuevas'
                            # Es crucial que df_nuevas no se modifique entre la descarga y este punto
                            # y que row_idx sea el índice correcto para .iloc
                            row_data_for_processing = df_nuevas.iloc[row_idx]
                            
                            processed_df_single = xml_processor.procesar_xml(local_path, row_data_for_processing)
                            
                            if processed_df_single is not None and not processed_df_single.empty:
                                all_processed_dataframes.append(processed_df_single)
                                num_processed_successfully += 1
                            else:
                                # Verificar si hay errores conocidos en los logs
                                error_type = 'otro_error'  # Valor por defecto
                                
                                # Examinar el contenido del XML para determinar el tipo de error
                                try:
                                    import xml.etree.ElementTree as ET
                                    tree = ET.parse(local_path)
                                    root = tree.getroot()
                                    
                                    # Verificar si falta RFC del emisor
                                    ns = {'cfdi': 'http://www.sat.gob.mx/cfd/4'}
                                    if root.find('.//cfdi:Emisor', ns) is None or \
                                       not root.find('.//cfdi:Emisor', ns).get('Rfc'):
                                        error_type = 'sin_rfc'
                                    # Verificar si no hay conceptos
                                    elif not root.findall('.//cfdi:Concepto', ns):
                                        error_type = 'sin_conceptos'
                                except Exception as xml_parse_error:
                                    error_type = 'error_parseo'
                                    
                                # Registrar el error en el contador de errores
                                error_counters[error_type] += 1
                                
                                # Registrar el XML con error en el archivo JSON
                                storage_manager.add_failed_xml(xml_uuid_for_log, error_type)
                                
                                num_processing_errors += 1
                                # Solo log a nivel DEBUG para no saturar la terminal
                                logger.debug(f"El procesamiento de {local_path} (UUID: {xml_uuid_for_log}) no generó datos o devolvió un DataFrame vacío. Tipo de error: {error_type}")
                                
                        except IndexError:
                            error_type = 'error_indice'
                            error_counters[error_type] += 1
                            storage_manager.add_failed_xml(xml_uuid_for_log, error_type)
                            num_processing_errors += 1
                            # Log a nivel DEBUG
                            logger.debug(f"Error de índice al intentar obtener row_data para el archivo {local_path} con row_idx {row_idx}.")
                            
                        except Exception as e:
                            error_type = 'otro_error'
                            error_counters[error_type] += 1
                            storage_manager.add_failed_xml(xml_uuid_for_log, error_type)
                            num_processing_errors += 1
                            # Log a nivel DEBUG, pero con más detalles
                            logger.debug(f"Error procesando el archivo {local_path} (UUID: {xml_uuid_for_log}): {e}")
                    else:
                        num_processing_errors += 1 # Contar como error si falta path o row_idx
                        logger.debug(f"Información incompleta para procesar un archivo: {file_info}. No se puede procesar.")

                print(f"\nResultados de la fase de procesamiento XML:")
                print(f"Total archivos procesados con éxito: {num_processed_successfully}")
                print(f"Total errores durante el procesamiento: {num_processing_errors}")
                
                # Mostrar resumen de errores por tipo
                if error_counters:
                    print("\nDetalle de XMLs con error:")
                    for error_type, count in error_counters.items():
                        error_desc = error_types.get(error_type, "Error desconocido")
                        print(f"- {count} XMLs con error: {error_desc}")
                    print(f"\nSe ha guardado el registro detallado de XMLs con error en: {storage_manager.failed_xml_file_path}")


                if all_processed_dataframes:
                    final_processed_df = pd.concat(all_processed_dataframes, ignore_index=True)
                    logger.info(f"Total de {len(final_processed_df)} conceptos extraídos y combinados de todos los XMLs.")
                    
                    # Asegurarse de que el directorio MODEL_DIR exista
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    
                    # Cargar el catálogo SAT desde archivo pickle
                    sat_path = os.path.join(MODEL_DIR, "catalogoCFDI4_sat.pkl")
                    if os.path.exists(sat_path):
                        try:
                            # Cargar el diccionario de descripciones SAT desde el archivo pickle
                            primeras_dos_columnas = pd.read_pickle(sat_path)
                            sat_descriptions = dict(zip(primeras_dos_columnas.iloc[:, 0], primeras_dos_columnas.iloc[:, 1]))
                            
                            # Añadir la columna 'sat' mapeando las claves de producto con las descripciones
                            if 'clave_producto' in final_processed_df.columns:
                                final_processed_df['sat'] = final_processed_df['clave_producto'].map(sat_descriptions).str.lower()
                                logger.info(f"Columna 'sat' añadida al DataFrame con descripciones del catálogo SAT.")
                                # Agregar columna uuid_concepto como UUID v5 único
                                from utils import concept_uuid5
                                final_processed_df['uuid_concepto'] = final_processed_df.apply(concept_uuid5, axis=1)
                                logger.info(f"Columna 'uuid_concepto' añadida al DataFrame como UUID v5 único para cada concepto.")
                            else:
                                logger.warning(f"No se encontró la columna 'CLAVE PROD.' en el DataFrame. No se pudo añadir la columna 'sat'.")
                        except Exception as e:
                            logger.error(f"Error al cargar o aplicar el catálogo SAT: {str(e)}")
                    else:
                        logger.warning(f"No se encontró el archivo del catálogo SAT en {sat_path}. No se añadió la columna 'sat'.")
                    

                    # # Guardar el DataFrame procesado final
                    # processed_filename = f"xml_data_procesado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    # processed_pickle_path = os.path.join(RESULTS_DIR, processed_filename)
                    # final_processed_df.to_pickle(processed_pickle_path)
                    # print(f"Datos XML procesados guardados en: {processed_pickle_path}")
                    
                    # Ejecutar predicción sobre los datos procesados
                    print("\nIniciando proceso de predicción automática...")
                    try:
                        predicted_df = run_prediction_pipeline(
                            final_processed_df,
                            DOWNLOADS_DIR,
                            RESULTS_DIR,
                            MODEL_DIR,
                            confidence_threshold=0.6
                        )
                        if predicted_df is not None:
                            print("Predicción completada exitosamente.")
                            # Definir columnas explícitamente
                            columnas_a_sumar = [
                                'cantidad', 'subtotal', 'descuento', 'venta_tasa_0', 'venta_tasa_16',
                                'total_iva', 'retencion_iva', 'retencion_isr', 'total_ish', 'total'
                            ]
                            columnas_a_conservar = [
                                'obra', 'cuenta_gasto', 'proveedor', 'residente', 'folio', 'estatus',
                                'fecha_factura', 'fecha_recepcion', 'fecha_pagada', 'fecha_autorizacion',
                                'clave_producto', 'clave_unidad', 'descripcion', 'unidad', 'precio_unitario',
                                'moneda', 'serie', 'url_pdf', 'url_oc', 'url_rem', 'xml_uuid', 'encontrado_en_diccionario',
                                'confianza_prediccion', 'subcategoria', 'sat', 'tipo_gasto'# Añadidas para preservar las predicciones
                            ]

                            # Mapear categorías
                            predicted_df = supabase_uploader.map_categoria(predicted_df)

                            # Agrupar
                            predicted_df = predicted_df.groupby(
                                ['uuid_concepto'], as_index=False
                            ).agg(
                                {
                                    **{col: 'sum' for col in columnas_a_sumar},
                                    **{col: 'first' for col in columnas_a_conservar}
                                }
                            )

                            # Subir predicciones a Supabase
                            supabase_uploader.subir_predicciones_portal_desglosado(predicted_df)
                            # # Guardar el DataFrame procesado final
                            # predicted_filename = f"2facturas_predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                            # predicted_path = os.path.join(RESULTS_DIR, predicted_filename)
                            # predicted_df.to_excel(predicted_path)
                            # print(f"Datos predicciones guardados en: {predicted_path}")
                            
                        else:
                            print("No se pudo completar la predicción. Consulte los logs para más detalles.")
                    except Exception as e:
                        logger.error(f"Error durante el proceso de predicción: {str(e)}")
                        print(f"Error durante el proceso de predicción: {str(e)}")
                else:
                    logger.warning("No se obtuvieron datos procesados de ningún archivo XML o todos los procesamientos fallaron.")
                    print("No se generó ningún archivo de datos XML procesados.")
            elif resultados_descarga.get('success_count', 0) == 0 and resultados_descarga.get('total_files', 0) > 0:
                logger.warning("No se descargó ningún archivo XML, por lo tanto no hay nada que procesar.")
                print("No se descargaron archivos XML, omitiendo fase de procesamiento.")
            else: # Cubre el caso donde df_nuevas era > 0 pero no se descargó nada (ej. todas las descargas fallaron)
                logger.info("No hay archivos XML descargados para procesar.")
                print("No hay archivos XML descargados para procesar.")
        else:
            print("\nNo hay facturas nuevas para descargar y procesar XMLs.")
            logger.info("No hay facturas nuevas, omitiendo descarga y procesamiento de XML.")
            
        print("\n===== Proceso completado =====\n")
            
    except Exception as e:
        logger.error(f"Error durante el proceso: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
