import os
import sys
import logging
import asyncio
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from auth_manager import AuthManager
from invoice_manager import InvoiceManager
from xml_downloader import XMLDownloader, XMLStorageManager
from xml_processor import XMLProcessor
from supabase_storage_manager import SupabaseStorageManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

def load_supabase_credentials():
    """
    Carga las credenciales de Supabase desde variables de entorno
    
    Returns:
        tuple: (supabase_url, supabase_key)
    """
    # Cargar variables de entorno
    load_dotenv()
    
    # Obtener credenciales de Supabase
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("No se encontraron credenciales de Supabase en las variables de entorno")
        logger.error("Por favor configure SUPABASE_URL y SUPABASE_KEY en el archivo .env")
        sys.exit(1)
    
    return supabase_url, supabase_key

def main():
    """
    Función principal que orquesta todo el proceso de consulta, descarga y almacenamiento.
    """
    print("\n===== Iniciando proceso de descarga de facturas y XMLs =====\n")
    
    # Cargar credenciales de Supabase
    supabase_url, supabase_key = load_supabase_credentials()
    
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
    
    # Limitar a 3 obras para la prueba
    # obras_prueba = obras_lista[:4] if len(obras_lista) > 4 else obras_lista
    obras_prueba = obras_lista
    # 2. Inicializar gestor de facturas
    invoice_manager = InvoiceManager(session)
    
    try:
        # 3. Consultar obras y obtener facturas
        print("\nConsultando facturas disponibles...")
        resultados = invoice_manager.consultar_facturas_sync(obras_prueba)
        
        # 4. Filtrar facturas pagadas
        print("\nFiltrando facturas pagadas...")
        df_pagadas = invoice_manager.filtrar_facturas_pagadas(resultados["facturas"])
        
        # 5. Verificar facturas nuevas usando Supabase Storage
        print("\nComprobando facturas nuevas...")
        df_nuevas = invoice_manager.filtrar_nuevas_facturas(
            df_pagadas, 
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )
        
        print(f"\nFacturas encontradas: {len(resultados['facturas'])}")
        print(f"Notas de crédito encontradas: {len(resultados['notas'])}")
        print(f"Errores: {len(resultados['errores'])}")
        print(f"\nFacturas pagadas: {len(df_pagadas)}")
        print(f"Facturas nuevas: {len(df_nuevas)}")
        
        # Guardar a Excel las facturas nuevas
        if not df_nuevas.empty:
            excel_path = os.path.expanduser(f"~/Downloads/facturas_nuevas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            df_nuevas.to_excel(excel_path, index=False)
            print(f"Facturas guardadas en: {excel_path}")
            
        # Guardar a Excel las facturas pagadas con columnas específicas
        if not df_pagadas.empty:
            # Crear una copia del DataFrame y seleccionar solo las columnas requeridas
            # df_pagadas_export = df_pagadas[['estatus', 'fecha_autorizacion', 'fecha_pagada', 'url_pdf', 'url_oc', 'url_rem', 'xml_uuid']].copy()
            df_pagadas_export = df_pagadas.copy()
            pagadas_excel_path = os.path.expanduser(f"~/Downloads/facturas_pagadas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            df_pagadas_export.to_excel(pagadas_excel_path, index=False)
            print(f"Facturas pagadas guardadas en: {pagadas_excel_path}")
            
        # 6. Descargar XMLs de facturas nuevas y almacenarlos en Supabase
        if len(df_nuevas) > 0:
            print(f"\nDescargando {len(df_nuevas)} archivos XML...")
            xml_downloader = XMLDownloader(supabase_url, supabase_key)
            xml_downloader._initialize_storage_manager() 
            storage_manager = xml_downloader.xml_storage
            
            resultados_descarga = xml_downloader.download_all_xmls_sync(df_nuevas)
            
            print(f"\nResultados de descarga y procesamiento XML:")
            print(f"Total archivos intentados: {resultados_descarga.get('total_files', 'N/A')}")
            print(f"Éxitos (descarga y procesamiento): {resultados_descarga.get('success_count', 'N/A')}")
            print(f"Errores (descarga o procesamiento): {resultados_descarga.get('error_count', 'N/A')}")

            df_procesado = resultados_descarga.get('processed_dataframe')

            # 8. Guardar Resultados Procesados
            if df_procesado is not None and not df_procesado.empty:
                print(f"\nGuardando {len(df_procesado)} conceptos en archivo Excel...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"desglosado_{timestamp}.xlsx"
                output_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
                os.makedirs(output_folder, exist_ok=True) 
                output_path = os.path.join(output_folder, output_filename)
                
                try:
                    df_procesado.to_excel(output_path, index=False)
                    print(f"\nArchivo Excel guardado exitosamente en: {output_path}")
                except Exception as save_err:
                    logger.error(f"Error al guardar el archivo Excel en {output_path}: {save_err}")
                    print(f"\nError al guardar el archivo Excel. Revise los logs.")
            else:
                print("\nNo se extrajeron conceptos de los XML descargados.")
        else:
            print("\nNo hay nuevas facturas para descargar.")
        
        print("\n===== Proceso completado =====\n")
            
    except Exception as e:
        logger.error(f"Error durante el proceso: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")

async def main_async():
    """
    Versión asíncrona de la función principal.
    """
    # Cargar credenciales de Supabase
    supabase_url, supabase_key = load_supabase_credentials()
    
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

    # 2. Inicializar gestor de facturas
    invoice_manager = InvoiceManager(session)
    
    try:
        # 3. Consultar obras y obtener facturas
        resultados = await invoice_manager.consultar_facturas(obras_lista)
        
        # 4. Filtrar facturas pagadas
        df_pagadas = invoice_manager.filtrar_facturas_pagadas(resultados["facturas"])
        
        # 5. Verificar facturas nuevas usando Supabase Storage
        df_nuevas = invoice_manager.filtrar_nuevas_facturas(
            df_pagadas, 
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )
        
        # Guardar a Excel las facturas nuevas
        if not df_nuevas.empty:
            excel_path = os.path.expanduser(f"~/Downloads/facturas_nuevas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            df_nuevas.to_excel(excel_path, index=False)
            print(f"Facturas guardadas en: {excel_path}")
        
        # 6. Descargar XMLs de facturas nuevas
        if len(df_nuevas) > 0:
            xml_downloader = XMLDownloader(supabase_url, supabase_key)
            await xml_downloader.download_all_xmls(df_nuevas)
        
    except Exception as e:
        logger.error(f"Error durante el proceso: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
