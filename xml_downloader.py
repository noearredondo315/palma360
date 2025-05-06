import os
import logging
import asyncio
import hashlib
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from concurrent.futures import ThreadPoolExecutor
from supabase import create_client
from xml_processor import XMLProcessor  # Import the processor
from tqdm.auto import tqdm  # Import tqdm for progress bars
from requests.exceptions import RequestException, HTTPError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('xml_downloader')

class XMLStorageManager:
    """
    Clase para gestionar la descarga y almacenamiento de archivos XML en Supabase Storage.
    
    Esta clase descarga archivos XML desde URLs proporcionadas en un DataFrame y los
    almacena en un bucket de Supabase Storage organizados por fecha.
    """
    
    def __init__(self, supabase_url, supabase_key, bucket_name="xml_files"):
        """
        Inicializa el gestor de almacenamiento de archivos XML en Supabase.
        
        Args:
            supabase_url (str): URL de la API de Supabase
            supabase_key (str): Clave de API de Supabase
            bucket_name (str): Nombre del bucket para los archivos XML
        """
        self.supabase = create_client(supabase_url, supabase_key)
        self.bucket_name = bucket_name
        self.current_date = datetime.now().strftime('%Y%m%d')
        # Asegurar que folder_path siempre tenga este formato
        self.folder_path = f"Corte_al_{self.current_date}"

        # Asegurarse de que el bucket existe
        self._ensure_bucket_exists()
        # Crear la carpeta principal una sola vez al inicio
        # Nota: Se crea implícitamente al subir, pero verificamos para log
        self._create_folder_if_not_exists(self.folder_path)

    def _ensure_bucket_exists(self):
        """
        Asegura que el bucket necesario existe en Supabase Storage.
        """
        try:
            # Verificar si el bucket existe
            buckets = self.supabase.storage.list_buckets()
            
            # Obtener nombres de los buckets existentes
            bucket_names = [bucket.name for bucket in buckets if hasattr(bucket, 'name')]
            
            logger.info(f"Buckets encontrados: {bucket_names}")
            
            # Crear bucket si no existe
            if str(self.bucket_name) not in bucket_names:
                logger.info(f"Creando bucket para XML: {self.bucket_name}")
                self.supabase.storage.create_bucket(str(self.bucket_name), options={"public": False})
            else:
                logger.info(f"Bucket XML '{self.bucket_name}' ya existe, omitiendo creación")
                
        except Exception as e:
            # Manejar específicamente el error de bucket duplicado
            error_str = str(e)
            if "Duplicate" in error_str and "already exists" in error_str:
                logger.warning(f"El bucket ya existe pero no es visible por RLS. Continuando...")
                return
            else:
                logger.error(f"Error al verificar/crear bucket XML: {error_str}")
                import traceback
                logger.error(f"Traceback completo: {traceback.format_exc()}")
                raise

    def _create_folder_if_not_exists(self, folder_path):
        """
        Verifica si una carpeta existe dentro del bucket, y si no, la crea implícitamente.
        
        Args:
            folder_path (str): Ruta de la carpeta a verificar/crear
            
        Returns:
            bool: True si la carpeta existe o se creó exitosamente
        """
        try:
            # En Supabase Storage, las carpetas se crean implícitamente al subir un archivo
            # con un prefijo de ruta. Subimos un archivo vacío para crear la carpeta.
            placeholder_path = f"{folder_path}/.placeholder"
            
            # Verificar si la carpeta ya existe listando archivos con ese prefijo
            files = self.supabase.storage.from_(self.bucket_name).list(folder_path)
            
            # Si ya existen archivos, la carpeta ya existe
            if files:
                logger.info(f"Carpeta '{folder_path}' ya existe en bucket {self.bucket_name}")
                return True
                
            # Crear un archivo placeholder para crear la carpeta
            self.supabase.storage.from_(self.bucket_name).upload(
                placeholder_path,
                b"",  # Contenido vacío
                {"content-type": "text/plain"}
            )
            
            logger.info(f"Carpeta '{folder_path}' creada en bucket {self.bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error al crear carpeta '{folder_path}': {str(e)}")
            return False
    
    def _file_exists_in_storage(self, bucket, file_path):
        """Verifica si un archivo existe en Supabase Storage.
        
        Args:
            bucket (str): Nombre del bucket
            file_path (str): Ruta del archivo en el bucket
            
        Returns:
            bool: True si el archivo existe, False en caso contrario
        """
        try:
            # Intentar obtener metadatos del archivo
            self.supabase.storage.from_(bucket).get_public_url(file_path)
            # Si no levanta excepción, el archivo existe
            return True
        except Exception:
            # Si hay una excepción, asumimos que el archivo no existe
            return False
            
    async def _upload_with_retry(self, bucket, file_path, content, executor, max_retries=3, base_delay=2):
        """Sube un archivo a Supabase Storage con retries exponenciales.
        
        Args:
            bucket (str): Nombre del bucket
            file_path (str): Ruta del archivo en el bucket
            content (bytes): Contenido del archivo
            executor (ThreadPoolExecutor): Ejecutor para operaciones bloqueantes
            max_retries (int): Número máximo de intentos
            base_delay (int): Retardo base en segundos para backoff exponencial
            
        Returns:
            dict: Resultado de la operación de subida
        """
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                # Intentar subir el archivo
                result = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    self.supabase.storage.from_(bucket).upload,
                    file_path,
                    content,
                    {"content-type": "application/xml", "upsert": "true"}
                )
                return result
            except Exception as e:
                last_error = e
                attempt += 1
                
                # Verificar si a pesar del error, el archivo se subió correctamente
                file_exists = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: self._file_exists_in_storage(bucket, file_path)
                )
                
                if file_exists:
                    logger.info(f"Archivo {file_path} parece haberse subido a pesar del error: {e}")
                    return {"status": "success", "path": file_path, "recovered": True}
                    
                # Calcular retardo exponencial con jitter
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1)) * (0.5 + 0.5 * (hash(file_path) % 100) / 100)
                    logger.warning(f"Intento {attempt} fallido para {file_path}: {e}. Reintentando en {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Máximo de intentos alcanzado para {file_path}: {e}")
        
        # Si llegamos aquí, todos los intentos fallaron
        raise last_error

    def should_retry_download(exception):
        """Condición para tenacity: reintentar si es RequestException, 
           pero NO si es HTTPError con status 4xx."""
        if isinstance(exception, HTTPError):
            # No reintentar en errores del cliente (4xx)
            if 400 <= exception.response.status_code < 500:
                logger.warning(f"Error HTTP {exception.response.status_code} detectado. No se reintentará la descarga para {exception.request.url}")
                return False
        # Reintentar para otros RequestException (conexión, timeout, errores 5xx)
        return isinstance(exception, RequestException)

    @retry(
        stop=stop_after_attempt(5),  # Reintentar hasta 5 veces
        wait=wait_exponential(multiplier=1, min=2, max=30),  # Espera exponencial entre reintentos
        retry=retry_if_exception(should_retry_download) # Usar la función de condición personalizada
    )
    def _download_xml(self, url):
        """
        Descarga un archivo XML desde una URL.
        
        Args:
            url (str): URL del archivo XML a descargar
            
        Returns:
            bytes: Contenido del archivo XML
        """
        # Añadir User-Agent puede ayudar con algunos servidores
        headers = {'User-Agent': 'Mozilla/5.0'} 
        response = requests.get(url, timeout=15, headers=headers) 
        response.raise_for_status()  # Lanza HTTPError para 4xx/5xx
        # Devolver bytes directamente
        return response.content

    def _generate_filename(self, url, uuid=None):
        """
        Genera un nombre de archivo único basado en la URL o UUID.
        
        Args:
            url (str): URL del archivo XML
            uuid (str, optional): UUID del documento si está disponible
            
        Returns:
            str: Nombre de archivo único
        """
        if uuid:
            return f"{uuid}.xml"
        else:
            # Usar un hash de la URL como nombre de archivo
            return f"{hashlib.md5(url.encode()).hexdigest()}.xml"
    
    async def download_and_store_xml(self, row_idx, df, executor):
        """
        Descarga y almacena un archivo XML de una fila del DataFrame.
        
        Args:
            row_idx (int): Índice de la fila en el DataFrame
            df (pd.DataFrame): DataFrame con las URLs y metadatos de los XMLs
            executor (ThreadPoolExecutor): Ejecutor para operaciones bloqueantes
            
        Returns:
            dict: Resultado de la operación con información relevante y datos procesados.
        """
        row = df.iloc[row_idx]
        url = row.get("url_xml")
        uuid = row.get("xml_uuid") # Obtener UUID para identificación
        filename = None # Inicializar filename
        processed_data_df = pd.DataFrame() # DataFrame vacío por defecto
        upload_result = None # Resultado de la subida
        success = False # Estado general de la operación
        error_message = None # Mensaje de error

        # Usar UUID o URL para identificar el XML en logs y errores
        xml_identifier = f"UUID {uuid}" if uuid else f"URL hash {hashlib.md5(url.encode()).hexdigest() if isinstance(url, str) else 'invalid_url'}"

        # Verificar si la URL es válida
        if pd.isna(url) or not isinstance(url, str):
            logger.warning(f"URL no válida en fila {row_idx} ({xml_identifier})")
            error_message = "URL no válida"
            # Devolver diccionario con estado de fallo
            return {
                "success": success,
                "row_idx": row_idx,
                "error": error_message,
                "filename": filename,
                "upload_result": upload_result,
                "processed_data": processed_data_df # DataFrame vacío
            }

        try:
            # Generar nombre de archivo usando UUID si está disponible
            filename = self._generate_filename(url, uuid)
            file_path = f"{self.folder_path}/{filename}"

            # 1. Descargar el XML
            logger.debug(f"Descargando {xml_identifier} desde {url}")
            xml_content = await asyncio.get_event_loop().run_in_executor(
                executor, self._download_xml, url
            )
            logger.debug(f"Descargado {xml_identifier}, tamaño: {len(xml_content)} bytes")

            # 2. Procesar el XML descargado (usando el método estático)
            logger.debug(f"Procesando contenido de {xml_identifier}")
            processed_data_df = XMLProcessor._parse_xml_content(xml_content, row, xml_identifier)
            if processed_data_df.empty:
                 logger.warning(f"El procesamiento de {xml_identifier} no generó datos (o hubo un error interno en el parseo). Ver logs de XMLProcessor.")
            else:
                 logger.debug(f"Procesados {len(processed_data_df)} conceptos de {xml_identifier}")

            # 3. Subir el XML a Supabase Storage (con reintentos)
            logger.debug(f"Subiendo {filename} a {self.bucket_name}/{file_path}")
            upload_result = await self._upload_with_retry(self.bucket_name, file_path, xml_content, executor)
            logger.info(f"Archivo {filename} subido exitosamente a {self.bucket_name}/{file_path}")
            success = True # Marcar como éxito si la subida fue correcta

        except HTTPError as e:
             # Capturar errores HTTP específicamente después de que los reintentos fallaron o no aplicaron
            if 400 <= e.response.status_code < 500:
                error_message = f"Error HTTP Cliente {e.response.status_code} al descargar {xml_identifier}"
                # Log como warning ya que no se reintentó y es esperado para links rotos
                logger.warning(error_message + f" desde {url}") 
            else: # Errores 5xx u otros HTTPError que superaron los reintentos
                 error_message = f"Error HTTP Servidor {e.response.status_code} persistente al descargar {xml_identifier}"
                 logger.error(error_message + f" desde {url}: {e}", exc_info=False) # No es necesario exc_info si ya se loggeó en retry
            # No continuar con el procesamiento/subida si la descarga falló permanentemente
            success = False

        except RequestException as e:
             # Capturar otros errores de red que superaron los reintentos
            error_message = f"Error de red persistente al descargar {xml_identifier}"
            logger.error(error_message + f" desde {url}: {e}", exc_info=False) # No es necesario exc_info si ya se loggeó en retry
            success = False
        
        except Exception as e:
             # Capturar cualquier otro error inesperado durante el proceso
            error_message = f"Error inesperado procesando {xml_identifier}"
            logger.error(error_message + f": {e}", exc_info=True)
            success = False
        
        # Devolver resultado completo
        return {
            "success": success,
            "row_idx": row_idx,
            "url": url, # Incluir URL para referencia
            "xml_uuid": uuid, # Incluir UUID para referencia
            "error": error_message,
            "filename": filename,
            "upload_result": upload_result,
            "processed_data": processed_data_df # Devuelve el DF procesado (puede estar vacío)
        }

class XMLDownloader:
    """Gestiona la descarga concurrente de archivos XML desde un DataFrame
    y los almacena en Supabase Storage, integrando el procesamiento.
    """
    
    def __init__(self, supabase_url=None, supabase_key=None):
        """
        Inicializa el gestor de descargas de XML.
        
        Args:
            supabase_url (str, optional): URL de la API de Supabase
            supabase_key (str, optional): Clave de API de Supabase
        """
        # Priorizar variables de entorno si los argumentos son None
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL y Key deben ser proporcionados o establecidos como variables de entorno.")

        self.xml_storage = None
        
        # Inicializar contadores y resultados
        self.total_files = 0
        self.completed_files = 0
        self.success_count = 0
        self.error_count = 0
        self.results = [] # Almacenará los dicts de resultado de cada tarea
        self.all_processed_dataframes = [] # Lista para acumular DataFrames procesados
    
    def _initialize_storage_manager(self):
        """
        Inicializa el gestor de almacenamiento si no está ya inicializado.
        """
        if not self.xml_storage and self.supabase_url and self.supabase_key:
            self.xml_storage = XMLStorageManager(
                self.supabase_url,
                self.supabase_key,
                bucket_name="xml_files"
            )
    
    async def download_all_xmls(self, df, max_workers=8):
        """
        Descarga y almacena todos los XMLs del DataFrame de forma concurrente.
        
        Args:
            df (pd.DataFrame): DataFrame con las URLs y metadatos de los XMLs
            max_workers (int, optional): Número máximo de trabajadores concurrentes
            
        Returns:
            dict: Resultados de la operación con estadísticas y DataFrame procesado.
        """
        # Validar entrada
        if df.empty:
            logger.warning("El DataFrame de entrada está vacío.")
            return {"total": 0, "success": 0, "errors": 0, "processed_dataframe": pd.DataFrame()}
            
        if "url_xml" not in df.columns:
            logger.error("La columna 'url_xml' es necesaria en el DataFrame.")
            # Podríamos lanzar un error o devolver un resultado indicando el fallo
            raise ValueError("La columna 'url_xml' falta en el DataFrame de entrada.")

        # Inicializar gestor de almacenamiento (si no existe)
        self._initialize_storage_manager()

        # Reiniciar contadores y resultados para esta ejecución
        self.total_files = len(df)
        self.completed_files = 0
        self.success_count = 0
        self.error_count = 0
        self.results = []
        self.all_processed_dataframes = [] # Limpiar lista de dataframes

        logger.info(f"Iniciando descarga y procesamiento de {self.total_files} archivos XML...")
        start_time = datetime.now()
        
        # Crear barra de progreso
        pbar = tqdm(total=self.total_files, desc="Descargando XMLs", unit="archivo")

        # Crear ThreadPoolExecutor para operaciones bloqueantes (descarga, subida)
        # El procesamiento XML (CPU-bound ligero) también puede ir aquí o podría 
        # evaluarse ProcessPoolExecutor si el parseo fuera muy pesado.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Crear tareas asyncio para cada fila del DataFrame
            tasks = [
                asyncio.create_task(
                    self.xml_storage.download_and_store_xml(idx, df, executor)
                )
                for idx in range(self.total_files)
            ]

            # Procesar tareas a medida que se completan
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    self.results.append(result) # Guardar el resultado detallado
                    self.completed_files += 1
                    
                    # Actualizar barra de progreso
                    pbar.update(1)
                    pbar.set_postfix(éxitos=self.success_count, errores=self.error_count)
                    
                    # Actualizar contadores basados en el 'success' del resultado
                    if result.get("success", False):
                        self.success_count += 1
                        # Acumular DataFrame procesado si no está vacío
                        processed_df = result.get("processed_data")
                        if processed_df is not None and not processed_df.empty:
                            self.all_processed_dataframes.append(processed_df)
                    else:
                        # Verificar si quizá se solucionó en un reintento posterior
                        upload_result = result.get("upload_result", {})
                        if isinstance(upload_result, dict) and upload_result.get("recovered", False):
                            # Si se detectó que el archivo existe a pesar del error, contar como éxito
                            self.success_count += 1
                            self.error_count -= 1  # Ajustar contador de errores
                            logger.info(f"Archivo en fila {result.get('row_idx', 'N/A')} recuperado y marcado como éxito")
                            # Agregar el DataFrame procesado si está disponible
                            processed_df = result.get("processed_data")
                            if processed_df is not None and not processed_df.empty:
                                self.all_processed_dataframes.append(processed_df)
                        else:
                            self.error_count += 1
                            logger.warning(f"Fallo en Tarea: Fila {result.get('row_idx', 'N/A')}, Error: {result.get('error', 'Desconocido')}")
                        
                    # Log de progreso
                    if self.completed_files % 50 == 0 or self.completed_files == self.total_files:
                        elapsed_time = datetime.now() - start_time
                        logger.info(f"Progreso: {self.completed_files}/{self.total_files} completados ({self.success_count} éxito, {self.error_count} errores). Tiempo: {elapsed_time}")
                        
                except Exception as task_exception:
                    # Capturar excepciones no manejadas dentro de la tarea (aunque deberían ser manejadas)
                    self.completed_files += 1
                    self.error_count += 1
                    logger.error(f"Excepción inesperada en tarea asyncio: {task_exception}")
                    # Actualizar barra de progreso en caso de error
                    pbar.update(1)
                    pbar.set_postfix(éxitos=self.success_count, errores=self.error_count)
                    # Considerar guardar esta excepción también si es necesario

        # Cerrar la barra de progreso
        pbar.close()
        
        end_time = datetime.now()
        total_time = end_time - start_time
        logger.info(f"Descarga y procesamiento completado en {total_time}.")
        logger.info(f"Resultados: {self.success_count} éxitos, {self.error_count} errores.")

        # Combinar todos los DataFrames procesados
        final_df = pd.DataFrame() # DataFrame vacío por defecto
        if self.all_processed_dataframes:
            try:
                final_df = pd.concat(self.all_processed_dataframes, ignore_index=True)
                logger.info(f"Total de {len(final_df)} conceptos extraídos y combinados.")
                
                # Si hubo errores pero todos los archivos se subieron, actualizar contador
                if self.success_count == self.total_files and self.error_count > 0:
                    logger.info(f"A pesar de {self.error_count} errores reportados, todos los archivos ({self.total_files}) fueron procesados correctamente")
                    self.error_count = 0
            except Exception as concat_error:
                 logger.error(f"Error al concatenar los DataFrames procesados: {concat_error}")
                 # Devolver DF vacío y quizás más detalles del error
                 final_df = pd.DataFrame() 
        else:
            logger.warning("No se obtuvieron datos procesados de ningún archivo.")

        # Devolver resumen y el DataFrame final
        return {
            "total_files": self.total_files,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_time": str(total_time),
            # "detailed_results": self.results, # Opcional: incluir detalles de cada archivo
            "processed_dataframe": final_df
        }

    # Función wrapper síncrona para conveniencia
    def download_all_xmls_sync(self, df, max_workers=8):
        """
        Versión sincrónica de download_all_xmls.
        
        Args:
            df (pd.DataFrame): DataFrame con las URLs y metadatos de los XMLs
            max_workers (int, optional): Número máximo de trabajadores concurrentes
            
        Returns:
            dict: Resultados de la operación con estadísticas y DataFrame procesado.
        """
        try:
            # Ejecutar la versión asíncrona usando asyncio.run()
            return asyncio.run(self.download_all_xmls(df, max_workers=max_workers))
        except RuntimeError as e:
            # Manejar error si ya hay un loop de asyncio corriendo (ej. en Jupyter)
            if "cannot run nested" in str(e):
                logger.error("No se puede ejecutar asyncio.run en un loop existente (ej. Jupyter). Usa await download_all_xmls directamente.")
                # Devolver un resultado indicando el error
                return {"total_files": 0, "success_count": 0, "error_count": 0, "total_time": "0", "processed_dataframe": pd.DataFrame(), "error_message": str(e)}
            else:
                raise # Re-lanzar otros RuntimeErrors

# Ejemplo de uso básico (requiere un DataFrame 'df_entrada' con columna 'XML')
# async def main():
#     # Cargar DataFrame (ejemplo)
#     data = {
#         "XML": ["https://example.com/file1.xml", "https://example.com/file2.xml"],
#         "XML_UUID": ["uuid1", "uuid2"]
#     }
#     df_entrada = pd.DataFrame(data)
    
#     # Crear instancia de XMLDownloader
#     downloader = XMLDownloader(supabase_url="https://example.supabase.co", supabase_key="your_supabase_key")
    
#     # Descargar y procesar XMLs
#     result = await downloader.download_all_xmls(df_entrada, max_workers=8)
    
#     # Imprimir resultado
#     print(result)
    
#     # Ejecutar el ejemplo
#     asyncio.run(main())
