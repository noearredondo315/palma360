import os
import logging
import asyncio
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, wait_fixed
from concurrent.futures import ThreadPoolExecutor
from xml_processor import XMLProcessor
from tqdm.auto import tqdm
from requests.exceptions import RequestException, HTTPError
from local_storage_manager import LocalStorageManager
import requests

# Configurar logging
logger = logging.getLogger('xml_downloader')

class LocalXMLStorageManager:
    """
    Clase para gestionar el almacenamiento de archivos XML en el sistema de archivos local.
    
    Esta clase guarda archivos XML en una estructura de carpetas local organizada por fecha.
    """
    
    def __init__(self, base_path="data", xml_storage_dir="downloaded_xmls"):
        """
        Inicializa el gestor de almacenamiento local de archivos XML.
        
        Args:
            base_path (str): Ruta base para las descargas (ej: 'data')
            xml_storage_dir (str): Directorio para los archivos XML (relativo a base_path)
        """
        self.base_path = os.path.abspath(base_path)
        self.xml_storage_dir = xml_storage_dir
        self.current_date_str = datetime.now().strftime('%Y%m%d')
        self.daily_folder_path = os.path.join(self.base_path, self.xml_storage_dir, f"Corte_al_{self.current_date_str}")

        # Asegurarse de que el directorio de descarga del día existe
        self._ensure_daily_folder_exists()

    def _ensure_daily_folder_exists(self):
        """
        Asegura que el directorio para los XMLs del día actual existe.
        """
        try:
            os.makedirs(self.daily_folder_path, exist_ok=True)
            logger.debug(f"Directorio de descarga XML del día asegurado: {self.daily_folder_path}")
        except Exception as e:
            logger.error(f"Error al crear directorio de descarga XML '{self.daily_folder_path}': {str(e)}")
            raise

    def _file_exists_locally(self, file_path):
        """Verifica si un archivo existe localmente.
        
        Args:
            file_path (str): Ruta completa del archivo local
            
        Returns:
            bool: True si el archivo existe, False en caso contrario
        """
        return os.path.exists(file_path)
            
    async def save_xml_content(self, file_name: str, content: bytes, executor: ThreadPoolExecutor):
        """Guarda el contenido XML en un archivo local dentro de la carpeta del día.

        Args:
            file_name (str): Nombre del archivo XML (ej: 'factura_XYZ.xml')
            content (bytes): Contenido binario del archivo XML.
            executor (ThreadPoolExecutor): Ejecutor para operaciones de E/S de archivo.

        Returns:
            dict: Resultado de la operación con información relevante y datos procesados.
        """
        file_path = os.path.join(self.daily_folder_path, file_name)
        
        try:
            # Usar run_in_executor para la operación de escritura de archivo bloqueante
            await asyncio.get_event_loop().run_in_executor(
                executor,
                self._write_file_sync,
                file_path,
                content
            )
            logger.debug(f"Archivo XML guardado localmente: {file_path}")
            return {"status": "success", "path": file_path, "local_path": file_path, "type": "xml"}
        except Exception as e:
            logger.error(f"Error al guardar el archivo XML {file_path} localmente: {str(e)}")
            # Devolver un diccionario con el error para que pueda ser manejado
            return {"status": "error", "file_name": file_name, "error": str(e), "type": "xml"}

    def _write_file_sync(self, file_path: str, content: bytes):
        """Función síncrona para escribir contenido en un archivo."""
        with open(file_path, 'wb') as f:
            f.write(content)
    
    def get_daily_folder_path(self):
        """Devuelve la ruta de la carpeta donde se guardan los XMLs del día."""
        return self.daily_folder_path

class XMLDownloader:
    """Gestiona la descarga concurrente de archivos XML desde un DataFrame
    y los almacena localmente, integrando el procesamiento.
    """
    
    def __init__(self, base_data_path=None, xml_storage_folder="downloaded_xmls", session=None):
        """
        Inicializa el gestor de descargas de XML.
        
        Args:
            base_data_path (str, optional): Ruta base para los datos (ej: 'data')
            xml_storage_folder (str, optional): Directorio para los archivos XML (relativo a base_data_path)
            session (requests.Session, optional): Sesión de requests autenticada.
        """
        self.xml_storage = LocalXMLStorageManager(base_path=base_data_path, xml_storage_dir=xml_storage_folder) # Usar el nuevo gestor local
        
        self.total_files = 0
        self.completed_files = 0
        self.success_count = 0
        self.error_count = 0
        self.results = []
        self.downloaded_xml_paths = [] # Para almacenar las rutas de los XML descargados
        self.session = session  # Guardar la sesión proporcionada
        
    def _initialize_storage_manager(self):
        # Este método ya no es estrictamente necesario ya que se inicializa en __init__,
        # pero lo mantenemos por si se añaden lógicas de reinicialización en el futuro.
        if self.xml_storage is None:
            # Esta línea no debería ejecutarse si __init__ funciona correctamente.
            self.xml_storage = LocalXMLStorageManager() 
        logger.debug("LocalXMLStorageManager inicializado.")

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

        # Reiniciar contadores y resultados para esta ejecución
        self.total_files = len(df)
        self.completed_files = 0
        self.success_count = 0
        self.error_count = 0
        self.results = []
        self.downloaded_xml_paths = [] # Reiniciar también esta lista

        logger.info(f"Iniciando descarga de {self.total_files} archivos XML...")
        start_time = datetime.now()
        
        # Crear barra de progreso
        pbar = tqdm(total=self.total_files, desc="Descargando XMLs", unit="archivo")

        # Crear ThreadPoolExecutor para operaciones bloqueantes (descarga, subida)
        # El procesamiento XML (CPU-bound ligero) también puede ir aquí o podría 
        # evaluarse ProcessPoolExecutor si el parseo fuera muy pesado.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Usar un enfoque de ventana deslizante para crear y gestionar tareas
            # en lugar de crear todas las tareas a la vez
            batch_size = min(max_workers * 3, 50)  # Limitar el número de tareas pendientes
            pending = set()
            row_idx = 0
            
            # Iniciar el primer lote de tareas
            for i in range(min(batch_size, self.total_files)):
                task = asyncio.create_task(
                    self._download_and_store_xml(row_idx, df, executor, pbar)
                )
                pending.add(task)
                row_idx += 1
            
            # Procesar tareas y añadir nuevas a medida que se completan anteriores
            while pending:
                # Esperar a que se complete una tarea
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                # Procesar las tareas completadas
                for task in done:
                    try:
                        result = await task
                        self.results.append(result)  # Guardar el resultado detallado
                        self.completed_files += 1
                        
                        # Actualizar barra de progreso
                        pbar.update(1)
                        pbar.set_postfix(éxitos=self.success_count, errores=self.error_count)
                        
                        # Actualizar contadores basados en el 'success' del resultado
                        if result.get("success", False):
                            self.success_count += 1
                            # Si el guardado fue exitoso, almacenar la información del archivo
                            if result.get("local_path") and result.get("xml_uuid"):
                                self.downloaded_xml_paths.append({
                                    "local_path": result.get("local_path"),
                                    "xml_uuid": result.get("xml_uuid"), # o row_idx si se prefiere
                                    "row_idx": result.get("row_idx") # Incluir row_idx para facilitar el join con metadata
                                })
                        else:
                            # Verificar si quizá se solucionó en un reintento posterior
                            upload_result = result.get("upload_result", {})
                            if isinstance(upload_result, dict) and upload_result.get("recovered", False):
                                # Si se detectó que el archivo existe a pesar del error, contar como éxito
                                self.success_count += 1
                                self.error_count -= 1  # Ajustar contador de errores
                                logger.info(f"Archivo en fila {result.get('row_idx', 'N/A')} recuperado y marcado como éxito")
                                # Almacenar información del archivo recuperado
                                if result.get("local_path") and result.get("xml_identifier"):
                                    self.downloaded_xml_paths.append({
                                        "local_path": result.get("local_path"),
                                        "xml_uuid": result.get("xml_uuid"),
                                        "row_idx": result.get("row_idx")
                                    })
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
                
                    # Añadir una nueva tarea si hay más para procesar
                    if row_idx < self.total_files:
                        task = asyncio.create_task(
                            self._download_and_store_xml(row_idx, df, executor, pbar)
                        )
                        pending.add(task)
                        row_idx += 1

        # Cerrar la barra de progreso
        pbar.close()
        
        end_time = datetime.now() # Definir end_time aquí
        total_time = end_time - start_time
        logger.info(f"Descarga completada en {total_time}.")
        logger.info(f"Resultados: {self.success_count} éxitos, {self.error_count} errores.")

        # Ya no se procesan DataFrames aquí
        if not self.downloaded_xml_paths and self.total_files > 0:
            logger.warning("No se descargó ningún archivo XML.")
        elif self.downloaded_xml_paths:
            logger.info(f"Se descargaron {len(self.downloaded_xml_paths)} archivos XML.")

        # Devolver resumen y la lista de archivos descargados
        return {
            "total_files": self.total_files,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_time": str(total_time),
            "download_folder_path": self.xml_storage.get_daily_folder_path(),
            "downloaded_xml_files_info": self.downloaded_xml_paths # Cambiado el nombre de la clave y su contenido
        }

    async def _download_and_store_xml(self, row_idx, df, executor, pbar):
        """
        Descarga y almacena un archivo XML de una fila del DataFrame.
        No procesa el XML aquí.
        
        Args:
            row_idx (int): Índice de la fila en el DataFrame
            df (pd.DataFrame): DataFrame con las URLs y metadatos de los XMLs
            executor (ThreadPoolExecutor): Ejecutor para operaciones bloqueantes
            pbar (tqdm): Barra de progreso
            
        Returns:
            dict: Resultado de la operación con información del archivo guardado o error.
        """
        row = df.iloc[row_idx]
        url = row.get("url_xml")
        uuid = row.get("xml_uuid") # Obtener UUID para identificación
        filename = None # Inicializar filename
        storage_result = None # Resultado del almacenamiento
        success = False # Estado general de la operación
        error_message = None # Mensaje de error
        local_file_path = None # Ruta del archivo guardado localmente

        # Usar UUID o URL para identificar el XML en logs y errores
        xml_identifier = f"UUID {uuid}" if uuid else f"URL hash {hashlib.md5(url.encode()).hexdigest() if isinstance(url, str) else 'invalid_url'}"

        # Verificar si la URL es válida
        if pd.isna(url) or not isinstance(url, str):
            logger.warning(f"URL no válida en fila {row_idx} ({xml_identifier})")
            error_message = "URL no válida"
            return {
                "success": success,
                "row_idx": row_idx,
                "xml_uuid": uuid,
                "xml_identifier": xml_identifier,
                "error": error_message
            }

        try:
            # Generar nombre de archivo usando UUID si está disponible
            filename = self._generate_file_name(row)
            # file_path = os.path.join(self.xml_storage.get_daily_folder_path(), filename) # REMOVED - path determined by storage manager

            # 1. Descargar el XML
            logger.debug(f"Descargando {xml_identifier} desde {url}")
            xml_content = await asyncio.get_event_loop().run_in_executor(
                executor, self._download_xml, url
            )
            logger.debug(f"Contenido descargado para {xml_identifier}, tamaño: {len(xml_content)} bytes")

            # 2. Procesar el XML descargado (usando el método estático) # REMOVED THIS STEP
            # logger.debug(f"Procesando contenido de {xml_identifier}")
            # processed_data_df = XMLProcessor._parse_xml_content(xml_content, row, xml_identifier)
            # if processed_data_df.empty:
            #      logger.warning(f"El procesamiento de {xml_identifier} no generó datos (o hubo un error interno en el parseo). Ver logs de XMLProcessor.")
            # else:
            #      logger.debug(f"Procesados {len(processed_data_df)} conceptos de {xml_identifier}")

            # 3. Guardar el XML localmente
            logger.debug(f"Guardando {filename} en {self.xml_storage.get_daily_folder_path()}")
            storage_result = await self.xml_storage.save_xml_content(filename, xml_content, executor)
            
            if storage_result and storage_result.get("status") == "success":
                logger.debug(f"Archivo {filename} guardado exitosamente en {storage_result.get('path')}")
                success = True
                local_file_path = storage_result.get('local_path') # o 'path'
                # self.downloaded_xml_paths.append(local_file_path) # MOVED to download_all_xmls
            else:
                error_message = storage_result.get("error", "Error desconocido al guardar XML")
                logger.error(f"Fallo al guardar {filename}: {error_message}")
                success = False

        except HTTPError as e:
            if 400 <= e.response.status_code < 500:
                error_message = f"Error HTTP Cliente {e.response.status_code} al descargar {xml_identifier}"
                logger.warning(error_message + f" desde {url}") 
            else:
                 error_message = f"Error HTTP Servidor {e.response.status_code} persistente al descargar {xml_identifier}"
                 logger.error(error_message + f" desde {url}: {e}", exc_info=False)
            success = False

        except RequestException as e:
            error_message = f"Error de red persistente al descargar {xml_identifier}"
            logger.error(error_message + f" desde {url}: {e}", exc_info=False)
            success = False
        
        except Exception as e:
            error_message = f"Error inesperado procesando descarga de {xml_identifier}"
            logger.error(error_message + f": {e}", exc_info=True)
            success = False
        
        # Devolver resultado completo
        return {
            "success": success,
            "row_idx": row_idx,
            "url": url,
            "xml_uuid": uuid,
            "xml_identifier": xml_identifier,
            "error": error_message,
            "filename": filename,
            "local_path": local_file_path, # Ruta del archivo si se guardó correctamente
            "upload_result": storage_result # Mantenemos esto por si 'recovered' es usado por LocalStorageManager
        }

    def _generate_file_name(self, row):
        """
        Genera un nombre de archivo único basado en la URL o UUID.
        
        Args:
            row (pd.Series): Fila del DataFrame con información del XML
            
        Returns:
            str: Nombre de archivo único
        """
        uuid = row.get("xml_uuid", None)
        if uuid:
            return f"{uuid}.xml"
        else:
            # Usar un hash de la URL como nombre de archivo
            url = row.get("XML", "")
            return f"{hashlib.md5(url.encode()).hexdigest()}.xml"

    @retry(
        stop=stop_after_attempt(2),  # Reducido a 2 reintentos 
        wait=wait_fixed(1),  # Espera fija de 1 segundo
        retry=retry_if_exception_type((requests.exceptions.RequestException,))  # Reintentar solo en errores de red
    )
    def _download_xml(self, url):
        """Descarga un archivo XML desde una URL."""
        try:
            # Usar la sesión proporcionada en __init__ o crear una nueva si no se pasó.
            current_session = self.session
            if current_session is None:
                logger.debug("Creando nueva sesión de requests para _download_xml (sesión no proporcionada en init)")
                current_session = requests.Session()
            
            # Reducir el timeout para fallar más rápido
            response = current_session.get(url, timeout=15) # Reducido a 15 segundos
            response.raise_for_status()
            # Cambiado de INFO a DEBUG para reducir los logs en la terminal
            logger.debug(f"Descarga exitosa de {url} (status: {response.status_code})")
            return response.content
        except Exception as e:
            # Comprobar si es un error de memoria para evitar reintentos en estos casos
            if 'Cannot allocate memory' in str(e):
                logger.error(f"Error de memoria al descargar {url}. No se reintentará.")
                # Forzar excepciones de memoria como no recuperables (no se reintentarán)
                raise MemoryError(f"Cannot allocate memory: {e}")
            else:
                logger.error(f"Error al descargar {url}: {e}", exc_info=False)
                raise

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
