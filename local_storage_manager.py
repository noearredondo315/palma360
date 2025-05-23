import os
import json
import datetime
import logging
import shutil

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('local_storage_manager')

class LocalStorageManager:
    """Clase para gestionar el almacenamiento y recuperación de archivos JSON de IDs procesados localmente.
    
    Esta clase proporciona métodos para:
    - Cargar el archivo JSON de los IDs de facturas procesadas
    - Crear copias de seguridad de dicho archivo JSON
    - Actualizar el archivo JSON con nuevos IDs de facturas
    - Registrar XMLs con errores para su posterior análisis
    """
    
    def __init__(self, base_data_path="data", processed_ids_dir="processed_ids", backup_dir="processed_ids_backup"):
        """Inicializa el gestor de almacenamiento local.
        
        Args:
            base_data_path (str): Ruta base para los datos (ej: 'data')
            processed_ids_dir (str): Directorio para el archivo de IDs procesados (relativo a base_data_path)
            backup_dir (str): Directorio para los backups (relativo a base_data_path)
        """
        self.base_data_path = os.path.abspath(base_data_path)
        self.processed_ids_path = os.path.join(self.base_data_path, processed_ids_dir)
        self.backup_path = os.path.join(self.base_data_path, backup_dir)
        
        self.main_file_name = "processed_ids.json"
        self.failed_xml_file_name = "failed_xmls.json"
        self.processed_ids_file_path = os.path.join(self.processed_ids_path, self.main_file_name)
        self.failed_xml_file_path = os.path.join(self.processed_ids_path, self.failed_xml_file_name)
        
        # Asegurarse de que los directorios existen
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self):
        """Asegura que los directorios necesarios existen localmente."""
        try:
            os.makedirs(self.processed_ids_path, exist_ok=True)
            logger.debug(f"Directorio de IDs procesados asegurado: {self.processed_ids_path}")
            
            os.makedirs(self.backup_path, exist_ok=True)
            logger.debug(f"Directorio de backups asegurado: {self.backup_path}")
                
        except Exception as e:
            logger.error(f"Error al crear directorios: {str(e)}")
            raise
    
    def backup_ids_file(self):
        """Crea una copia de seguridad del archivo JSON actual de IDs procesados.
            
        Returns:
            str: Nombre del archivo de respaldo creado, o None si no hay archivo original.
        """
        if not os.path.exists(self.processed_ids_file_path):
            logger.debug(f"No existe el archivo {self.processed_ids_file_path} para respaldar.")
            return None
            
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file_name = f"backup_{timestamp}_{self.main_file_name}"
            backup_file_path = os.path.join(self.backup_path, backup_file_name)
            
            shutil.copy2(self.processed_ids_file_path, backup_file_path)
            
            logger.debug(f"Archivo respaldado como: {backup_file_path}")
            return backup_file_name
            
        except Exception as e:
            logger.error(f"Error al crear respaldo: {str(e)}")
            raise
    
    def get_processed_ids(self):
        """Obtiene el conjunto de IDs de facturas ya procesadas desde el archivo local.
        
        Returns:
            tuple: (set de IDs de facturas procesadas, nombre_del_archivo_usado)
                   Retorna (set(), self.main_file_name) si el archivo no existe o está vacío.
        """
        try:
            if os.path.exists(self.processed_ids_file_path):
                with open(self.processed_ids_file_path, 'r', encoding='utf-8') as f:
                    ids_procesados = set(json.load(f))
                logger.debug(f"Se cargaron {len(ids_procesados)} IDs procesados desde {self.processed_ids_file_path}")
                
                # Crear respaldo del archivo original antes de cualquier modificación potencial (aunque aquí solo leemos)
                self.backup_ids_file()
                
                return ids_procesados, self.main_file_name
            else:
                logger.debug(f"Archivo {self.processed_ids_file_path} no encontrado. Creando nuevo conjunto de IDs procesados.")
                return set(), self.main_file_name # Devuelve el nombre esperado para consistencia
                
        except json.JSONDecodeError as e:
            logger.error(f"Error al decodificar JSON de {self.processed_ids_file_path}: {str(e)}. Tratando como archivo vacío.")
            return set(), self.main_file_name
        except Exception as e:
            logger.error(f"Error al obtener IDs procesados: {str(e)}")
            return set(), self.main_file_name
    
    def update_processed_ids(self, ids_procesados, file_name=None):
        """Actualiza el archivo JSON local con los IDs de facturas procesadas.
        
        Args:
            ids_procesados (set): Conjunto de IDs de facturas procesadas
            file_name (str, optional): Nombre del archivo a actualizar (ignorado, siempre usa self.main_file_name)
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        # file_name es ignorado para mantener la firma, pero siempre usamos self.processed_ids_file_path
        target_file_path = self.processed_ids_file_path
            
        try:
            # Convertir el conjunto a lista y luego a JSON
            ids_json = json.dumps(list(ids_procesados), indent=4) # indent para legibilidad
            
            # Escribir el archivo actualizado
            with open(target_file_path, 'w', encoding='utf-8') as f:
                f.write(ids_json)
            
            logger.debug(f"Archivo de IDs actualizado: {target_file_path} con {len(ids_procesados)} IDs")
            return True
            
        except Exception as e:
            logger.error(f"Error al actualizar archivo de IDs {target_file_path}: {str(e)}")
            return False

    def get_failed_xmls(self):
        """Obtiene el registro de XMLs con errores desde el archivo local.
        
        Returns:
            dict: Diccionario con los XMLs que tuvieron errores, agrupados por tipo de error.
                  Formato: {
                      "sin_rfc": ["UUID1", "UUID2", ...],
                      "sin_conceptos": [...],
                      "error_descarga": [...],
                      ...
                  }
        """
        try:
            if os.path.exists(self.failed_xml_file_path):
                with open(self.failed_xml_file_path, 'r', encoding='utf-8') as f:
                    failed_xmls = json.load(f)
                
                total_errors = sum(len(uuids) for uuids in failed_xmls.values())
                logger.debug(f"Se cargaron {total_errors} XMLs con errores distribuidos en {len(failed_xmls)} categorías")
                return failed_xmls
            else:
                logger.debug(f"Archivo de XMLs con errores {self.failed_xml_file_path} no encontrado. Inicializando registro vacío.")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Error al decodificar JSON de {self.failed_xml_file_path}: {str(e)}. Tratando como archivo vacío.")
            return {}
        except Exception as e:
            logger.error(f"Error al obtener XMLs con errores: {str(e)}")
            return {}
    
    def update_failed_xmls(self, failed_xmls):
        """Actualiza el archivo JSON local con los XMLs que tuvieron errores.
        
        Args:
            failed_xmls (dict): Diccionario con los XMLs que tuvieron errores, agrupados por tipo de error
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        try:
            # Convertir a JSON con formato legible
            failed_json = json.dumps(failed_xmls, indent=4)
            
            # Escribir el archivo actualizado
            with open(self.failed_xml_file_path, 'w', encoding='utf-8') as f:
                f.write(failed_json)
            
            total_errors = sum(len(uuids) for uuids in failed_xmls.values())
            logger.debug(f"Archivo de XMLs con errores actualizado: {self.failed_xml_file_path} con {total_errors} XMLs distribuidos en {len(failed_xmls)} categorías")
            return True
            
        except Exception as e:
            logger.error(f"Error al actualizar archivo de XMLs con errores {self.failed_xml_file_path}: {str(e)}")
            return False
            
    def add_failed_xml(self, uuid, error_type):
        """Añade un XML con error al registro.
        
        Args:
            uuid (str): UUID del XML que tuvo error
            error_type (str): Tipo de error (ej: 'sin_rfc', 'sin_conceptos', 'error_descarga')
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        try:
            # Obtener registro actual
            failed_xmls = self.get_failed_xmls()
            
            # Inicializar la categoría si no existe
            if error_type not in failed_xmls:
                failed_xmls[error_type] = []
                
            # Añadir UUID si no está ya en la lista
            if uuid not in failed_xmls[error_type]:
                failed_xmls[error_type].append(uuid)
                
            # Actualizar el archivo
            return self.update_failed_xmls(failed_xmls)
            
        except Exception as e:
            logger.error(f"Error al añadir XML {uuid} con error {error_type}: {str(e)}")
            return False

# Ejemplo de uso (opcional, para pruebas)
if __name__ == '__main__':
    # Crear una instancia del gestor de almacenamiento local
    # Esto creará las carpetas 'data/processed_ids' y 'data/processed_ids_backup' 
    # en el directorio donde se ejecute el script, si no existen.
    storage_manager = LocalStorageManager(base_data_path="../data_test") # Usar ruta relativa para prueba

    # Obtener IDs procesados (estará vacío la primera vez o si el archivo no existe)
    current_ids, file_used = storage_manager.get_processed_ids()
    print(f"IDs actuales: {current_ids} (del archivo: {file_used})")

    # Simular nuevos IDs procesados
    new_ids_to_add = {"id123", "id456"}
    updated_ids = current_ids.union(new_ids_to_add)

    # Actualizar el archivo de IDs procesados
    if storage_manager.update_processed_ids(updated_ids):
        print(f"Archivo de IDs actualizado con éxito.")
    else:
        print(f"Error al actualizar el archivo de IDs.")

    # Verificar que los IDs se cargan correctamente después de la actualización
    reloaded_ids, _ = storage_manager.get_processed_ids()
    print(f"IDs recargados: {reloaded_ids}")

    # Simular más IDs y actualizar de nuevo para probar el backup
    more_ids = {"id789"}
    final_ids = reloaded_ids.union(more_ids)
    storage_manager.update_processed_ids(final_ids)
    print(f"IDs finales: {final_ids}")

    # Opcional: Limpiar la carpeta de prueba después de ejecutar
    # import shutil
    # shutil.rmtree("../data_test")
    # print("Carpeta de prueba ../data_test eliminada.")
