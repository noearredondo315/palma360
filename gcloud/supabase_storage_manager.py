import os
import json
import datetime
import logging
from supabase import create_client

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('supabase_storage_manager')

class SupabaseStorageManager:
    """Clase para gestionar el almacenamiento y recuperación de archivos JSON en Supabase Storage.
    
    Esta clase proporciona métodos para:
    - Descargar el archivo JSON más reciente de los IDs de facturas procesadas
    - Crear copias de seguridad de los archivos JSON
    - Actualizar el archivo JSON con nuevos IDs de facturas
    """
    
    def __init__(self, supabase_url, supabase_key, bucket_name="1facturas", backup_bucket_name="facturas_backup"):
        """Inicializa el gestor de almacenamiento de Supabase.
        
        Args:
            supabase_url (str): URL de la API de Supabase
            supabase_key (str): Clave de API de Supabase
            bucket_name (str): Nombre del bucket principal para los archivos JSON
            backup_bucket_name (str): Nombre del bucket para las copias de seguridad
        """
        self.supabase = create_client(supabase_url, supabase_key)
        self.bucket_name = bucket_name
        self.backup_bucket_name = backup_bucket_name
        self.current_date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.main_file_pattern = f"facturas_hashlib_id_{self.current_date}.json"
        
        # Asegurarse de que los buckets existen
        self._ensure_buckets_exist()
    
    def _ensure_buckets_exist(self):
        """Asegura que los buckets necesarios existen en Supabase Storage."""
        try:
            # Verificar si el bucket principal existe
            buckets = self.supabase.storage.list_buckets()
            
            # Depurar la respuesta para ver la estructura
            logger.info(f"Respuesta de list_buckets: {buckets}")
            
            # Obtener nombres de los buckets existentes - asegurando el tipo correcto
            bucket_names = [bucket.name for bucket in buckets if hasattr(bucket, 'name')]
            
            logger.info(f"Buckets encontrados: {bucket_names}")
            logger.info(f"Bucket a verificar: '{self.bucket_name}' (tipo: {type(self.bucket_name).__name__})")
            
            # Crear bucket principal si no existe
            if str(self.bucket_name) not in bucket_names:
                logger.info(f"Creando bucket principal: {self.bucket_name}")
                self.supabase.storage.create_bucket(str(self.bucket_name), options={"public": False})
            else:
                logger.info(f"Bucket principal '{self.bucket_name}' ya existe, omitiendo creación")
            
            # Crear bucket de backup si no existe
            if str(self.backup_bucket_name) not in bucket_names:
                logger.info(f"Creando bucket de backup: {self.backup_bucket_name}")
                self.supabase.storage.create_bucket(str(self.backup_bucket_name), options={"public": False})
            else:
                logger.info(f"Bucket de backup '{self.backup_bucket_name}' ya existe, omitiendo creación")
                
        except Exception as e:
            # Manejar específicamente el error de bucket duplicado
            error_str = str(e)
            if "Duplicate" in error_str and "already exists" in error_str:
                logger.warning(f"El bucket ya existe pero no es visible por RLS. Continuando...")
                # No propagar este error específico, ya que el bucket existe aunque no podamos verlo
                return
            else:
                logger.error(f"Error al verificar/crear buckets: {error_str}")
                # Mostrar el error completo para depuración
                import traceback
                logger.error(f"Traceback completo: {traceback.format_exc()}")
                raise
    
    def _get_latest_file(self):
        """Obtiene el nombre del archivo JSON más reciente en el bucket principal.
        
        Returns:
            str: Nombre del archivo más reciente o el patrón actual si no existe ninguno
        """
        try:
            # Listar archivos en el bucket principal
            response = self.supabase.storage.from_(self.bucket_name).list()
            files = response
            
            # Filtrar archivos JSON y ordenar por fecha de creación (más reciente primero)
            json_files = [f for f in files if f["name"].endswith(".json")]
            if not json_files:
                return self.main_file_pattern
            
            # Ordenar por fecha de última modificación (más reciente primero)
            json_files.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Devolver el nombre del archivo más reciente
            return json_files[0]["name"]
            
        except Exception as e:
            logger.error(f"Error al obtener el archivo más reciente: {str(e)}")
            # En caso de error, usar el patrón actual
            return self.main_file_pattern
    
    def backup_ids_file(self, current_file_name):
        """Crea una copia de seguridad del archivo JSON actual.
        
        Args:
            current_file_name (str): Nombre del archivo actual a respaldar
            
        Returns:
            str: Nombre del archivo de respaldo creado
        """
        try:
            # Descargar el archivo actual
            content = self.supabase.storage.from_(self.bucket_name).download(current_file_name)
            
            # Crear nombre para el archivo de respaldo con timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file_name = f"backup_{timestamp}_{current_file_name}"
            
            # Subir el contenido al bucket de backup
            self.supabase.storage.from_(self.backup_bucket_name).upload(
                backup_file_name,
                content,
                {"content-type": "application/json"}
            )
            
            logger.info(f"Archivo respaldado como: {backup_file_name}")
            return backup_file_name
            
        except Exception as e:
            logger.error(f"Error al crear respaldo: {str(e)}")
            raise
    
    def get_processed_ids(self):
        """Obtiene el conjunto de IDs de facturas ya procesadas desde Supabase Storage.
        
        Returns:
            set: Conjunto de IDs de facturas procesadas
        """
        try:
            # Obtener el nombre del archivo más reciente
            latest_file = self._get_latest_file()
            
            try:
                # Intentar descargar el archivo
                content = self.supabase.storage.from_(self.bucket_name).download(latest_file)
                ids_procesados = set(json.loads(content.decode('utf-8')))
                logger.info(f"Se cargaron {len(ids_procesados)} IDs procesados desde {latest_file}")
                
                # Crear respaldo del archivo original
                self.backup_ids_file(latest_file)
                
                return ids_procesados, latest_file
                
            except Exception as e:
                # Si el archivo no existe o hay un error al descargarlo
                logger.warning(f"No se pudo descargar {latest_file}: {str(e)}")
                logger.info("Creando nuevo archivo de IDs procesados")
                return set(), self.main_file_pattern
                
        except Exception as e:
            logger.error(f"Error al obtener IDs procesados: {str(e)}")
            # En caso de error, devolver un conjunto vacío
            return set(), self.main_file_pattern
    
    def update_processed_ids(self, ids_procesados, file_name=None):
        """Actualiza el archivo JSON con los IDs de facturas procesadas en Supabase Storage.
        
        Args:
            ids_procesados (set): Conjunto de IDs de facturas procesadas
            file_name (str, optional): Nombre del archivo a actualizar
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        if file_name is None:
            file_name = self.main_file_pattern
            
        try:
            # Convertir el conjunto a lista y luego a JSON
            ids_json = json.dumps(list(ids_procesados))
            
            # Subir el archivo actualizado
            self.supabase.storage.from_(self.bucket_name).upload(
                file_name,
                ids_json.encode('utf-8'),
                {"content-type": "application/json", "upsert": "true"}  # Sobrescribir si existe
            )
            
            logger.info(f"Archivo de IDs actualizado: {file_name} con {len(ids_procesados)} IDs")
            return True
            
        except Exception as e:
            logger.error(f"Error al actualizar IDs procesados: {str(e)}")
            return False
    
    def filter_new_invoices(self, df_pagadas):
        """Filtra las facturas que no han sido procesadas anteriormente.
        
        Args:
            df_pagadas (pd.DataFrame): DataFrame con facturas pagadas
            
        Returns:
            tuple: (DataFrame con facturas nuevas, nombre del archivo actualizado)
        """
        try:
            # Obtener IDs procesados y nombre del archivo actual
            ids_procesados, current_file = self.get_processed_ids()
            
            # Filtrar facturas nuevas
            df_nuevas = df_pagadas[~df_pagadas['xml_uuid'].isin(ids_procesados)].copy()
            
            # Si hay facturas nuevas, actualizar el conjunto de IDs y el archivo
            if not df_nuevas.empty:
                # Actualizar el conjunto con los nuevos IDs
                ids_procesados.update(df_nuevas['xml_uuid'].tolist())
                
                # Actualizar el archivo en Supabase
                self.update_processed_ids(ids_procesados, current_file)
                
                logger.info(f"Se encontraron {len(df_nuevas)} facturas nuevas de {len(df_pagadas)} facturas pagadas")
            else:
                logger.info("No se encontraron facturas nuevas")
            
            return df_nuevas, current_file
            
        except Exception as e:
            logger.error(f"Error al filtrar facturas nuevas: {str(e)}")
            # En caso de error, devolver DataFrame vacío
            import pandas as pd
            return pd.DataFrame(), self.main_file_pattern


