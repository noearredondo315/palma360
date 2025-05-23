import os
import pandas as pd
from supabase import create_client
import logging
from dotenv import load_dotenv
import os

load_dotenv()  # Carga desde .env

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('supabase_uploader')


class SupabaseUploader:
    """Clase para gestionar la carga de datos a Supabase."""

    def guardar_lote_fallido(self, batch, lote_num):
        """
        Guarda un lote fallido en formato JSON para reintento posterior.
        
        Args:
            batch (list): Lista de registros del lote fallido
            lote_num (int): Número del lote
            
        Returns:
            str: Ruta donde se guardó el archivo JSON
        """
        import os
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        carpeta = "lotes_fallidos"
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(carpeta, f"lote_{lote_num:03d}_{timestamp}.json")
        
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        
        logger.warning(f"Lote {lote_num} guardado en {ruta} para reintento posterior")
        return ruta
        
    def reintentar_lotes_fallidos(self):
        """
        Reintenta la carga de todos los lotes fallidos guardados en la carpeta 'lotes_fallidos'.
        Utiliza tenacity para realizar reintentos automáticos en caso de fallos de conexión.
        
        Returns:
            dict: Resultado de los reintentos
        """
        import os
        import json
        import glob
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
        import requests
        
        carpeta = "lotes_fallidos"
        if not os.path.exists(carpeta):
            logger.info("No hay carpeta de lotes fallidos para reintentar")
            return {"message": "No hay lotes fallidos para reintentar", "count": 0}
        
        archivos_json = glob.glob(os.path.join(carpeta, "*.json"))
        if not archivos_json:
            logger.info("No hay archivos de lotes fallidos para reintentar")
            return {"message": "No hay lotes fallidos para reintentar", "count": 0}
        
        logger.info(f"Encontrados {len(archivos_json)} archivos de lotes fallidos para reintentar")
        
        # Definir función de inserción con reintentos
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            retry=retry_if_exception_type((requests.exceptions.RequestException,))
        )
        def insertar_con_reintentos(batch, nombre_archivo):
            """Insertar lote con reintentos automáticos"""
            response = (
                self.supabase.table("portal_desglosado")
                .insert(batch)
                .execute()
            )
            return response
        
        # Resultados
        resultados = {
            "total": len(archivos_json),
            "exito": 0,
            "error": 0,
            "detalle": []
        }
        
        # Procesar cada archivo de lote fallido
        for archivo in archivos_json:
            nombre_archivo = os.path.basename(archivo)
            logger.info(f"Reintentando carga del lote fallido: {nombre_archivo}")
            
            try:
                # Cargar datos del archivo JSON
                with open(archivo, "r", encoding="utf-8") as f:
                    batch = json.load(f)
                
                # Reintentar inserción con manejo automático de reintentos
                try:
                    response = insertar_con_reintentos(batch, nombre_archivo)
                    
                    # Verificar respuesta
                    if hasattr(response, 'data') and response.data:
                        registros_exito = len(response.data)
                        logger.info(f"Reintento exitoso para {nombre_archivo}: {registros_exito} registros cargados")
                        
                        # Renombrar archivo para indicar que fue procesado exitosamente
                        archivo_procesado = archivo.replace('.json', '.procesado.json')
                        os.rename(archivo, archivo_procesado)
                        
                        resultados["exito"] += 1
                        resultados["detalle"].append({
                            "archivo": nombre_archivo,
                            "estado": "exito",
                            "registros": registros_exito
                        })
                    else:
                        raise Exception("Respuesta sin datos claros de éxito")
                        
                except Exception as retry_error:
                    logger.error(f"Error en reintento de {nombre_archivo} después de múltiples intentos: {str(retry_error)}")
                    resultados["error"] += 1
                    resultados["detalle"].append({
                        "archivo": nombre_archivo,
                        "estado": "error",
                        "error": str(retry_error)
                    })
            except Exception as e:
                logger.error(f"Error al procesar archivo {nombre_archivo}: {str(e)}")
                resultados["error"] += 1
                resultados["detalle"].append({
                    "archivo": nombre_archivo,
                    "estado": "error_procesamiento",
                    "error": str(e)
                })
        
        # Resumen final
        if resultados["exito"] > 0:
            logger.info(f"Reintento completado: {resultados['exito']} lotes exitosos, {resultados['error']} con error")
        else:
            logger.warning(f"Reintento completado sin éxito: {resultados['error']} lotes con error")
            
        return resultados

    def subir_predicciones_portal_desglosado(self, df_predicciones):
        """
        Inserta todos los registros del DataFrame en la tabla portal_desglosado de Supabase.
        No realiza comparación ni actualización, solo inserción masiva.
        Convierte automáticamente columnas con objetos uuid.UUID a string.
        Procesa los datos en lotes para evitar errores de tamaño de payload.
        Guarda lotes fallidos en disco para reintento posterior.
        
        Args:
            df_predicciones (pd.DataFrame): DataFrame con las predicciones a subir
        Returns:
            dict: Respuesta de la operación de Supabase
        """
        import uuid
        if df_predicciones.empty:
            logger.warning("No hay predicciones para cargar a Supabase (portal_desglosado)")
            return {"message": "No hay predicciones para cargar", "count": 0}
        try:
            import numpy as np
            df_temp = df_predicciones.copy()
            
            # Convertir columnas de tipo Timestamp a string
            for col in df_temp.columns:
                if pd.api.types.is_datetime64_any_dtype(df_temp[col]):
                    df_temp[col] = df_temp[col].apply(
                        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else None
                    )
                # Convertir columnas de tipo UUID a string
                if df_temp[col].dtype == 'object' and df_temp[col].apply(lambda x: isinstance(x, uuid.UUID)).any():
                    df_temp[col] = df_temp[col].apply(lambda x: str(x) if isinstance(x, uuid.UUID) else x)
            
            # Reemplazar NaN, inf y -inf por None
            df_temp = df_temp.replace([np.nan, np.inf, -np.inf], None)
            
            # Convertir a registros
            registros = df_temp.to_dict(orient="records")
            
            # Procesar en lotes para evitar payloads demasiado grandes
            batch_size = 1000  # Ajusta este tamaño según sea necesario
            total_registros = len(registros)
            total_procesados = 0
            total_exito = 0
            total_error = 0
            lotes_fallidos = []
            
            logger.info(f"Iniciando carga de {total_registros} registros en lotes de {batch_size}")
            
            for i in range(0, total_registros, batch_size):
                batch = registros[i:i+batch_size]
                batch_num = i//batch_size + 1
                total_batches = (total_registros + batch_size - 1)//batch_size
                
                logger.info(f"Procesando lote {batch_num} de {total_batches} ({len(batch)} registros)")
                
                try:
                    response = (
                        self.supabase.table("portal_desglosado")
                        .insert(batch)
                        .execute()
                    )
                    
                    # Verificar respuesta
                    if hasattr(response, 'data') and response.data:
                        registros_exito = len(response.data)
                        total_exito += registros_exito
                        logger.info(f"Lote {batch_num}: {registros_exito} registros cargados exitosamente")
                    else:
                        logger.warning(f"Lote {batch_num}: Respuesta sin datos claros de éxito")
                    
                    total_procesados += len(batch)
                    
                except Exception as batch_error:
                    logger.error(f"Error al cargar lote {batch_num}: {str(batch_error)}")
                    ruta_guardado = self.guardar_lote_fallido(batch, batch_num)
                    lotes_fallidos.append({
                        "lote": batch_num,
                        "registros": len(batch),
                        "ruta": ruta_guardado,
                        "error": str(batch_error)
                    })
                    total_error += len(batch)
            
            logger.info(f"Carga completa: {total_exito} exitosos, {total_error} con error de un total de {total_registros}")
            
            resultados_reintentos = None
            if lotes_fallidos:
                logger.warning(f"Se guardaron {len(lotes_fallidos)} lotes fallidos para reintento posterior")
                logger.info("Iniciando reintentos automáticos para lotes fallidos...")
                resultados_reintentos = self.reintentar_lotes_fallidos()
                
                if resultados_reintentos.get("exito", 0) > 0:
                    total_exito += resultados_reintentos.get("exito", 0)
                    total_error -= resultados_reintentos.get("exito", 0)
                    logger.info(f"Reintentos completados: {resultados_reintentos['exito']} lotes recuperados")
            
            return {
                "message": "Predicciones cargadas exitosamente en portal_desglosado",
                "count": total_procesados,
                "exito": total_exito,
                "error": total_error,
                "total": total_registros,
                "lotes_fallidos": lotes_fallidos,
                "resultados_reintentos": resultados_reintentos
            }
        except Exception as e:
            logger.error(f"Error al cargar predicciones en portal_desglosado: {e}")
            return {"message": f"Error al cargar predicciones: {e}", "count": 0}


    def __init__(self, url=None, key=None):
        """Inicializa el cargador de Supabase.
        
        Args:
            url (str, optional): URL de Supabase. Si no se proporciona, se busca en variables de entorno.
            key (str, optional): Clave de Supabase. Si no se proporciona, se busca en variables de entorno.
        """
        # Si no se proporcionan credenciales, buscar en variables de entorno
        self.url = url or os.environ.get('SUPABASE_URL')
        self.key = key or os.environ.get('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("Se requieren URL y clave de Supabase. Proporciónalas como parámetros o configura variables de entorno.")

        # Inicializar cliente de Supabase
        self.supabase = create_client(self.url, self.key)
        logger.info("Cliente Supabase inicializado correctamente")

    def cargar_facturas_pagadas(self, df_pagadas):
        """Carga el DataFrame de facturas pagadas a la tabla portal_concentrado en Supabase.
        Implementa una lógica optimizada para comparar y actualizar solo registros que han cambiado:
        - Registros nuevos: inserta todos los campos
        - Registros existentes: compara solo 'estatus' y 'obra', si hay cambios actualiza el registro completo
        Usa operaciones por lotes para mayor eficiencia.
        
        Args:
            df_pagadas (pd.DataFrame): DataFrame con facturas pagadas procesadas
            
        Returns:
            dict: Respuesta de la operación de Supabase
        """
        # Verificar que el DataFrame tiene la columna xml_uuid como clave para upsert
        if 'xml_uuid' not in df_pagadas.columns:
            logger.error("El DataFrame debe contener la columna 'xml_uuid' para realizar upsert")
            raise ValueError("El DataFrame debe contener la columna 'xml_uuid' para realizar upsert")

        # Si no hay facturas, retornar temprano
        if df_pagadas.empty:
            logger.warning("No hay facturas para cargar a Supabase")
            return {"message": "No hay facturas para cargar", "count": 0}
        
        try:
            # Hacer una copia del DataFrame para no modificar el original
            df_temp = df_pagadas.copy()
            
            # Asegurar que xml_uuid es string y normalizar
            df_temp['xml_uuid'] = df_temp['xml_uuid'].astype(str).str.lower().str.strip()
            
            # Columnas que consideramos para determinar si un registro ha cambiado
            columnas_a_comparar = ['estatus', 'obra']
            
            # 1. Obtener registros existentes con las columnas que nos interesan para comparar
            logger.info("Consultando registros existentes en Supabase para comparación...")
            response_existing = (
                self.supabase.table("portal_concentrado")
                .select("xml_uuid,estatus,obra,tipo_gasto")
                .execute()
            )
            
            # Si no hay datos, todos son nuevos
            if not response_existing.data:
                logger.info("No se encontraron registros existentes en Supabase")
                # Insertar todos los registros de una vez, asegurando que las fechas sean strings
                df_temp_safe = df_temp.copy()
                
                # Convertir todas las columnas de tipo Timestamp a string
                for col in df_temp_safe.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_temp_safe[col]):
                        df_temp_safe[col] = df_temp_safe[col].apply(
                            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else None
                        )
                        
                payload = df_temp_safe.to_dict(orient="records")
                response = (
                    self.supabase.table("portal_concentrado")
                    .insert(payload)
                    .execute()
                )
                return {
                    "message": "Operación completada exitosamente", 
                    "count": len(df_temp),
                    "nuevos": len(df_temp),
                    "actualizados": 0
                }
                
            # 2. Convertir los datos existentes a un DataFrame para facilitar la comparación
            df_existente = pd.DataFrame(response_existing.data)
            
            # Normalizar los UUIDs en los datos existentes para una comparación consistente
            if not df_existente.empty:
                df_existente['xml_uuid'] = df_existente['xml_uuid'].astype(str).str.lower().str.strip()
                # Convertir fechas a objetos datetime para comparación adecuada
                if 'fecha_pagada' in df_existente.columns:
                    df_existente['fecha_pagada'] = pd.to_datetime(df_existente['fecha_pagada'], errors='coerce')
            
            # 3. Convertir fechas en el DataFrame temporal también
            if 'fecha_pagada' in df_temp.columns:
                df_temp['fecha_pagada'] = pd.to_datetime(df_temp['fecha_pagada'], errors='coerce')
            
            # 4. Identificar registros nuevos y existentes
            uuids_existentes = set(df_existente['xml_uuid'].unique()) if not df_existente.empty else set()
            
            # Dividir el DataFrame en nuevos y existentes
            df_nuevos = df_temp[~df_temp['xml_uuid'].isin(uuids_existentes)]
            df_posibles_actualizaciones = df_temp[df_temp['xml_uuid'].isin(uuids_existentes)]
            
            # 5. Entre los existentes, identificar cuáles realmente cambiaron
            registros_a_actualizar = []
            count_sin_cambios = 0
            
            if not df_posibles_actualizaciones.empty and not df_existente.empty:
                # Para cada registro que podría actualizarse, verificar si realmente cambió
                for _, fila_nueva in df_posibles_actualizaciones.iterrows():
                    uuid = fila_nueva['xml_uuid']
                    # Encontrar el registro correspondiente en los datos existentes
                    fila_existente = df_existente[df_existente['xml_uuid'] == uuid]
                    
                    if not fila_existente.empty:
                        # Verificar si hay cambios en las columnas de interés
                        cambios = False
                        
                        for col in columnas_a_comparar:
                            if col in fila_nueva and col in fila_existente.columns:
                                # Manejar valores nulos/NaN consistentemente
                                val_nuevo = None if pd.isna(fila_nueva[col]) else fila_nueva[col]
                                val_existente = None if pd.isna(fila_existente.iloc[0][col]) else fila_existente.iloc[0][col]
                                
                                # Comparar valores y marcar si hay cambios
                                if val_nuevo != val_existente:
                                    cambios = True
                                    break  # No necesitamos seguir comprobando si ya detectamos cambios
                        
                        # Si hay cambios, agregar TODAS las columnas del registro a actualizar
                        if cambios:
                            # Crear un diccionario con TODAS las columnas del DataFrame
                            datos_completos = {}
                            for col in fila_nueva.index:
                                # Asegurar que las fechas sean strings, no objetos Timestamp
                                if isinstance(fila_nueva[col], pd.Timestamp):
                                    datos_completos[col] = fila_nueva[col].strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    datos_completos[col] = None if pd.isna(fila_nueva[col]) else fila_nueva[col]
                            registros_a_actualizar.append(datos_completos)
                        else:
                            count_sin_cambios += 1
            
            # Contar operaciones realizadas
            count_operations = 0
            count_nuevos = len(df_nuevos)
            count_actualizados = len(registros_a_actualizar)
            
            # 6. Insertar registros nuevos (todos los campos) en lote
            if not df_nuevos.empty:
                logger.info(f"Insertando {count_nuevos} nuevos registros...")
                # Convertir DataFrame a diccionario pero asegurando que las fechas sean strings
                # Primero hacemos una copia para no modificar el original
                df_nuevos_safe = df_nuevos.copy()
                
                # Convertir todas las columnas de tipo Timestamp a string
                for col in df_nuevos_safe.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_nuevos_safe[col]):
                        df_nuevos_safe[col] = df_nuevos_safe[col].apply(
                            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else None
                        )
                
                payload_nuevos = df_nuevos_safe.to_dict(orient="records")
                response_nuevos = (
                    self.supabase.table("portal_concentrado")
                    .insert(payload_nuevos)
                    .execute()
                )
                count_operations += count_nuevos
                logger.info(f"Insertados {count_nuevos} nuevos registros")
            
            # 7. Actualizar registros existentes que han cambiado (actualiza el registro completo) en lote
            if registros_a_actualizar:
                logger.info(f"Actualizando {count_actualizados} registros modificados (de {len(df_posibles_actualizaciones)} existentes)...")
                
                # Actualizar los registros en lotes de 100 para mejorar rendimiento sin sobrecargar
                batch_size = 100
                for i in range(0, len(registros_a_actualizar), batch_size):
                    batch = registros_a_actualizar[i:i+batch_size]
                    try:
                        # No necesitamos convertir otra vez los valores Timestamp porque ya lo hicimos al crear datos_completos
                        
                        # Usar upsert con on_conflict (sintaxis de Python) correctamente
                        response_batch = (
                            self.supabase.table("portal_concentrado")
                            .upsert(batch, on_conflict="xml_uuid")
                            .execute()
                        )
                        logger.info(f"Procesado lote {i//batch_size + 1} de {(len(registros_a_actualizar) + batch_size - 1)//batch_size}")
                    except Exception as e:
                        logger.error(f"Error al actualizar lote: {str(e)}")
                
                count_operations += count_actualizados
                logger.info(f"Actualizados {count_actualizados} registros existentes")
            else:
                logger.info(f"No se requieren actualizaciones. {count_sin_cambios} registros sin cambios.")
            
            return {
                "message": "Operación completada exitosamente", 
                "count": count_operations,
                "nuevos": count_nuevos,
                "actualizados": count_actualizados,
                "sin_cambios": count_sin_cambios
            }
        
        except Exception as e:
            logger.error(f"Error al cargar facturas a Supabase: {str(e)}")
            raise

    def map_categoria(self, df):
        """
        Asigna el campo categoria_id en el DataFrame de facturas pagadas según la subcategoría predicha,
        utilizando el mapeo almacenado en la tabla 'categorias_subcategorias' de Supabase.

        Args:
            df (pd.DataFrame): DataFrame que debe contener la columna 'subcategoria'.
                Se le añadirá la columna 'categoria_id' mapeada desde Supabase.

        Raises:
            Exception: Propaga cualquier excepción que ocurra durante el proceso e imprime el error en el logger.
        """
        try:
            # Obtener el mapeo de subcategorías a categoria_id desde Supabase
            logger.info("Consultando registros existentes en Supabase para comparación...")
            response = self.supabase.table('categorias_subcategorias').select('subcategoria, categoria_id').execute()
            categoria_mapping = {row['subcategoria']: row['categoria_id'] for row in response.data}

            # Mapear categoria_id en el DataFrame según la subcategoría predicha
            df['categoria_id'] = df['subcategoria'].map(categoria_mapping)

            # Verificar si hay subcategorías no mapeadas a categoria_id
            if df['categoria_id'].isna().any():
                print("Advertencia: Algunas subcategorías no tienen categoria_id asignada")
                # Opcional: aquí se puede asignar un valor por defecto o manejar el caso según necesidad

            return df
        except Exception as e:
            logger.error(f"Error al mapear las categorías: {str(e)}")
            raise

