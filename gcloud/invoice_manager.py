import os
import json
import logging
import asyncio
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
import requests.exceptions

# Importar el gestor de almacenamiento en Supabase
from supabase_storage_manager import SupabaseStorageManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('invoice_manager')

# Configurar la ruta del archivo de registro
import platform
if platform.system() == "Windows":
    registro_file = os.path.join(os.path.expanduser("~"), "Documents", "facturas_procesadas.json")
else:  # macOS, Linux, etc.
    registro_file = os.path.join(os.path.expanduser("~"), "Documents", "facturas_procesadas.json")

# Utilidades para procesar HTML (trasladadas desde utils.py)
def limpiar_texto(texto):
    """Limpia y formatea el texto eliminando espacios múltiples y caracteres especiales."""
    import re
    texto = re.sub(r'\s{2,}', ' ', texto)
    texto = texto.replace('_lnfd_', '\n').strip()
    return texto if texto else '-'

def procesar_fecha(onclick):
    """Extrae información de fechas de una cadena onclick."""
    import re
    match = re.search(
        r"OpenmodalFechas\('([^']+)','([^']+)','([^']+)','([^']+)','([^']+)','([^']+)'\)", onclick
    )
    if match:
        fechas = {
            "fecha_factura": limpiar_texto(match.group(1)),
            "fecha_recepcion": limpiar_texto(match.group(2)),
            "fecha_contrarecibo": limpiar_texto(match.group(3)),
            "fecha_autorizacion": limpiar_texto(match.group(4)),
            "fecha_pagada": limpiar_texto(match.group(5)),
            "fecha_alta": limpiar_texto(match.group(6)),
        }
        return fechas
    return {}

def procesar_enlace(onclick, clase):
    """Procesa enlaces de XML y contrarecibos."""
    import re
    clase_valida = clase.strip() in ["btn btn-primary btn-sm", "btn btn-info btn-sm"]
    if not clase_valida:
        return None
    onclick = onclick.replace('\\&quot;', '"').replace('\\"', '"').strip()
    if "openXML" in onclick:
        match = re.search(r"openXML\('([^']+)','([^']*)'\)", onclick)
        if match:
            rfc = match.group(1)
            fname = match.group(2)
            if fname:
                return f"https://palmaterraproveedores.centralinformatica.com/Download.ashx?id={fname}&rfc={rfc}&contrarecibo=0"
    elif "openContrareciboRdte" in onclick:
        match = re.search(r"openContrareciboRdte\('([^']+)','([^']*)'\)", onclick)
        if match:
            rfc = match.group(1)
            fname = match.group(2)
            if fname:
                return f"https://palmaterraproveedores.centralinformatica.com/Download.ashx?id={fname}&rfc={rfc}&contrarecibo=1"
    return None


def extraer_uuid(onclick, clase):
    """Extrae el UUID (fname) de los enlaces de XML y contrarecibos."""
    import re
    clase_valida = clase.strip() in ["btn btn-primary btn-sm", "btn btn-info btn-sm"]
    if not clase_valida:
        return None
    onclick = onclick.replace('\\&quot;', '"').replace('\\\'', '"').strip()
    if "openXML" in onclick:
        match = re.search(r"openXML\('([^']+)','([^']*)'\)", onclick)
        if match:
            fname = match.group(2)
            if fname:
                # Quitar la extensión .xml si existe
                if fname.lower().endswith('.xml'):
                    fname = fname[:-4]  # Eliminar los últimos 4 caracteres (.xml)
                return fname
            return None
    elif "openContrareciboRdte" in onclick:
        match = re.search(r"openContrareciboRdte\('([^']+)','([^']*)'\)", onclick)
        if match:
            fname = match.group(2)
            if fname:
                # Quitar la extensión .xml si existe
                if fname.lower().endswith('.xml'):
                    fname = fname[:-4]  # Eliminar los últimos 4 caracteres (.xml)
                return fname
            return None
    return None

def procesar_notas_credito(html_content):
    """Procesa el contenido HTML para extraer información de notas de crédito."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    notas_credito_data = []

    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) > 0:
            # Extraer columnas clave de la fila original
            obra = limpiar_texto(cells[0].get_text(strip=True)) if len(cells) > 0 else "-"
            proveedor = limpiar_texto(cells[1].get_text(strip=True)) if len(cells) > 1 else "-"
            encargado = limpiar_texto(cells[2].get_text(strip=True)) if len(cells) > 2 else "-"

            # Variables para guardar los enlaces
            nc_pdf_link = None
            nc_xml_link = None
            fechas = {}

            # Buscar enlaces en la fila
            enlaces = row.find_all("a")
            for enlace in enlaces:
                texto_enlace = enlace.text.strip().upper()
                onclick = enlace.get("onclick", "")
                clase = " ".join(enlace.get("class", []))

                if "OpenmodalFechas" in onclick:
                    fechas = procesar_fecha(onclick)

                # Detectar PDF de la nota de crédito (NC)
                if texto_enlace == "NC":
                    if "disabled" not in clase and onclick:
                        nc_pdf_link = procesar_enlace(onclick, clase)

                # Detectar XML (último enlace XML en la fila es de NC)
                if texto_enlace == "XML":
                    if "disabled" not in clase and onclick:
                        nc_xml_link = procesar_enlace(onclick, clase)

            # Si se detecta un enlace NC válido, crear una fila para la nota de crédito
            if nc_pdf_link or nc_xml_link:
                notas_credito_data.append({
                    "Obra": obra,
                    "Proveedor": proveedor,
                    "Residente": encargado,
                    "Número": "",
                    "Estatus": "Nota de crédito",
                    "XML": nc_xml_link if nc_xml_link else None,
                    **fechas,
                })

    # Crear un DataFrame con las notas de crédito y eliminar filas sin PDF
    return pd.DataFrame(notas_credito_data)

def procesar_html_content(html_content):
    """Procesa el contenido HTML para extraer información de facturas."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    data_rows = []
    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) > 0:
            obra = limpiar_texto(cells[0].get_text(strip=True))
            proveedor = limpiar_texto(cells[1].get_text(strip=True))
            encargado = limpiar_texto(cells[2].get_text(strip=True))
            numero = limpiar_texto(cells[3].get_text(strip=True)) if len(cells) > 3 else "-"
            estatus = limpiar_texto(cells[4].get_text(strip=True)) if len(cells) > 4 else "-"
            total = limpiar_texto(cells[5].get_text(strip=True)) if len(cells) > 5 else "-"

            xml_link = pdf_link = oc_link = rem_link = None
            fechas = {}

            for enlace in row.find_all("a"):
                onclick = enlace.get("onclick", "")
                clase = " ".join(enlace.get("class", []))
                if "OpenmodalFechas" in onclick:
                    fechas = procesar_fecha(onclick)
                elif "XML" in enlace.text.strip().upper() and not xml_link:
                    xml_link = procesar_enlace(onclick, clase)
                elif "PDF" in enlace.text.strip().upper() and not pdf_link:
                    pdf_link = procesar_enlace(onclick, clase)
                elif "OC" in enlace.text.strip().upper() and not oc_link:
                    oc_link = procesar_enlace(onclick, clase)
                elif "REM" in enlace.text.strip().upper() and not rem_link:
                    rem_link = procesar_enlace(onclick, clase)

            # Extraer UUIDs para los enlaces
            xml_uuid = None
            
            for enlace in row.find_all("a"):
                onclick = enlace.get("onclick", "")
                clase = " ".join(enlace.get("class", []))
                if "XML" in enlace.text.strip().upper() and not xml_uuid:
                    xml_uuid = extraer_uuid(onclick, clase)
                    
            # Determinar tipo_gasto basado en el nombre de la obra
            if " / Servicios" in obra:
                tipo_gasto = "SERVICIO"
            elif " / Garantías" in obra:
                tipo_gasto = "GARANTIA"
            else:
                tipo_gasto = "COSTO DIRECTO"
            
            data_rows.append({
                "obra": obra,
                "tipo_gasto": tipo_gasto,
                "proveedor": proveedor,
                "residente": encargado,
                "folio": numero,
                "estatus": estatus,
                "total": total,
                "url_xml": xml_link,
                "url_pdf": pdf_link,
                "url_oc": oc_link,
                "url_rem": rem_link,
                "xml_uuid": xml_uuid,
                **fechas,
            })
    return pd.DataFrame(data_rows)

class InvoiceManager:
    """Clase para gestionar consultas y procesamiento de facturas sin dependencias de UI."""
    
    def __init__(self, session=None):
        """Inicializa el gestor de facturas.
        
        Args:
            session (requests.Session, optional): Sesión de requests con las cookies autenticadas.
                                               Si no se proporciona, se creará una nueva.
        """
        self.session = session or requests.Session()
        if not self.session.headers.get('Content-Type'):
            self.session.headers.update({'Content-Type': 'application/json; charset=utf-8'})
            
        self.df_resultados_facturas = []
        self.df_resultados_nc = []
        self.errores = []
        self.df_facturas_final = None
        self.df_notas_final = None
        self.obras_lista = []
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def _realizar_consulta_facturas(self, obra_name, obra_value):
        """Realiza la consulta de facturas para una obra específica.
        
        Args:
            obra_name (str): Nombre de la obra
            obra_value (str): ID de la obra
            
        Returns:
            tuple: (df_facturas, df_notas)
        """
        import html
        
        # URL y datos para la consulta de facturas
        url = "https://palmaterraproveedores.centralinformatica.com/WSUnico.asmx/GetAllFacturasBusqueda"

        # Consulta para Facturas (CFDI)
        data = {
            "Estatus": "0",
            "Residente": "",
            "Obra": obra_value,
            "Proveedor": "",
            "Fecha": "",
            "FechaDesde": "",
            "FechaHasta": "",
            "Numero": ""
        }

        try:
            # Realizamos la solicitud POST
            logger.info(f"Consultando facturas para obra: {obra_name} (ID: {obra_value})")
            response = self.session.post(url, json=data, timeout=15)
            if response.status_code != 200:
                raise ValueError(f"Error en la consulta: {response.status_code}")

            # Procesar la respuesta JSON
            response_data = json.loads(response.content.decode("utf-8"))
            nested_json = response_data.get("d", "")
            if not nested_json:
                logger.warning(f"No se encontraron datos para la obra: {obra_name}")
                return pd.DataFrame(), pd.DataFrame()

            nested_data = json.loads(nested_json)
            if "tbodyFacturas" not in nested_data:
                logger.warning(f"No se encontraron facturas para la obra: {obra_name}")
                return pd.DataFrame(), pd.DataFrame()

            extracted_html = nested_data.get("tbodyFacturas", "")
            if not extracted_html:
                logger.warning(f"HTML de facturas vacío para la obra: {obra_name}")
                return pd.DataFrame(), pd.DataFrame()

            clean_html = html.unescape(extracted_html)

            # Procesar HTML para facturas y notas
            df = procesar_html_content(clean_html)
            df_nc = procesar_notas_credito(clean_html)

            # Añadir el ID de obra a los DataFrames
            df['cuenta_gasto'] = obra_value
            logger.info(f"Consultados {len(df)} facturas y {len(df_nc)} notas de crédito para obra: {obra_name}")
            return df, df_nc
        except Exception as e:
            logger.error(f"Error consultando obra {obra_name}: {str(e)}")
            raise
    
    async def consulta_factura_async(self, obra):
        """Versión asíncrona para consultar una factura.
        
        Args:
            obra (dict): Diccionario con 'name' y 'value' de la obra
            
        Returns:
            tuple: (df_facturas, df_notas, obra, error)
        """
        try:
            # Usamos un executor para ejecutar la función bloqueante en un hilo separado
            loop = asyncio.get_event_loop()
            df_facturas, df_nc = await loop.run_in_executor(
                None,
                lambda: self._realizar_consulta_facturas(obra['name'], obra['value'])
            )
            return df_facturas, df_nc, obra, None
        except Exception as e:
            error_msg = f"Error consultando la obra {obra['name']}: {str(e)}"
            logger.error(error_msg)
            return pd.DataFrame(), pd.DataFrame(), obra, error_msg
    
    async def consultar_facturas(self, obras_lista, max_workers=None):
        """Consulta las facturas de múltiples obras de forma concurrente.
        
        Args:
            obras_lista (list): Lista de diccionarios con 'name' y 'value' de cada obra
            max_workers (int, optional): Número máximo de consultas concurrentes
            
        Returns:
            dict: {'facturas': df_facturas, 'notas': df_notas, 'errores': errores}
        """
        # Guardar la lista de obras para referencia
        self.obras_lista = obras_lista
        
        # Limpiar resultados anteriores
        self.df_resultados_facturas.clear()
        self.df_resultados_nc.clear()
        self.errores.clear()
        
        # Calcular el número máximo de trabajadores
        max_workers = max_workers or min(32, len(obras_lista))
        
        # Lanzar todas las consultas de forma asíncrona
        logger.info(f"Iniciando consulta de facturas para {len(obras_lista)} obras")
        
        # Crear tareas para cada obra
        tareas = [self.consulta_factura_async(obra) for obra in obras_lista]
        
        # Ejecutar las tareas y procesar los resultados
        for i, tarea_completada in enumerate(asyncio.as_completed(tareas)):
            df_facturas, df_nc, obra, error = await tarea_completada
            
            if error:
                self.errores.append(error)
            else:
                # Siempre añadimos el resultado al contador aunque esté vacío
                if not df_facturas.empty:
                    self.df_resultados_facturas.append(df_facturas)
                if not df_nc.empty:
                    self.df_resultados_nc.append(df_nc)
            
            # Reportar progreso
            logger.info(f"Progreso: {i+1}/{len(obras_lista)} obras procesadas")
        
        # Consolidar resultados
        return self._consolidar_resultados()
    
    def _consolidar_resultados(self):
        """Consolida los resultados de las consultas en DataFrames globales.
        
        Returns:
            dict: {'facturas': df_global, 'notas': df_nc_global, 'errores': errores}
        """
        # Consolidar facturas
        if self.df_resultados_facturas:
            df_global = pd.concat(self.df_resultados_facturas, ignore_index=True)
            self.df_facturas_final = df_global
        else:
            df_global = pd.DataFrame()
            self.df_facturas_final = df_global
        
        # Consolidar notas de crédito
        if self.df_resultados_nc:
            df_nc_global = pd.concat(self.df_resultados_nc, ignore_index=True)
            self.df_notas_final = df_nc_global
        else:
            df_nc_global = pd.DataFrame()
            self.df_notas_final = df_nc_global
        
        logger.info(f"Consulta finalizada: {len(df_global)} facturas y {len(df_nc_global)} notas de crédito")
        logger.info(f"Errores: {len(self.errores)}")
        
        return {
            'facturas': df_global,
            'notas': df_nc_global,
            'errores': self.errores
        }
    
    def consultar_facturas_sync(self, obras_lista):
        """Versión sincrónica (bloqueante) de consultar_facturas.
        
        Args:
            obras_lista (list): Lista de diccionarios con 'name' y 'value' de cada obra
            
        Returns:
            dict: {'facturas': df_facturas, 'notas': df_notas, 'errores': errores}
        """
        # Para entornos que no soportan asyncio o cuando se prefiere un enfoque sincrónico
        import asyncio
        
        try:
            # Crear un nuevo evento loop si estamos en un hilo que no tiene uno
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Ejecutar la consulta asíncrona y esperar a que termine
            return loop.run_until_complete(self.consultar_facturas(obras_lista))
        finally:
            # Cerramos el loop para liberarlo
            if loop and loop.is_running():
                loop.stop()
            if loop and not loop.is_closed():
                loop.close()
    
    def filtrar_facturas_pagadas(self, df_global):
        """Filtra las facturas pagadas y en proceso de pago.
        
        Args:
            df_global (pd.DataFrame): DataFrame con todas las facturas
            
        Returns:
            pd.DataFrame: DataFrame con facturas filtradas y procesadas
        """
        # Filtrar facturas pagadas y en proceso de pago
        df_pagadas = df_global[df_global['estatus'].isin(["Pagada", "Proceso de Pago", "RevisaRes"])].copy()
        
        # Convertir las fechas al formato datetime y luego al formato Supabase
        date_columns = [
            "fecha_factura", "fecha_recepcion", "fecha_contrarecibo",
            "fecha_autorizacion", "fecha_pagada", "fecha_alta"
        ]
        for col in date_columns:
            if col in df_pagadas.columns:
                # Convertir a datetime
                df_pagadas[col] = pd.to_datetime(df_pagadas[col], format="%d/%m/%y %H:%M", errors='coerce')
                # Formatear para Supabase, manejando NaT (Not a Time) que resultan de 'coerce'
                # Usamos fillna(pd.NA) para convertir NaT a Nulos que Supabase pueda manejar mejor que un string 'NaT'
                df_pagadas[col] = df_pagadas[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else pd.NA)

        return df_pagadas
    
    def filtrar_nuevas_facturas(self, df_pagadas, supabase_url=None, supabase_key=None, bucket_name="1facturas", backup_bucket_name="facturas_backup"):
        """Filtra las facturas que no han sido procesadas anteriormente usando Supabase Storage.
        
        Args:
            df_pagadas (pd.DataFrame): DataFrame con facturas pagadas
            supabase_url (str, optional): URL de la API de Supabase
            supabase_key (str, optional): Clave de API de Supabase
            bucket_name (str, optional): Nombre del bucket principal para los archivos JSON
            backup_bucket_name (str, optional): Nombre del bucket para las copias de seguridad
            
        Returns:
            pd.DataFrame: DataFrame con facturas nuevas
        """
        # Verificar que se proporcionaron las credenciales de Supabase
        if not supabase_url or not supabase_key:
            # Intentar obtener de variables de entorno si no se proporcionaron directamente
            import os
            supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
            supabase_key = supabase_key or os.environ.get("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                error_msg = "Error: Se requieren las credenciales de Supabase (URL y Key) para gestionar los IDs de facturas"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        try:
            # Crear gestor de almacenamiento de Supabase
            storage_manager = SupabaseStorageManager(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                bucket_name=bucket_name,
                backup_bucket_name=backup_bucket_name
            )
            
            # Filtrar facturas nuevas usando el gestor de Supabase
            df_nuevas, current_file = storage_manager.filter_new_invoices(df_pagadas)
            
            logger.info(f"Proceso completado. Archivo actual: {current_file}")
            return df_nuevas
            
        except Exception as e:
            logger.error(f"Error al filtrar facturas nuevas con Supabase: {str(e)}")
            # En caso de error, usar el método local como respaldo
            return self._filtrar_nuevas_facturas_local(df_pagadas)
            
    def _filtrar_nuevas_facturas_local(self, df_pagadas):
        """[MÉTODO DE RESPALDO] Filtra las facturas nuevas usando el sistema de archivos local.
        Este método solo se usa como respaldo si falla la integración con Supabase.
        
        Args:
            df_pagadas (pd.DataFrame): DataFrame con facturas pagadas
            
        Returns:
            pd.DataFrame: DataFrame con facturas nuevas
        """
        import datetime
        
        logger.warning("Utilizando método local (respaldo) para filtrar facturas nuevas")
        
        # Definir la ruta del archivo de registro local
        registro_file = os.path.expanduser(
            f'~/Downloads/facturas_hashlib_id_{datetime.datetime.now().strftime("%d%m%Y")}.json'
        )
        
        # Leer el registro existente (si existe)
        if os.path.exists(registro_file):
            with open(registro_file, "r") as registro_entrada:
                ids_procesados = set(json.load(registro_entrada))
        else:
            ids_procesados = set()
        
        # Filtrar facturas nuevas
        df_nuevas = df_pagadas[~df_pagadas['hash5'].isin(ids_procesados)].copy()
        
        # Actualizar el registro con los nuevos IDs
        ids_procesados.update(df_nuevas['hash5'].tolist())
        
        with open(registro_file, "w") as registro_salida:
            json.dump(list(ids_procesados), registro_salida)
        
        logger.info(f"[LOCAL] Se encontraron {len(df_nuevas)} facturas nuevas de {len(df_pagadas)} facturas pagadas")
        return df_nuevas
