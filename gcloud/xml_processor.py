import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import tempfile
import shutil
import io  # Import io for BytesIO

# Configurar logging
logger = logging.getLogger('xml_processor')

class XMLProcessor:
    """Gestiona el procesamiento de archivos XML (ahora principalmente a través de métodos estáticos)."""

    errores = [] # Make it a class variable if needed, or remove if procesar_xml is fully deprecated

    @staticmethod
    def _parse_xml_content(xml_content_bytes: bytes, row_data: pd.Series, xml_identifier: str = 'N/A'):
        """Parsea el contenido XML (bytes) y extrae sus conceptos."""
        columnas_empty = [
            'obra', 'cuenta_gasto', 'proveedor', 'residente', 'folio', 'estatus',
            'fecha_factura', 'fecha_recepcion', 'fecha_pagada', 'fecha_autorizacion',
            'clave_producto', 'clave_unidad', 'cantidad', 'descripcion', 'unidad',
            'precio_unitario', 'subtotal', 'descuento', 'venta_tasa_0', 'venta_tasa_16', 'moneda',
            'total_iva', 'retencion_iva', 'retencion_isr', 'total_ish', 'total', 'serie',
            'url_pdf', 'url_oc', 'url_rem', 'xml_uuid' # Añadido XML_UUID
        ]
        
        # Adaptar nombres de columnas si es necesario desde row_data
        # (Ejemplo: si en row_data viene 'Fecha Factura' pero necesitamos 'FECHA_FACTURA')
        # Esta lógica asume que row_data ya tiene los nombres correctos o se mapean aquí.
        
        try:
            # Usar ET.fromstring para parsear desde bytes
            root = ET.fromstring(xml_content_bytes)
            version = root.attrib.get('Version', '4.0') # Default a 4.0 si no existe

            # Definir namespaces según la versión
            if version.startswith("4."):
                namespace = {'cfdi': 'http://www.sat.gob.mx/cfd/4'}
            elif version.startswith("3.3"):
                namespace = {'cfdi': 'http://www.sat.gob.mx/cfd/3'}
            else:
                logger.warning(f"Versión de CFDI no soportada '{version}' en XML {xml_identifier}. Intentando procesar con NS 4.0...")
                namespace = {'cfdi': 'http://www.sat.gob.mx/cfd/4'}

            implocal_namespace = {'implocal': 'http://www.sat.gob.mx/implocal'}

            emisor_node = root.find('.//cfdi:Emisor', namespace)
            rfc_emisor = emisor_node.attrib.get('Rfc', None) if emisor_node is not None else None
            if not rfc_emisor:
                logger.warning(f"No se encontró RFC del Emisor en XML {xml_identifier}. brincando...")
                # Considerar si devolver DF vacío o permitir continuar

            conceptos_nodes = root.findall('.//cfdi:Concepto', namespace)
            if not conceptos_nodes:
                logger.warning(f"El XML {xml_identifier} no contiene conceptos.")
                return pd.DataFrame(columns=columnas_empty)

            # Buscar el nodo del emisor
            emisor_node = root.find('.//cfdi:Emisor', namespace)
            if emisor_node is not None:
                rfc_emisor = emisor_node.attrib.get('Rfc', None)
                if not rfc_emisor:
                    raise ValueError(f"No se encontró el atributo 'Rfc' en el nodo Emisor en el archivo {xml_file}.")
            else:
                raise ValueError(f"No se encontró el nodo 'cfdi:Emisor' en el archivo {xml_file}.")

            letras_iniciales = re.match(r"^[A-Z]+", rfc_emisor).group()


            conceptos = []
            for concepto in conceptos_nodes:
                cantidad = float(concepto.attrib.get('Cantidad', 0))
                clave_prod_serv = concepto.attrib.get('ClaveProdServ', '')
                clave_unidad = concepto.attrib.get('ClaveUnidad', '')
                descripcion = concepto.attrib.get('Descripcion', '')
                valor_unitario = float(concepto.attrib.get('ValorUnitario', 0))
                subtotal = float(concepto.attrib.get('Importe', 0))
                unidad = concepto.attrib.get('Unidad', '')
                descuento = float(concepto.attrib.get('Descuento', 0))

                iva_traslado = 0.0
                retencion_iva = 0.0
                retencion_isr = 0.0
                ish_importe = 0.0
                base_0_concepto = 0.0
                base_16_concepto = 0.0

                impuestos = concepto.find('.//cfdi:Impuestos', namespace)
                if impuestos is not None:
                    traslados = impuestos.findall('.//cfdi:Traslado', namespace)
                    for traslado in traslados:
                        impuesto_tasa = traslado.attrib.get('TasaOCuota', '')
                        impuesto_tipo = traslado.attrib.get('Impuesto', '')
                        importe_base = float(traslado.attrib.get('Base', 0))
                        importe_traslado = float(traslado.attrib.get('Importe', 0))
                        
                        if impuesto_tipo == '002':
                            if impuesto_tasa == '0.000000':
                                base_0_concepto += importe_base
                            elif impuesto_tasa in ["0.160000", "0.16"]:
                                base_16_concepto += importe_base
                                iva_traslado += importe_traslado

                    retenciones = impuestos.findall('.//cfdi:Retencion', namespace)
                    for retencion in retenciones:
                        impuesto = retencion.attrib.get('Impuesto', '')
                        importe_retencion = float(retencion.attrib.get('Importe', 0))
                        if impuesto == '002':
                            retencion_iva += importe_retencion
                        elif impuesto == '001':
                            retencion_isr += importe_retencion

                complemento = root.find('.//cfdi:Complemento', namespace)
                if complemento is not None:
                    impuestos_locales = complemento.find('.//implocal:ImpuestosLocales', implocal_namespace)
                    if impuestos_locales is not None:
                        traslados_loc = impuestos_locales.findall('.//implocal:TrasladosLocales', implocal_namespace)
                        for tras_loc in traslados_loc:
                            if 'ISH' in tras_loc.attrib.get('ImpLocTrasladado', '').upper():
                                ish_importe += float(tras_loc.attrib.get('Importe', 0))
                
                total_importe = subtotal - descuento + iva_traslado - retencion_iva - retencion_isr + ish_importe

                # Determinar tipo_gasto basado en el nombre de la obra
                obra_nombre = row_data.get('obra', '')
                if ' / Servicios' in obra_nombre:
                    tipo_gasto = 'SERVICIO'
                elif ' / Garantías' in obra_nombre:
                    tipo_gasto = 'GARANTIA'
                else:
                    tipo_gasto = 'COSTO DIRECTO'

                # Mapear desde row_data a las columnas del DataFrame final
                concepto_data = {
                    'obra': obra_nombre,
                    'tipo_gasto': tipo_gasto,
                    'cuenta_gasto': row_data.get('cuenta_gasto', ''), # Asumiendo mapeo
                    'proveedor': row_data.get('proveedor', ''),
                    'residente': row_data.get('residente', ''),
                    'folio': row_data.get('folio', ''), 
                    'estatus': row_data.get('estatus', ''),
                    'fecha_factura': row_data.get('fecha_factura', ''),
                    'fecha_recepcion': row_data.get('fecha_recepcion', ''),
                    'fecha_pagada': row_data.get('fecha_pagada', ''),
                    'fecha_autorizacion': row_data.get('fecha_autorizacion', ''),
                    'clave_producto': clave_prod_serv,
                    'clave_unidad': clave_unidad,
                    'cantidad': cantidad,
                    'descripcion': descripcion,
                    'unidad': unidad,
                    'precio_unitario': valor_unitario,
                    'subtotal': subtotal,
                    'descuento': descuento,
                    'venta_tasa_0': base_0_concepto,
                    'venta_tasa_16': base_16_concepto,
                    'moneda': root.attrib.get('Moneda', ''),
                    'total_iva': iva_traslado,
                    'retencion_iva': retencion_iva,
                    'retencion_isr': retencion_isr,
                    'total_ish': ish_importe,
                    'total': total_importe,
                    'serie': letras_iniciales,
                    'url_pdf': row_data.get('url_pdf', ''), 
                    'url_oc': row_data.get('url_oc', ''),   
                    'url_rem': row_data.get('url_rem', ''), 
                    'xml_uuid': row_data.get('xml_uuid', '') 
                }
                conceptos.append(concepto_data)

            # Asegurarse que el DataFrame resultante tenga todas las columnas esperadas
            df_result = pd.DataFrame(conceptos)
            # Reordenar y añadir columnas faltantes si es necesario (ej. si no hubo conceptos)
            # Esto es una forma simple, se puede mejorar
            for col in columnas_empty:
                 if col not in df_result.columns:
                     df_result[col] = None # o pd.NA o 0 según el tipo esperado
            # Reordenar columnas para que coincidan con columnas_empty puede ser útil
            # df_result = df_result.reindex(columns=columnas_empty)
            # Simplificado: devolver las columnas que se generaron
            return df_result

        except ET.ParseError as e:
            logger.error(f"Error al parsear XML {xml_identifier}: {e}")
            # Devolver DF vacío en caso de error de parseo
            return pd.DataFrame(columns=columnas_empty)
        except Exception as e:
            logger.error(f"Error inesperado procesando XML {xml_identifier}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Devolver DF vacío en caso de otro error
            return pd.DataFrame(columns=columnas_empty)

    # This method might still be useful for processing local files directly
    # If used, it should perhaps not rely on self.errores unless an instance is created
    def procesar_xml(self, xml_file_path, row_data):
        """Procesa un archivo XML local y extrae sus conceptos (Método original adaptado)."""
        xml_uuid = row_data.get('xml_uuid', 'N/A')
        try:
            with open(xml_file_path, 'rb') as f: # Leer en modo binario
                xml_content_bytes = f.read()
            # Llamar al método estático de parseo
            return self._parse_xml_content(xml_content_bytes, row_data, xml_identifier=f"UUID {xml_uuid} (path: {xml_file_path})")

        except FileNotFoundError:
            logger.error(f"Archivo XML no encontrado localmente: {xml_file_path} para UUID {xml_uuid}")
            XMLProcessor.errores.append({'UUID': xml_uuid, 'Error': 'FileNotFoundLocally', 'Path': xml_file_path})
        except Exception as e:
            logger.error(f"Error leyendo o procesando archivo XML {xml_file_path} para UUID {xml_uuid}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            XMLProcessor.errores.append({'UUID': xml_uuid, 'Error': str(e), 'Path': xml_file_path}) # Append to class variable if kept
        
        # Retornar DataFrame vacío en caso de error antes del parseo
        columnas_empty_original = [ # Re-definir por si acaso, aunque idealmente estaría como constante de clase
            'obra', 'cuenta_gasto', 'proveedor', 'residente', 'folio', 'estatus',
            'fecha_factura', 'fecha_recepcion', 'fecha_pagada', 'fecha_autorizacion',
            'clave_producto', 'clave_unidad', 'cantidad', 'descripcion', 'unidad',
            'precio_unitario', 'subtotal', 'descuento', 'venta_tasa_0', 'venta_tasa_16', 'moneda',
            'total_iva', 'retencion_iva', 'retencion_isr', 'total_ish', 'total', 'serie',
            'url_pdf', 'url_oc', 'url_rem', 'xml_uuid'
        ]
        return pd.DataFrame(columns=columnas_empty_original)

    def procesar_xmls_desde_supabase(self, df):
        """
        Descarga y procesa archivos XML desde Supabase Storage basándose en el DataFrame de entrada.
        NOTA: Esta función se volverá obsoleta si el procesamiento se integra en la descarga.
        """
        pass # Function body removed as it's obsolete and caused circular dependency issues
