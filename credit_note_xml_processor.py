"""credit_note_xml_processor.py

Módulo encargado de procesar los XML correspondientes a Notas de Crédito (NC).
Actualmente reutiliza la lógica del ``XMLProcessor`` para extraer conceptos, pero
se crea como una clase separada para mantener una clara separación de
responsabilidades respecto al procesamiento de facturas.

Si en el futuro las Notas de Crédito requieren reglas especiales de parseo
(p. ej. manejo distinto de impuestos o nodos específicos), este módulo es el
lugar indicado para extender dicha funcionalidad sin afectar el flujo de
facturas.
"""

import uuid
from xml_processor import XMLProcessor
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import re
from datetime import datetime, timezone


class CreditNoteXMLProcessor(XMLProcessor):
    """Procesa archivos XML de Notas de Crédito.

    Por ahora hereda completamente la funcionalidad de ``XMLProcessor``.  Se
    mantiene una clase dedicada para:

    1. Facilitar la extensión futura de reglas particulares para NC.
    2. Permitir un registro de logs independiente si se desea.
    """

    # Sobrescribimos el método de parseo para añadir lógica específica de NC

    logger = logging.getLogger('credit_note_xml_processor')

    @staticmethod
    def _parse_sat_datetime(dt_str: str):
        """Intenta parsear fechas del SAT (YYYY-MM-DDTHH:MM:SS) devolviendo tz-naive datetime.
        Devuelve None si falla."""
        if not dt_str:
            return None
        try:
            # Maneja variantes ISO (con o sin zona)
            return datetime.fromisoformat(dt_str)
        except ValueError:
            # Expresión regular básica para YYYY-MM-DD HH:MM:SS o con "T"
            m = re.match(r'^(\d{4})-(\d{2})-(\d{2})[T\s](\d{2}):(\d{2}):(\d{2})', dt_str)
            if m:
                y, mo, d, hh, mm, ss = map(int, m.groups())
                return datetime(y, mo, d, hh, mm, ss)
        return None

    @staticmethod
    def _parse_xml_content(xml_content_bytes: bytes, row_data: pd.Series, xml_identifier: str = 'N/A'):
        """Parsea XML de Nota de Crédito, añadiendo columnas específicas.

        - uuid_nota_credito: UUID tomado del nodo TimbreFiscalDigital dentro de Complemento.
        - xml_uuid: UUID del nodo CfdiRelacionado (relación con factura original).
        """
        # Primero parseamos con la lógica base para reutilizar todo el procesamiento de conceptos
        df_conceptos = XMLProcessor._parse_xml_content(xml_content_bytes, row_data, xml_identifier)

        # Si el DataFrame viene vacío, igual intentamos extraer los UUID para coherencia
        try:
            root = ET.fromstring(xml_content_bytes)
            version = root.attrib.get('Version', '4.0')
            # Namespaces según versión
            if version.startswith("4."):
                ns_cfdi = 'http://www.sat.gob.mx/cfd/4'
            else:
                ns_cfdi = 'http://www.sat.gob.mx/cfd/3'

            ns = {
                'cfdi': ns_cfdi,
                'tfd': 'http://www.sat.gob.mx/TimbreFiscalDigital'
            }

            # Obtener UUID del TimbreFiscalDigital (nota de crédito)
            tfd_node = root.find('.//tfd:TimbreFiscalDigital', ns)
            uuid_nc = tfd_node.attrib.get('UUID') if tfd_node is not None else None

            # Obtener UUID relacionado (factura origen)
            cfdi_rel = root.find('.//cfdi:CfdiRelacionados/cfdi:CfdiRelacionado', ns)
            uuid_rel = cfdi_rel.attrib.get('UUID') if cfdi_rel is not None else None

            # Extraer Folio del Comprobante (folio propio de la NC)
            folio_nc = root.attrib.get('Folio')

            # Extraer FechaTimbrado para sobrescribir fecha_factura
            fecha_timbrado = tfd_node.attrib.get('FechaTimbrado') if tfd_node is not None else None

            # Si no se encontró CfdiRelacionados, intentar derivar de xml_uuid original (puede tener sufijo _NC)
            if not uuid_rel:
                original_uuid_value = str(df_conceptos['xml_uuid'].iloc[0]) if not df_conceptos.empty else ''
                cleaned_uuid = re.sub(r'_NC$', '', original_uuid_value, flags=re.IGNORECASE)
                try:
                    uuid_obj = uuid.UUID(cleaned_uuid)
                    uuid_rel = str(uuid_obj)
                    CreditNoteXMLProcessor.logger.debug(f"UUID relacionado derivado exitosamente de nombre de archivo: {uuid_rel}")
                except ValueError:
                    CreditNoteXMLProcessor.logger.warning(
                        f"UUID relacionado inválido '{original_uuid_value}' para {xml_identifier}. Se descarta la nota de crédito."
                    )
                    # Devolver DF vacío para no procesar esta NC
                    return pd.DataFrame(columns=df_conceptos.columns)

            # Agregar/actualizar columnas al DataFrame resultante
            if 'uuid_nota_credito' not in df_conceptos.columns:
                df_conceptos['uuid_nota_credito'] = uuid_nc
            else:
                df_conceptos.loc[:, 'uuid_nota_credito'] = uuid_nc

            if uuid_rel:
                df_conceptos['xml_uuid'] = uuid_rel

            # Sustituir folio y fecha_factura con los valores del XML
            if folio_nc:
                df_conceptos['folio'] = folio_nc
            if fecha_timbrado:


                # Parsear FechaTimbrado con una expresión regular robusta y convertirla a formato UTC
                fecha_dt = CreditNoteXMLProcessor._parse_sat_datetime(fecha_timbrado)



                if fecha_dt is not None:
                    # Asegurar zona horaria UTC
                    if fecha_dt.tzinfo is None:
                        fecha_dt = fecha_dt.replace(tzinfo=timezone.utc)
                    # Formatear para Supabase: 'YYYY-MM-DD HH:MM:SS+00'
                    fecha_formateada = fecha_dt.strftime('%Y-%m-%d %H:%M:%S')
                    df_conceptos['fecha_factura'] = fecha_formateada
                else:
                    CreditNoteXMLProcessor.logger.warning(
                        f"FechaTimbrado '{fecha_timbrado}' no pudo convertirse a datetime para {xml_identifier}"
                    )

            return df_conceptos

        except Exception as e:
            CreditNoteXMLProcessor.logger.error(f"Error extra adicional procesando UUIDs específicos en {xml_identifier}: {e}")
            # En caso de error, devolver el DF base (ya contiene columnas vacías)
            return df_conceptos
