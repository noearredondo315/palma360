import uuid
import hashlib
import pandas as pd
import decimal as D
import re, unicodedata
import pathlib

def norm_txt(x):
    """Normaliza texto: quita espacios, pasa a minúsculas y elimina duplicados de espacio."""
    return " ".join(str(x).strip().lower().split()) if pd.notna(x) else ""

def norm_num(x):
    """Normaliza números: convierte a string decimal normalizado con 6 decimales."""
    if pd.isna(x):
        return ""
    d = D.Decimal(str(x)).quantize(D.Decimal("0.000000"))
    return format(d.normalize(), "f")

NAMESPACE = uuid.UUID("123e4567-e89b-12d3-a456-426614174000")

def concept_uuid5(row):
    """
    Genera un UUID v5 único para el concepto usando los campos clave.
    """
    base = "|".join([
        str(row["xml_uuid"]),
        str(row["clave_producto"]),
        norm_txt(row["descripcion"]),
        norm_num(row["cantidad"]),
        norm_num(row["precio_unitario"]),
    ])
    return uuid.uuid5(NAMESPACE, base)


def filtrar_fechas(df):
    """Filtra las facturas pagadas y en proceso de pago.
    
    Args:
        df_global (pd.DataFrame): DataFrame con todas las facturas
        
    Returns:
        pd.DataFrame: DataFrame con facturas filtradas y procesadas
    """
    print(df.info())
    print(df['fecha_factura'].head())
    # Convertir las fechas al formato datetime y luego al formato Supabase
    date_columns = ["fecha_factura", "fecha_recepcion", "fecha_pagada", "fecha_autorizacion"]
    for col in date_columns:
        if col in df.columns:
            # Convertir a datetime
            df[col] = pd.to_datetime(df[col], format="%d/%m/%y %H:%M", errors='coerce')
            # Formatear para Supabase, manejando NaT
            df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else pd.NA)

    return df

def limpiar_uuid_dataframe(
    df: pd.DataFrame,
    col_uuid: str = "xml_uuid",
    log_descartados: str | None = None
) -> pd.DataFrame:
    """
    Devuelve un DataFrame sin filas cuyo UUID sea inválido,
    manteniendo la columna 'xml_uuid' con su nombre original.

    Args
    ----
    df : pd.DataFrame
        DataFrame original que contiene la columna con UUIDs.
    col_uuid : str, opcional (default='xml_uuid')
        Nombre de la columna que almacena los UUID.
    log_descartados : str | None
        Si se indica, guarda en esa ruta un CSV con las filas descartadas.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame sólo con filas cuyo UUID es válido
        y con la misma estructura de columnas que el original.
    """
    EXOTIC = r'[\u2010\u2011\u2012\u2013\u2014\u2212]'   # guiones tipográficos

    def _fix_hyphens(u):
        """Normaliza, sustituye guiones raros y valida."""
        if pd.isna(u):
            return None
        s = unicodedata.normalize("NFKC", str(u).strip())
        s = re.sub(EXOTIC, '-', s)
        if re.fullmatch(r'[0-9A-Fa-f]{32}', s):
            s = f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"
        try:
            return str(uuid.UUID(s))
        except ValueError:
            return None

    # --- Procesamiento -------------------------------------------------
    df_proc = df.copy()
    df_proc["_tmp_raw"] = df_proc[col_uuid]                    # copia de respaldo
    df_proc[col_uuid] = df_proc["_tmp_raw"].apply(_fix_hyphens)

    # Válidos / inválidos
    df_valid   = df_proc[df_proc[col_uuid].notna()].copy()
    df_invalid = df_proc[df_proc[col_uuid].isna()]

    # Log opcional
    if log_descartados and not df_invalid.empty:
        pathlib.Path(log_descartados).parent.mkdir(parents=True, exist_ok=True)
        df_invalid.drop(columns=["_tmp_raw"]).to_csv(log_descartados, index=False)

    # Limpieza final: quita la columna temporal
    df_valid = df_valid.drop(columns=["_tmp_raw"])

    return df_valid