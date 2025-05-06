# Palmaterra Proveedores Invoice Downloader

## Descripción

Este proyecto automatiza el proceso de descarga y procesamiento de facturas y sus archivos XML asociados desde el portal de proveedores de Palmaterra ([https://palmaterraproveedores.centralinformatica.com/](https://palmaterraproveedores.centralinformatica.com/)). Utiliza Selenium para la autenticación, Requests para interactuar con la API del portal, BeautifulSoup para parsear HTML, y Supabase para almacenar los archivos XML descargados y llevar un registro de las facturas procesadas.

## Características

- Autenticación automática en el portal de proveedores.
- Consulta de facturas disponibles por obra.
- Filtrado de facturas por estado (ej. "Pagada").
- Comparación con registros en Supabase Storage para identificar facturas nuevas.
- Descarga concurrente de archivos XML para facturas nuevas.
- Almacenamiento de archivos XML en Supabase Storage, organizados por fecha.
- Procesamiento del contenido XML para extraer conceptos y detalles fiscales (CFDI 3.3 y 4.0).
- Exportación de listas de facturas (nuevas, pagadas) y conceptos detallados a archivos Excel en la carpeta `~/Downloads`.
- Uso de reintentos (retries) para manejar errores de red.
- Registro (logging) detallado del proceso.

## Configuración

1.  **Clonar el Repositorio (si aplica)**
    ```bash
    # git clone <url-del-repositorio>
    # cd <directorio-del-proyecto>
    ```
    O asegúrate de estar en el directorio del proyecto:
    `/Users/noearredondo/Documents/PALMATERRA | 360/Download_Web`

2.  **Crear y Activar un Entorno Virtual**
    ```bash
    python -m venv .venv
    # En macOS/Linux
    source .venv/bin/activate
    # En Windows
    # .\.venv\Scripts\activate
    ```

3.  **Instalar Dependencias**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar Variables de Entorno**
    Crea un archivo llamado `.env` en la raíz del proyecto con tus credenciales de Supabase:
    ```dotenv
    SUPABASE_URL="TU_SUPABASE_URL"
    SUPABASE_KEY="TU_SUPABASE_KEY"
    ```

5.  **Credenciales del Portal**
    Actualmente, las credenciales de inicio de sesión para el portal de Palmaterra están fijas en el archivo `auth_manager.py` (`FIXED_USERNAME`, `FIXED_PASSWORD`). Modifícalas si es necesario.

## Uso

Ejecuta el script principal desde la terminal:

```bash
python main.py
```

El script realizará el proceso completo y generará archivos Excel con los resultados en tu carpeta de Descargas (`~/Downloads`). Revisa la salida de la consola y los logs para más detalles.

## Módulos Principales

-   `main.py`: Orquesta el flujo completo del proceso.
-   `auth_manager.py`: Gestiona la autenticación en el portal web usando Selenium.
-   `invoice_manager.py`: Maneja la consulta, filtrado y procesamiento de datos de facturas.
-   `xml_downloader.py`: Descarga y almacena archivos XML en Supabase, e invoca al procesador.
-   `xml_processor.py`: Parsea y extrae información detallada de los archivos XML (CFDI).
-   `supabase_storage_manager.py`: Gestiona la interacción con Supabase Storage para almacenar/recuperar archivos y IDs procesados.

## Dependencias

Las dependencias del proyecto están listadas en el archivo `requirements.txt`.
