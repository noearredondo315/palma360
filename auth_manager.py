import os
import json
import time
import requests
from bs4 import BeautifulSoup
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('auth_manager')

# Credenciales fijas para el entorno de prueba
FIXED_USERNAME = "p-noebz"
FIXED_PASSWORD = "edoz35"

class ProgressReporter:
    """
    Clase simple para reportar progreso sin dependencias de UI.
    """
    def report_progress(self, step, message):
        """Reporta el progreso de una operación."""
        logger.info(f"Paso {step}: {message}")
    
    def report_error(self, message):
        """Reporta un error en una operación."""
        logger.error(f"Error: {message}")

# URLs para requests
BASE_URL = "https://palmaterraproveedores.centralinformatica.com/"
URL_OBRAS = BASE_URL + "WSUnico.asmx/filtroObra"

def extraer_campos_ocultos(html):
    """
    Extrae los campos ocultos del formulario HTML para el login.
    
    Args:
        html: Contenido HTML de la página
        
    Returns:
        Diccionario con los campos ocultos y sus valores
    """
    soup = BeautifulSoup(html, "html.parser")
    data = {}
    for inp in soup.select("input[type=hidden][name]"):
        data[inp['name']] = inp.get('value', '')
    return data

def obtener_obras_y_residentes(username=None, password=None, reporter=None):
    """
    Función que utiliza requests y BeautifulSoup para iniciar sesión en el portal
    y obtener la lista de obras disponibles.
    
    Args:
        username: Nombre de usuario para el login (opcional, usa FIXED_USERNAME si es None)
        password: Contraseña para el login (opcional, usa FIXED_PASSWORD si es None)
        reporter: Objeto opcional para reportar progreso (debe implementar report_progress y report_error)
        
    Returns:
        Diccionario con {'obras': [...], 'cookies': {...}}
    """
    # Usar credentials fijas si no se proporcionan
    username = username or FIXED_USERNAME
    password = password or FIXED_PASSWORD
    
    # Usar reporter por defecto si no se proporciona
    if reporter is None:
        reporter = ProgressReporter()
    
    try:
        # Crear sesión de requests
        session = requests.Session()
        
        # Paso 1: GET inicial para extraer campos ocultos
        reporter.report_progress(1, "Ingresando a la web...")
        resp = session.get(BASE_URL, timeout=30)
        resp.raise_for_status()
        data = extraer_campos_ocultos(resp.text)
        
        # Paso 2: POST login
        reporter.report_progress(2, "Verificando credenciales...")
        data.update({
            "txtUsuario": username,
            "txtPassword": password,
            "Button1": "Entrar",
        })
        resp = session.post(BASE_URL, data=data, timeout=30)
        resp.raise_for_status()
        
        # Verificar si el inicio de sesión fue exitoso
        if "liConsulta" not in resp.text:
            reporter.report_error("Usuario o contraseña incorrectos. Verifique sus credenciales.")
            return {'obras': [], 'cookies': {}}
        
        # Paso 3: Headers comunes para AJAX
        reporter.report_progress(3, "Consultando obras disponibles...")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-Requested-With": "XMLHttpRequest"
        }
        
        # Paso 4: POST a filtroObra
        obras_resp = session.post(URL_OBRAS, headers=headers, json={}, timeout=30)
        obras = json.loads(obras_resp.json()["d"])
        
        reporter.report_progress(4, "Proceso completado exitosamente.")
        
        return {
            "obras": obras,
            "cookies": session.cookies.get_dict(),
        }
        
    except Exception as e:
        reporter.report_error(f"Ocurrió un error: {str(e)}")
        return {'obras': [], 'cookies': {}}

class AuthManager:
    """
    Clase simplificada para autenticación en entorno web usando requests sin dependencias de UI.
    """
    def __init__(self):
        self.reporter = ProgressReporter()
    
    def obtener_credenciales(self):
        """Retorna las credenciales fijas para el entorno de prueba."""
        return FIXED_USERNAME, FIXED_PASSWORD
    
    def iniciar_sesion(self):
        """
        Inicia el proceso de inicio de sesión y carga de obras.
        Retorna directamente el resultado sin usar hilos ni componentes UI.
        """
        username, password = self.obtener_credenciales()
        logger.info(f"Iniciando sesión con usuario: {username}")
        
        return obtener_obras_y_residentes(username, password, self.reporter)


