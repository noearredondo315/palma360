import os
import json
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
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

def obtener_obras_y_residentes(username=None, password=None, reporter=None):
    """
    Función que utiliza Selenium y BeautifulSoup para iniciar sesión en el portal
    y obtener la lista de obras y residentes disponibles.
    
    Args:
        username: Nombre de usuario para el login (opcional, usa FIXED_USERNAME si es None)
        password: Contraseña para el login (opcional, usa FIXED_PASSWORD si es None)
        reporter: Objeto opcional para reportar progreso (debe implementar report_progress y report_error)
        
    Returns:
        Diccionario con {'obras': [...], 'residentes': [...], 'cookies': {...}}
    """
    # Usar credentials fijas si no se proporcionan
    username = username or FIXED_USERNAME
    password = password or FIXED_PASSWORD
    
    # Usar reporter por defecto si no se proporciona
    if reporter is None:
        reporter = ProgressReporter()
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    obras = []
    residentes = []
    cookies = {}

    try:
        # Paso 1: Ingresar a la web
        driver.get("https://palmaterraproveedores.centralinformatica.com/")
        reporter.report_progress(1, "Ingresando a la web...")

        # Paso 2: Ingresando usuario y contraseña
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Usuario..']")))
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Contraseña..']")))
        username_input = driver.find_element(By.XPATH, "//input[@placeholder='Usuario..']") 
        password_input = driver.find_element(By.XPATH, "//input[@placeholder='Contraseña..']") 
        username_input.send_keys(username)
        password_input.send_keys(password)

        login_button = driver.find_element(By.ID, "Button1")
        login_button.click()
        reporter.report_progress(2, "Verificando credenciales...")

        # Verificar si el inicio de sesión fue exitoso
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "liConsulta")))
        except Exception:
            # Si no aparece el elemento esperado, asumir que las credenciales son incorrectas
            reporter.report_error("Usuario o contraseña incorrectos. Verifique sus credenciales.")
            driver.quit()
            return {'obras': obras, 'residentes': residentes, 'cookies': cookies}

        # Paso 3: Entrando a la sección de consulta
        consulta_button = driver.find_element(By.ID, "liConsulta")
        consulta_button.click()
        reporter.report_progress(3, "Entrando a la sección de consulta...")

        # Paso 4: Obteniendo residentes y obras
        WebDriverWait(driver, 5).until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#txtObras option")) > 1)
        WebDriverWait(driver, 5).until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#txtResidente option")) > 1)
        page_html = driver.page_source
        soup = BeautifulSoup(page_html, "html.parser")

        obras_select = soup.find("select", {"id": "txtObras"})
        if obras_select:
            obras = [{"value": option.get("value", "").strip(), "name": option.text.strip()} for option in
                     obras_select.find_all("option") if option.get("value", "").strip()]

        residentes_select = soup.find("select", {"id": "txtResidente"})
        if residentes_select:
            residentes = [{"value": option.get("value", "").strip(), "name": option.text.strip()} for option in
                          residentes_select.find_all("option") if option.get("value", "").strip()]

        # Emitir progreso al completar
        reporter.report_progress(4, "Obteniendo residentes y obras...")

        cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}

    except Exception as e:
        reporter.report_error(f"Ocurrió un error: {str(e)}")
    finally:
        driver.quit()

    return {'obras': obras, 'residentes': residentes, 'cookies': cookies}

class AuthManager:
    """
    Clase simplificada para autenticación en entorno web sin dependencias de UI.
    """
    def __init__(self):
        self.reporter = ProgressReporter()
    
    def obtener_credenciales(self):
        """Retorna las credenciales fijas para el entorno de prueba."""
        return FIXED_USERNAME, FIXED_PASSWORD
    
    def iniciar_sesion(self):
        """
        Inicia el proceso de inicio de sesión y carga de obras y residentes.
        Retorna directamente el resultado sin usar hilos ni componentes UI.
        """
        username, password = self.obtener_credenciales()
        logger.info(f"Iniciando sesión con usuario: {username}")
        
        return obtener_obras_y_residentes(username, password, self.reporter)


