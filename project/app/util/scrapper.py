import asyncio
from playwright.async_api import async_playwright
import json
import logging
import os
from db_manager import DBManager

# Configuración del logger específico para el scrapper
base_dir = os.path.dirname(os.path.abspath(__file__))  # Ruta base del archivo actual
logs_dir = os.path.join(base_dir, 'logs')  # Ruta al directorio de logs

# Crear el directorio de logs si no existe
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Ruta del archivo de log
log_file_path = os.path.join(logs_dir, 'scrapper.log')

scrapper_logger = logging.getLogger("scrapper_logger")
scrapper_logger.setLevel(logging.INFO)

# Configurar el manejador de archivo para escribir logs en el directorio de logs
scrapper_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
scrapper_handler.setLevel(logging.INFO)

scrapper_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
scrapper_handler.setFormatter(scrapper_formatter)
scrapper_logger.addHandler(scrapper_handler)
scrapper_logger.propagate = False

# Verifica que el procedimiento de la licitacion sea abierto o abierto simplificado 
async def checkProcedimiento(db, id):
    tabla = "Licitaciones_test"
    fila = await asyncio.to_thread(db.searchTable, tabla, {"COD": id})
    if fila:
        if fila[0][13].lower() == "abierto simplificado" or fila[0][13].lower() == "abierto":
            return True
        else: 
            scrapper_logger.warning(f"Expediente sin procedimiento abierto: {fila[0][0]}")
    else:
        scrapper_logger.error(f"No se encontró el expediente {id} en la tabla de {tabla}")

async def safe_goto(page, url, max_retries=3, timeout=45000):
    retries = 0
    while retries < max_retries:
        try:
            await page.goto(url, timeout=timeout)
            scrapper_logger.info(f"Accedido correctamente al URL: {url}")
            return True  # Éxito
        except Exception as e:
            retries += 1
            scrapper_logger.warning(f"Error al acceder al URL {url}. Intento {retries} de {max_retries}")
            await asyncio.sleep(2)  # Espera antes de reintentar

    return False  # Fallo después de los reintentos

# Accede a link, rellena el formulario y devuelve la lista de expedientes encontrados.
async def poblar_db():

    strings_path = os.path.join(os.path.join(base_dir, 'JSON'), 'Strings.json')
    with open(strings_path, 'r', encoding='utf-8') as f:
        strings = json.load(f)
        
    strings = strings["Formulario"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            if not await safe_goto(page, strings["Pagina"]):
                raise asyncio.TimeoutError()

            await page.click(strings["Licitaciones"], timeout=60000)
            await page.fill(strings["FechaMin"], "01-01-2020")
            await page.fill(strings["FechaMax"], "01-01-2024")
            await page.select_option(strings["Presentacion"], "Electrónica")
            await page.select_option(strings["Tipo"], "Obras")
            await page.select_option(strings["Pais"], "España")
            await page.click(strings["Nuts"])
            await page.select_option(strings["Valencia"], "ES52 Comunitat Valenciana")
            await page.click(strings["NutsClose"])
            await page.click(strings["BusquedaAvanzada"])
            await page.click(strings["Organizacion"])
            await page.click('#maceoArbol > div.tafelTree_root > div:nth-child(3) img.tafelTreeopenable')
            await page.click('#tafelTree_maceoArbol_id_16')

            # Build the full selector for the button inside the container
            button_selector = r'#viewns_Z7_AVEQAI930OBRD02JPMTPG21004\:form1\:botonAnadirMostrarPopUpArbolEO'

            # Click the button
            await page.click(button_selector)
            await page.click(strings["Buscar"], timeout = 60000)
            await page.wait_for_load_state("load")

        except Exception as e:
            scrapper_logger.error("No se pudo rellenar el formulario", e)
            await browser.close()
            return None

        scrapper_logger.info("Formulario rellenado correctamente.")

        i = 0
        pageErrors = []
        db = DBManager()
        db.openConnection()
        # Limpia previamente la tabla de codigos
        await asyncio.to_thread(db.truncateTable, "Codigos_test")
        
        while True:
            try:
                i += 1
                # Espera a que el selector esté disponible
                await page.wait_for_selector("#myTablaBusquedaCustom tr", timeout=60000)
                # Busca la tabla
                tabla = await page.query_selector("#myTablaBusquedaCustom")

                # Encuentra todas las filas de la tabla
                filas = await tabla.query_selector_all("tbody tr")

               # Parsear las filas y celdas
                for fila in filas:
                    # Extrae los codigos de cada expediente
                    celdas = await fila.query_selector_all("td")
                    datos_fila = [await celda.inner_text() for celda in celdas]
                    expediente = datos_fila[0].split("\n")[0]

                    # Extraer href de los enlaces dentro de las celdas
                    enlaces = await fila.query_selector_all("a")
                    href = [await enlace.get_attribute("href") for enlace in enlaces][1]
                    await asyncio.to_thread(db.insertDb, "Codigos_test", (expediente, href))
                    scrapper_logger.info(f"Expediente: {expediente}, Href: {href}")

                # Va a la siguiente página
                next_page_button = await page.query_selector(strings["SiguientePag"])
                if next_page_button:
                    await next_page_button.click(timeout = 100000)
                    await page.wait_for_load_state("load")
                else:
                    db.closeConnection()
                    break

            except Exception as e:
                scrapper_logger.error(f"Error obteniendo expedientes pagina {i}: {e}")
                pageErrors.append(i)
                
        scrapper_logger.info("Buscador cerrado corrrectamente")
        await browser.close()

async def scratchAnexoParallel():
    """Ejecuta varios navegadores en paralelo para procesar expedientes."""
    scrapper_logger.info("Iniciando scrapper de Documentos en paralelo.")
    with open('code/JSON/Strings.json', 'r', encoding='utf-8') as file:
        strings = json.load(file)["Anexo"]

    db = DBManager()
    db.openConnection()

    await asyncio.to_thread(db.truncateTable, "Documentos")

    total_expedientes = db.getTableSize("Codigos_test")
    num_browsers = 4  # Número de navegadores paralelos
    expedientes_por_browser = total_expedientes // num_browsers

    # Crear tareas para cada navegador
    tasks = []
    for i in range(num_browsers):
        start = i * expedientes_por_browser + 1
        end = start + expedientes_por_browser
        if i == num_browsers - 1:  # Asegurar que el último navegador procese el resto
            end = total_expedientes + 1
        tasks.append(process_subgroup(start, end, db, strings))

    # Ejecutar todas las tareas en paralelo
    await asyncio.gather(*tasks)

    db.closeConnection()

async def process_subgroup(start, end, db, strings):
    """Procesa un subconjunto de expedientes en un navegador."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        page.set_default_timeout(120000)

        for j in range(start, end):
            cond1 = True    # Indica si se encontró o no el anexo 1
            cond2 = False   # Indica si se encontró o no el modificado
            hrefs = []      # Guarda los url de los posibles anexos encontrados
            expediente = db.fetchData("Codigos_test", j)  # Obtiene el expediente de la base de datos
            if not expediente:
                continue
            url = expediente[1]
            cod = expediente[0]
            lista = (cod, "N/A", "N/A", "N/A", 0, 0, 0)

            try:
                if not await safe_goto(page, url):
                    raise Exception(f"No se pudo acceder al expediente {cod} con url: {url}")

                # Espera a que el selector esté disponible
                await page.wait_for_selector(f"#{strings['Tabla']} tbody tr", timeout=60000)

                # Busca la tabla
                tabla = await page.query_selector(f"#{strings['Tabla']}")
                filas = await tabla.query_selector_all("tbody tr")

                # Procesa las filas de la tabla
                for fila in filas:
                    celdas = await fila.query_selector_all("td div")
                    tiene_id = await fila.get_attribute("id") is not None
                    if not tiene_id:
                        for celda in celdas:
                            texto_celda = await celda.inner_text()
                            if "Pliego" in texto_celda:
                                cond1 = False
                                enlaces = await celdas[2].query_selector_all("a")
                                href_aux = await enlaces[0].get_attribute("href")
                                hrefs.append(href_aux)

                            if "Modificado" in texto_celda:
                                cond2 = True

                # Procesa los enlaces encontrados
                for href in hrefs:
                    url_p, url_pliego, titulo, mult, index = await scratchPliego(page, href, False)
                    if url_pliego != "N/A":
                        if url_p != "N/A":
                            # Encontró el anexo I, si encontró un modificado el index lo referencia, en caso contrario se mentiene como 0
                            lista = (cod, url_p[index], url_pliego, titulo[index], 1, mult, cond2)
                        else:
                            # No encontró el anexo I
                            lista = (cod, "N/A", url_pliego, "N/A", 0, 0, cond2)
                    else:
                        raise Exception(f"No se encontró cláusula administrativa en el url: {url}")

                if cond1:
                    raise Exception(f"No se encontró la casilla de Pliego en el url: {url}")

            except Exception as e:
                scrapper_logger.error(e)

            finally:
                try:
                    await asyncio.to_thread(db.insertDb, "Documentos", lista)
                except Exception as e:
                    try:
                        scrapper_logger.error(f"Error al insertar en la base de datos, intentando de nuevo: {e}")
                        db.cnxn.rollback()
                        db.reset_cursor()
                        await asyncio.to_thread(db.insertDb, "Documentos", lista)
                    except Exception as e:
                        scrapper_logger.error(f"Falló de nuevo: {e}")
                        db.cnxn.rollback()
                        db.reset_cursor()

        await browser.close()

# Accede a un expediente concreto y extrae la información
async def scratchUrls():

    scrapper_logger.info("Iniciando scrapper de expedientes.")
    with open('code/JSON/Strings.json', 'r', encoding='utf-8') as f:
        strings = json.load(f)
    
    strings = strings["Expediente"]
    db = DBManager()
    db.openConnection()

    # await asyncio.to_thread(db.truncateTable, "Licitaciones_test")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        for j in range(1, db.getTableSize("Codigos_test")+1):
            expediente = db.fetchData("Codigos_test", j)
            cod = expediente[0]
            url = expediente[1]
            scrapper_logger.info(f"{j}- Accediendo al expediente {cod} con url: {url}")
            if not await safe_goto(page, url):
                scrapper_logger.error(f"No se pudo acceder al expediente {cod} con url: {url}")
                continue

            i = 0
            aux_contents = [""]*19

            for word in strings["Atributos"]:
                element_text = await page.inner_text(f"#{word}")
                element_text = element_text.replace('\n', ' ').replace('\r', ' ').replace(u'\xa0', u' ').strip()
                if not element_text:
                    element_text = "N/A"
                aux_contents[i] = element_text
                i += 1
            
            for word in strings["Informacion"]:
                try:
                    element_text = await page.inner_text(f"#{word}", timeout=1000)
                except:
                    element_text = "N/A"

                element_text = element_text.replace('\n', ' ').replace('\r', ' ').replace(u'\xa0', u' ').strip()
                if not element_text:
                    element_text = "N/A"
                aux_contents[i] = element_text
                i += 1

            aux_contents = [cod.strip()] + aux_contents
            await asyncio.to_thread(db.insertDb, "Licitaciones_test", aux_contents)
            await asyncio.sleep(0.1)  # Ensures sequential execution with a small delay
            scrapper_logger.info(f"Expediente {cod} insertado correctamente en la tabla Licitaciones_test")

    scrapper_logger.info("ScratchURLs finalizado.")
    db.closeConnection()
    await browser.close()

# Busca el anexo I a partir del link del expediente y lo inserta en la tabla de Anexo
# La funcion accede e inserta datos en la base de datos
async def scratchAnexo(max_retries=3):
    scrapper_logger.info("Iniciando scrapper de Documentos.")
    with open('code/JSON/Strings.json', 'r', encoding='utf-8') as file:
        strings = json.load(file)
    
    strings = strings["Anexo"]
    db = DBManager()
    db.openConnection()

    await asyncio.to_thread(db.truncateTable, "Documentos")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        for j in range(1, db.getTableSize("Codigos_test")+1):
            cond1 = True # Indica si se encontró o no el anexo 1
            cond2 = False # Indica si se encontró o no el modificado
            hrefs = []
            expediente = db.fetchData("Codigos_test", j) # Obtiene el expediente de la base de datos
            url = expediente[1]
            cod = expediente[0]
            scrapper_logger.info(f"{j} Accediendo al expediente {cod}")
            lista = (cod, "N/A", "N/A", "N/A", 0, 0, 0)

            retries = 0
            retries_cond = True

            while retries < max_retries and retries_cond:
                try: 
                    # Revisasi es procedimiento abierto
                    if not await checkProcedimiento(db, cod.strip()):
                        raise Exception(f"El expediente {cod} no es de procedimiento abierto")
                    
                    if not await safe_goto(page, url):
                        raise Exception(f"No se pudo acceder al expediente {cod} con url: {url}")
                    
                    # Espera a que el selector esté disponible
                    await page.wait_for_selector(f"#{strings['Tabla']} tbody tr", timeout=60000)

                    # Busca la tabla
                    tabla = await page.query_selector(f"#{strings['Tabla']}")

                    # Encuentra todas las filas de la tabla
                    filas = await tabla.query_selector_all("tbody tr")

                    url_p, titulo, cont, index = "", "", 0, 0

                    # Busca en la columna de Documento Pliego y Modificado
                    # Si encuentra pliego busca en el html por el anexo 1
                    # Si encuentra modificado guarda un booleano en la base de datos
                    for fila in filas:
                        # Extrae las celdas de cada fila
                        celdas = await fila.query_selector_all("td div")
                        tiene_id = await fila.get_attribute("id") is not None # El cuerpo tiene varios elementos, solo nos interesan los que no tienen id
                        if not tiene_id: 
                            for celda in celdas:
                                texto_celda = await celda.inner_text()
                                if "Pliego" in texto_celda:
                                    cond1 = False
                                    scrapper_logger.info(f"Encontrado 'Pliego' en la celda: {texto_celda}")
                                    enlaces = await celdas[2].query_selector_all("a")
                                    href_aux = await enlaces[0].get_attribute("href")
                                    hrefs.append(href_aux)
                                    scrapper_logger.info(f"Enlace del documento de pliegos encontrado: {href_aux}")

                                if "Modificación" in texto_celda:
                                    scrapper_logger.info(f"Modificado encontrado en la celda: {texto_celda}")
                                    cond2 = True

                    for href in hrefs:
                        url_p, url_pliego, titulo, cont, index = await scratchPliego(page, href)
                        if url_pliego != "N/A": 
                            if url_p != "N/A":
                                scrapper_logger.info(f"Se encontraron {cont} anexos I en {href}")
                                lista = (cod, url_p[index], url_pliego, titulo[index], 1, cont > 1, cond2)
                                retries_cond = False
                                    
                            else: 
                                scrapper_logger.info(f"No se encontró anexo I en el pliego {href}")
                                lista = (cod, "N/A", url_pliego,"N/A", 0, 0, cond2)
                                retries_cond = False
                        
                        else:
                            raise Exception(f"No se encontró clausula administrativa en  el url: {url}")

                    if cond1:                            
                        raise Exception(f"No se encontró la casilla de Pliego en el url: {url}")

                except Exception as e:
                    retries += 1
                    if retries < max_retries:
                        scrapper_logger.warning(f"Fallo en scratchAnexo, intento nº {retries}: {e}")
                    else:
                        scrapper_logger.error(f"Máximo de intentos alcanzados en scratchAnexo: {e}")
                    

            await asyncio.to_thread(db.insertDb, "Documentos", lista)
                
    scrapper_logger.info("ScratchAnexo finalizado.")
    db.closeConnection()
    await browser.close()     
            

# Busca dentro de la página del pliego y devuelve el url del Anexo 1 en caso de encontrarlo, en caso contrario de vuelve None
# La función no interactia con la base de datos
async def scratchPliego(page, url, logger_active=True, max_retries=3):
    if logger_active: 
        scrapper_logger.info(f"Accediendo al pliego en el url: {url}")

    # Valor de retorno predeterminado
    result = ("N/A", "N/A", "N/A", False, 0)
    retries = 0

    while retries < max_retries:
        try:
            # Intentar navegar al URL
            if not await safe_goto(page, url):
                raise asyncio.TimeoutError(f"No se pudo acceder al URL: {url}")

            # Esperar a que el selector esté disponible
            await page.wait_for_selector(".boxWithBackground", timeout=60000)

            # Buscar el primer objeto con la clase boxWithBackground
            box = await page.query_selector(".boxWithBackground")

            if box:
                # Buscar todos los enlaces dentro del objeto boxWithBackground
                enlaces = await box.query_selector_all("a")
                enlaces = [enlace for enlace in enlaces if await enlace.get_attribute("title") is None]

                aux_index = 1
                hrefs = []
                href_pliego = ""
                texto_enlaces = []

                for index, enlace in enumerate(enlaces):
                    texto_enlace = await enlace.inner_text()
                    texto_enlace = texto_enlace.lower()
                    href = await enlace.get_attribute("href")

                    if texto_enlace == "pliego cláusulas administrativas":
                        if logger_active: 
                            scrapper_logger.info(f"Pliego cláusulas administrativas: {href}")
                        href_pliego = href
                        continue

                    elif filter(texto_enlace):
                        if logger_active: 
                            scrapper_logger.info(f"Documento aceptado: {texto_enlace}")
                        hrefs.append(href)
                        texto_enlaces.append(texto_enlace)

                        if "Modificado" in texto_enlace:
                            if logger_active: 
                                scrapper_logger.info(f"Modificado encontrado: {texto_enlace}")
                            aux_index = index

                    else:
                        if logger_active: 
                            scrapper_logger.info(f"Documento rechazado: {texto_enlace}")

                if len(hrefs) > 0:
                    if logger_active:
                        scrapper_logger.info(f"Enlace del ANEXO encontrado: {hrefs}")
                    result = (hrefs, href_pliego, texto_enlaces, len(hrefs), aux_index - 1)
                else:
                    if logger_active: 
                        scrapper_logger.warning("No se encontró el documento Anexo")
                    result = ("N/A", href_pliego, texto_enlaces, len(hrefs), aux_index - 1)
            else:
                scrapper_logger.error("No se encontró el objeto boxWithBackground")

            # Si todo fue exitoso, salimos del bucle
            break

        except Exception as e:
            retries += 1
            scrapper_logger.error(f"Error al procesar el pliego en el url {url}, intento {retries} de {max_retries}: {e}")
            if retries == max_retries:
                scrapper_logger.error(f"Falló después de {max_retries} intentos: {e}")

        finally:
            # Volver a la página anterior para intentar de nuevo
            await page.go_back()

    if retries > 1 and retries < max_retries:
        scrapper_logger.info(f"Acceso al pliego {url} exitoso después de {retries} intentos.")  

    return result
    
def filter(text):
    if "anexo" in text:
        wanted = [" i ", "_i ", "_i_", " i_", "-i ", " i-", "-i- ", "_i.", " i.", "-i-.", "-i."]
        unwanted = ["responsable"]
        return any(substring in text for substring in wanted) and not any(substring in text for substring in unwanted)
        
    return "annex1" in text


async def test_scratchPliego(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await scratchPliego(page, url)
    
def main():
    # Carga los códigos de página
    asyncio.run(poblar_db())
    asyncio.run(scratchUrls())
    asyncio.run(scratchAnexo())
    # asyncio.run(scratchAnexoParallel())
    # CONTIENE
    # asyncio.run(test_scratchPliego("https://contrataciondelestado.es/FileSystem/servlet/GetDocumentByIdServlet?DocumentIdParam=upTP%2BWKezQgrGjlOQ5Wk5bOeZz2dJ1QhGjQ8oNEDLzCAUoQffpeZQ1wH4Ex%2BHJUQfmwwb2f3zxQg9DibGAGMj3vcPSj7iRpSFPmejtp0mXs%3D&cifrado=QUC1GjXXSiLkydRHJBmbpw%3D%3D"))
    # NO CONTIENE
    # asyncio.run(test_scratchPliego("https://contrataciondelestado.es/FileSystem/servlet/GetDocumentByIdServlet?DocumentIdParam=dyFvDN%2Bteayy6iS%2BHszmINmGMqoWNrkPFjpwsSr6PeEQPabw/tMxluWDko2dWfmQXQgxSiNLZOYWwkJ89XF81JqMTvbf4/PiZgtq1ONWRWs%3D&cifrado=QUC1GjXXSiLkydRHJBmbpw%3D%3D"))

if __name__ == "__main__":
    main()