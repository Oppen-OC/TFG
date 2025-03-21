import asyncio
from playwright.async_api import async_playwright
import json
import logging
from db_manager import DBManager
import tracemalloc

tracemalloc.start()

# Configuración del logger específico para el scrapper
scrapper_logger = logging.getLogger("scrapper_logger")
scrapper_logger.setLevel(logging.INFO)
scrapper_handler = logging.FileHandler("scrapper.log", mode='w')
scrapper_handler.setLevel(logging.INFO)
scrapper_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
scrapper_handler.setFormatter(scrapper_formatter)
scrapper_logger.addHandler(scrapper_handler)
scrapper_logger.propagate = False

# Verifica que el procedimiento de la licitacion sea abiertop o abierto simplificado 
def checkProcedimiento(db, index):
    fila = db.fetchData("Licitaciones", index)
    if fila[13].lower() == "abierto simplificado" or fila[13].lower() == "abierto":
        return True
    else: 
        scrapper_logger.warning(f"Expediente sin procedimiento abierto: {fila[13]}")


# Accede a link, rellena el formulario y devuelve la lista de expedientes encontrados.
async def poblar_db():
    with open('code/Strings.json', 'r', encoding='utf-8') as f:
        strings = json.load(f)
        
    strings = strings["Formulario"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto(strings["Pagina"])
            await page.click(strings["Licitaciones"], timeout = 60000)
            await page.fill(strings["FechaMin"], "01-01-2020")
            await page.fill(strings["FechaMax"], "01-01-2024")
            await page.select_option(strings["Presentacion"], "Electrónica")
            await page.select_option(strings["Tipo"], "Obras")
            await page.select_option(strings["Pais"], "España")
            await page.click(strings["Nuts"])
            await page.select_option(strings["Valencia"], "ES52 Comunitat Valenciana")
            await page.click(strings["NutsClose"])
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
        # await asyncio.to_thread(db.truncateTable, "Codigos")
        
        while True:
            try:
                i += 1
                # Espera a que el selector esté disponible
                await page.wait_for_selector("#myTablaBusquedaCustom tr")
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
                    await asyncio.to_thread(db.insertDb, "Codigos", (expediente, href))
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

# Accede a un expediente concreto y extrae la información
async def scratchUrls():

    scrapper_logger.info("Iniciando scrapper de expedientes.")
    with open('code/Strings.json', 'r', encoding='utf-8') as f:
        strings = json.load(f)
    
    strings = strings["Expediente"]
    db = DBManager()
    db.openConnection()

    await asyncio.to_thread(db.truncateTable, "Licitaciones")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        for j in range(1, db.getTableSize("Codigos")):
            expediente = db.fetchData("Codigos", j)
            url = expediente[1]
            scrapper_logger.info(f"{j}- Accediendo al expediente {expediente} con url: {url}")
            try: 
                await page.goto(url)
            except Exception as e:
                scrapper_logger.error(f"Error al acceder al url {url}: {e}")
                continue

            i = 0
            aux_contents = [""]*19

            for word in strings["Atributos"]:
                element_text = await page.inner_text(f"#{word}", timeout=30000)
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

            aux_contents = [expediente[0].strip()] + aux_contents
            await asyncio.to_thread(db.insertDb, "Licitaciones", aux_contents)
            scrapper_logger.info(f"Expediente {j} insertado correctamente en la base de datos.")

    db.closeConnection()
    await browser.close()

# Busca el anexo I a partir del link del expediente
async def scratchAnexo():
    scrapper_logger.info("Iniciando scrapper de Anexos.")
    with open('code/Strings.json', 'r', encoding='utf-8') as f:
        strings = json.load(f)
    
    strings = strings["Anexo"]
    db = DBManager()
    db.openConnection()

    # db.comp()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        for j in range(1, db.getTableSize("Codigos")):
            cond = True
            expediente = db.fetchData("Codigos", j)
            url = expediente[1]
            scrapper_logger.info(f"Accediendo al url: {url}")
            if not checkProcedimiento(db, j):
                continue

            try: 
                await page.goto(url)
            except Exception as e:
                scrapper_logger.error(f"Error al acceder al url {url}: {e}")
                continue
            
            # Espera a que el selector esté disponible
            try:
                try:
                    await page.wait_for_selector(f"#{strings['Tabla']} tbody tr")
                except Exception as e:
                    scrapper_logger.error("El documento no cuenta con Anexo")
                # Busca la tabla
                tabla = await page.query_selector(f"#{strings['Tabla']}")

                # Encuentra todas las filas de la tabla
                filas = await tabla.query_selector_all("tbody tr")

                # Parsear las filas y celdas
                for fila in filas:
                    # Extrae las celdas de cada fila
                    celdas = await fila.query_selector_all("td div")
                    for celda in celdas:
                        texto_celda = await celda.inner_text()
                        if "Pliego" in texto_celda:
                            cond = False
                            scrapper_logger.info(f"Encontrado 'Pliego' en la celda: {texto_celda}")
                            enlaces = await celdas[2].query_selector_all("a")
                            href = await enlaces[0].get_attribute("href")
                            scrapper_logger.info(f"Enlace del documento de pliegos encontrado: {href}")
                            await scratchPliego(page, href)
                            break
                    if not cond: break
                if cond:
                    scrapper_logger.error(f"No se encontró la casilla de Pliego en el url: {url}")
            except Exception as e:
                scrapper_logger.error(e)
                continue

    db.closeConnection()
    await browser.close()     
            
    
async def scratchPliego(page, url):
    scrapper_logger.info(f"Accediendo al pliego en el url: {url}")
    await page.goto(url)
    print()
    print(url)

    try:
        await page.goto(url)
        # Espera a que el selector esté disponible
        await page.wait_for_selector(".boxWithBackground")
        
        # Busca el primer objeto con la clase boxWithBackground
        box = await page.query_selector(".boxWithBackground")
        
        if box:
            # Busca todos los enlaces dentro del primer objeto boxWithBackground
            enlaces = await box.query_selector_all("a")
            for enlace in enlaces:
                texto_enlace = await enlace.inner_text()
                texto_enlace = texto_enlace.lower()
                if "anexo" in texto_enlace or "annex1" in texto_enlace:
                    print("Anexo encontrado")
                    if contains_unwanted_substrings(texto_enlace):
                        print("Bien", texto_enlace)
                        href = await enlace.get_attribute("href")
                        scrapper_logger.info(f"Enlace del ANEXO encontrado: {href}")
                        return href
                else: print("Mal", texto_enlace)
            scrapper_logger.warning("No se encontró el documento Anexo")
        else:
            scrapper_logger.error("No se encontró el objeto boxWithBackground")
    except Exception as e:
        scrapper_logger.error(f"Error al procesar el pliego en el url {url}: {e}")
    

def contains_unwanted_substrings(text):
    unwanted_substrings = ["ii", "iii", "iv", "control", "seguridad", "mejoras"]
    return not any(substring in text for substring in unwanted_substrings)


async def test_scratchPliego(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await scratchPliego(page, url)
    
def main():
    # Carga los códigos de página

    # asyncio.run(poblar_db())
    # asyncio.run(scratchUrls())
    asyncio.run(scratchAnexo())
    # CONTIENE
    # asyncio.run(test_scratchPliego("https://contrataciondelestado.es/FileSystem/servlet/GetDocumentByIdServlet?DocumentIdParam=j5PwB4/qj%2Bdv2IbmNoZjSoXEiHgQGDlEd7C7/oOJrUwVJ73aAB1cKVAu0BTM/cI09rkIetxSbxO6dJYqBD1lepk2iAOGI%2BuU5VaqQsax2Jo%3D&cifrado=QUC1GjXXSiLkydRHJBmbpw%3D%3D"))
    # NO CONTIENE
    # asyncio.run(test_scratchPliego("https://contrataciondelestado.es/FileSystem/servlet/GetDocumentByIdServlet?DocumentIdParam=dyFvDN%2Bteayy6iS%2BHszmINmGMqoWNrkPFjpwsSr6PeEQPabw/tMxluWDko2dWfmQXQgxSiNLZOYWwkJ89XF81JqMTvbf4/PiZgtq1ONWRWs%3D&cifrado=QUC1GjXXSiLkydRHJBmbpw%3D%3D"))

if __name__ == "__main__":
    main()