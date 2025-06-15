import threading
import time
from .util.main  import main as ragmain # Ajusta el import según tu estructura
from .util.scrapper import update
from .util.db_manager import DBManager

def periodic_task():
    while True:
        print("Ejecutando búsqueda de licitaciones y actualizando la base de datos...")
        update()

        db = DBManager()
        db.openConnection()

        # 1. Busca todos los documentos con anexo_i = 1
        docs_con_anexo = db.searchAllTable("Documentos", {"anexo_i": 1})

        # 2. Obtén todos los COD de la tabla anexoi
        cods_anexoi = set(row[0] for row in db.searchAllTable("anexoi", {}))  # row[0] debe ser el campo COD
        
        db.closeConnection()

        # 3. Filtra los documentos cuyo COD no está en anexoi
        docs_sin_anexoi = [doc for doc in docs_con_anexo if doc[0] not in cods_anexoi]  # doc[0] debe ser el campo COD

        for cod in docs_sin_anexoi:
            ragmain(cod)

        time.sleep(4000)  # Espera 4000 segundos (aproximadamente 1 hora)

def start_periodic_task():
    thread = threading.Thread(target=periodic_task, daemon=True)
    thread.start()