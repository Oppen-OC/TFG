import threading
import time
from .util.main import main as ragmain
from .util.scrapper import update
from .util.db_manager import DBManager
from .util.mail_sender import enviar_correo_licitacion  # Ajusta el nombre de la función si es distinto

def cumple_condiciones(documento, condiciones):
    """
    documento: dict con los campos del documento
    condiciones: lista de dicts [{'campo': ..., 'valor': ...}, ...]
    """
    for cond in condiciones:
        campo = cond['campo']
        valor = cond['valor']
        # Compara como string, puedes adaptar según el tipo de campo
        if str(documento.get(campo, '')).lower() not in str(valor).lower():
            return False
    return True

def periodic_task(cond=False):
    while False:
        print("Ejecutando búsqueda de licitaciones y actualizando la base de datos...")
        update()
        if cond:
            break

        db = DBManager()
        db.openConnection()

        # 1. Busca todos los documentos con anexo_i = 1
        docs_con_anexo = db.searchAllTable("Documentos", {"anexo_i": 1})

        # 2. Obtén todos los COD de la tabla anexoi
        cods_anexoi = set(row[0] for row in db.searchAllTable("anexoi", {}))  # row[0] debe ser el campo COD

        # 3. Filtra los documentos cuyo COD no está en anexoi
        docs_sin_anexoi = [doc for doc in docs_con_anexo if doc[0] not in cods_anexoi]  # doc[0] debe ser el campo COD

        # 4. Para cada usuario con alertas
        usuarios = set(row[1] for row in db.searchAllTable("alertas", {}))  # row[1] debe ser el campo user
        for user in usuarios:
            # Busca el correo del usuario (ajusta según tu modelo de usuarios)
            # Aquí se asume que tienes una tabla usuarios con campos user y email
            user_data = db.searchAllTable("usuarios", {"username": user})
            if not user_data or not user_data[0][2]:  # Ajusta el índice del email si es necesario
                continue
            email = user_data[0][2]

            # Obtén las alertas del usuario
            alertas = db.searchAllTable("alertas", {"user": user})
            for alerta in alertas:
                # alerta[4] debe ser el campo condiciones (ajusta si es otro índice)
                import json
                try:
                    condiciones = json.loads(alerta[4])
                except Exception:
                    continue

                for doc in docs_sin_anexoi:
                    # Convierte doc a dict con nombres de campo
                    columnas = db.get_column_names("Documentos")  # Implementa este método si no lo tienes
                    doc_dict = dict(zip(columnas, doc))
                    if cumple_condiciones(doc_dict, condiciones):
                        # Envía el correo
                        enviar_correo_licitacion(doc_dict['COD'], email)

        db.closeConnection()
        time.sleep(4000)  # Espera 4000 segundos (aproximadamente 1 hora)

def start_periodic_task():
    thread = threading.Thread(target=periodic_task, args=(True,), daemon=True)
    thread.start()