from .document_handler import download_to_json
from .rag_operations import main as rag_main
from .db_manager import DBManager

def main(cod):
    db = DBManager()
    db.openConnection()
    data = db.searchTable("Documentos", {"COD": cod})
    if data[1] == "":
        download_to_json(data[2])
    else: 
        download_to_json(data[1])
    
    rag_main(cod)
