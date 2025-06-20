from .document_handler import download_to_json
from .rag_operations import main as rag_main

def main(cod):
   
    db = DBManager()
    db.openConnection() 
    data = db.searchTable("Documentos", {"COD": cod})
    
    if data:
        download_to_json(data[1])
        rag_main(cod)
    else:
        print("No data")
    
    """
    download_to_json(r"https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=0228702d-bab1-4757-8402-1454d0ba3cc2")
    """


if __name__ == "__main__":
    db = DBManager()
    db.openConnection()
    # Buscar todos los documentos con anexoI = 1
    documentos = db.searchAllTable("documentos", {"anexo_I": 1})
    documentos_anexos = db.viewTable("Anexoi")
    codigos = [fila[0] for fila in documentos_anexos]
    # Convertir a lista (por si acaso)
    print(len(documentos))
    documentos = list(documentos)
    for documento in documentos:
        if documento[0] not in codigos:
            print("COD:", documento[0])
            elemento = documento[0] 
            main(elemento)

    db.closeConnection()
