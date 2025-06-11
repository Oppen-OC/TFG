from .document_handler import download_to_json
from .rag_operations import main as rag_main
from .db_manager import DBManager

def main(cod):
   
    db = DBManager()
    db.openConnection() 
    data = db.searchTable("Documentos", {"COD": cod})
    
    if data:
        if data[1] == "":
            download_to_json(data[2])
        else: 
            download_to_json(data[1])
    else:
        print("No data")
    
    """
    download_to_json(r"https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=0228702d-bab1-4757-8402-1454d0ba3cc2")
    """
    rag_main(cod)


if __name__ == "__main__":
    main("CMAYOR/2022/03Y03/8")
