from document_handler import download_to_json
from rag_operations import main as rag_main
from db_manager import DBManager
import os

def main(cod):
    """
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
    download_to_json("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=bef38c3d-cdd4-4f14-85c6-f798c5231334")
    rag_main(cod)


if __name__ == "__main__":
    main("CMAYOR/2022/03Y03/101")
