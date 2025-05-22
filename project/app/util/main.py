from document_handler import download_to_json
from db_manager import DBManager
import os

def main(cod):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_path = os.path.join(base_dir,"temp_files", "temp_document.json")

    db = DBManager()
    db.openConnection() 
    data = db.searchTable("Documentos", {"COD": cod})
    """
    if data[1] == "":
        download_to_json(data[2])
    else: 
        download_to_json(data[1])
    """
    #download_to_json("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=7e880529-03ca-45cc-b001-fbec307680bb")
    from rag_operations import main as rag_main
    rag_main(cod)

if __name__ == "__main__":
    main("CMAYOR/2022/03Y03/8")