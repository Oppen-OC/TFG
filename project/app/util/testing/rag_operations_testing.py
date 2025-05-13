import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_operations import main as rag_main
from document_handler import download_to_json

def main(cod):
    # download_to_json("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=2bb03544-9fba-4cf1-bbaf-b6aa42922806")
    download_to_json("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=82056ca9-2a4c-489b-8002-9b33d10893b1")
    time.sleep(0.5)
    rag_main(cod)


if __name__ == "__main__":
    main("C78/22G4384/22")