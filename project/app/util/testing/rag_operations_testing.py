import os
import sys
from rag_evaluator import RAGEvaluator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_operations import main as rag_main
from document_handler import download_to_json

def main(cod):
    # download_to_json("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=2bb03544-9fba-4cf1-bbaf-b6aa42922806")
    # download_to_json("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=82056ca9-2a4c-489b-8002-9b33d10893b1")
    # SIMILAR download_to_json("https://contrataciondelestado.es/FileSystem/servlet/GetDocumentByIdServlet?DocumentIdParam=9IPiROPs9j9MqNJ5X1m3jn%2BofP/zkimJIB2XVB9fqb1eYB%2B3Bkl70y3AoouUnC5i2oRQTEF2ITtAVbwIKOn6nUgSDa96T/xlqHT4rwF9q/QC1/zDIE0Kw/PWNnLS0Z0z&cifrado=QUC1GjXXSiLkydRHJBmbpw%3D%3D")
    # download_to_json("https://contrataciondelestado.es/FileSystem/servlet/GetDocumentByIdServlet?DocumentIdParam=ZSuz4Wha8bBHerlCxaIoljpjsqOGOGs2b7ZZJf2GRpds1e0QEUC4s9mpwg%2Ba/S5xPDTQ7o8Wpp%2BmCnITbQuM2jHE1e5sQRVE7VHlx07c%2BGiB0nvVKRzfe4rpHcnlPhSZ&cifrado=QUC1GjXXSiLkydRHJBmbpw%3D%3D")
    #download_to_json("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=dacb4122-403f-4e9a-b7f9-980060862eda")
    download_to_json("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=0228702d-bab1-4757-8402-1454d0ba3cc2")
    rag_main(cod)
    # RAGEvaluator


if __name__ == "__main__":
    main("CMAYOR/2023/03Y05/19")