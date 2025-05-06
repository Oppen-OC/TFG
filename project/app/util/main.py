from .document_handler import download_to_json, idDocType
from .rag_operations import main as rag_main

def main(link):
    doc_type = idDocType()
    download_to_json(link, doc_type)
    rag_main()

if __name__ == "__main__":
    main()
