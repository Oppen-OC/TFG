import requests
from odf.opendocument import load
from odf.text import P
from docx import Document
import os
import PyPDF2
import pdfplumber


def idDocType(title):
    parts = title.split('.')
    return parts[-1] if parts else None


def download_and_convert_to_text(url, file_type, name="", output_dir=None):
    try:
        # Usar el directorio de logs si no se proporciona output_dir
        if output_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))  # Ruta base del archivo actual
            output_dir = os.path.join(base_dir, 'temp_files')  # Ruta al directorio de logs

        # Crear el directorio de logs si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Descargar el documento
        response = requests.get(url)
        response.raise_for_status()

        # Guardar el documento temporalmente
        temp_doc_path = os.path.join(output_dir, f"temp_document.{file_type}")
        with open(temp_doc_path, 'wb') as temp_doc_file:
            temp_doc_file.write(response.content)

        # Convertir el documento a texto
        if file_type == 'odt':
            doc = load(temp_doc_path)
            text = "###\n".join([str(para) for para in doc.getElementsByType(P)])

        elif file_type == 'docx':
            doc = Document(temp_doc_path)
            text = "###\n".join([para.text for para in doc.paragraphs])

        elif file_type == 'pdf':
            text = ""
            try:
                with pdfplumber.open(temp_doc_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + "\f\n"
            except Exception as e:
                print(f"Error al leer el archivo PDF: {e}")
                return None
        elif file_type == "zip":
            print("Funcionalidad aun no implementada")
            return None
        else:
            print(f"Tipo de archivo no soportado: {file_type}")
            return None

        # Guardar el texto en un archivo llamado "temp.txt"
        text_file_path = os.path.join(output_dir, f"{name}temp.txt")
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        # Eliminar el archivo temporal
        os.remove(temp_doc_path)

        return text_file_path

    except requests.RequestException as e:
        print(f"Error al descargar el documento: {e}")
        return None
    except Exception as e:
        print(f"Error al convertir el documento a texto: {e}")
        return None
    

def main():
    
    odt = "https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=a9af0a63-71e1-4f05-b71c-55e7a7e1c107"
    pdf = "https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=ed28a294-d582-4936-84d5-3364242e1f8b"
    docx = "https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=0cf6ca6f-add4-4d37-a48d-9e33e72e2adb"
    #text_file_path_odt = download_and_convert_to_text(odt, "odt")
    # print("odt", text_file_path_odt)
    text_file_path_pdf = download_and_convert_to_text(pdf, "pdf")
    # print("pdf", text_file_path_pdf)
    # text_file_path_docx = download_and_convert_to_text(docx, "docx")
    #print("docx", text_file_path_docx)

if __name__ == "__main__":
    main()
