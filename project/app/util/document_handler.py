import json
import requests
from odf.opendocument import load
from odf.text import P
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pdfplumber
import re


def idDocType(title):
    parts = title.split('.')
    return parts[-1] if parts else None

def section_identifier(text, anterior):
    # Expresión regular para detectar letras seguidas de ". " o "APARTADO X\n"
    pattern = r"(?:^|\n)[A-ZÑ]{1}\. |(?:^|\n)APARTADO [A-ZÑ]{1}\n"
    secciones = re.findall(pattern, text)  # Encuentra todas las coincidencias
    ans = [anterior] if anterior else []
    for seccion in secciones:
        seccion = seccion.replace("\n", "")
        if seccion not in ans:
            ans.append(seccion)
    return ans

def spliter(ant, text, size = 512, overlap=64):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size,chunk_overlap=overlap)
    texts = text_splitter.split_text(ant)
    resto = size - len(texts[-1])
    texts[-1] = texts[-1] + text[:resto]
    return texts

def download_to_text(url, file_type):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Ruta base del archivo actual
        output_dir = os.path.join(base_dir, 'temp_files')  # Ruta al directorio de logs

        os.makedirs(output_dir, exist_ok=True)

        # Descargar el documento
        response = requests.get(url)
        response.raise_for_status()
        text = ""

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

        if file_type == 'pdf':
            with pdfplumber.open(temp_doc_path) as pdf:
                for page in pdf.pages:
                    size = (0.1 * page.width, 0.1 * page.height, 0.90 * page.width, 0.90 * page.height)
                    cropped_page = page.within_bbox(size)
                    text += cropped_page.extract_text(layout=True) + "\n"
                    text += f"##PAGINA:{page.page_number}##" + "\n\f\n"

        elif file_type == "zip":
            print("Compatibilidad con .zip aun no implementada")
            return None
        else:
            print(f"Tipo de archivo no soportado: {file_type}")
            return None

        # Guardar el texto en un archivo llamado "temp.txt"
        text_file_path = os.path.join(output_dir, "temp_document.txt")
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        # Eliminar el archivo temporal
        # os.remove(temp_doc_path)

        return text_file_path

    except requests.RequestException as e:
        print(f"Error al descargar el documento: {e}")
        return None
    except Exception as e:
        print(f"Error al convertir el documento a texto: {e}")
        return None


def download_to_json(url, file_type):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Base directory
        output_dir = os.path.join(base_dir, 'temp_files')  # Output directory

        os.makedirs(output_dir, exist_ok=True)

        # Download the document
        response = requests.get(url)
        response.raise_for_status()

        # Save the document temporarily
        temp_doc_path = os.path.join(output_dir, f"temp_document.{file_type}")
        with open(temp_doc_path, 'wb') as temp_doc_file:
            temp_doc_file.write(response.content)

        # Extract text and metadata
        data = []  # List to store text and metadata
        if file_type == 'odt':
            doc = load(temp_doc_path)
            paragraphs = [str(para) for para in doc.getElementsByType(P)]
            for i, para in enumerate(paragraphs):
                data.append({"page": i + 1, "text": para})

        elif file_type == 'docx':
            doc = Document(temp_doc_path)
            paragraphs = [para.text for para in doc.paragraphs]
            for i, para in enumerate(paragraphs):
                data.append({"page": i + 1, "text": para})

        elif file_type == 'pdf':
            with pdfplumber.open(temp_doc_path) as pdf:
                ant_text = ""
                ant_table = None
                ant_sec = None
                for i, page in enumerate(pdf.pages):
                    size = (0.1 * page.width, 0.1 * page.height, 0.95 * page.width, 0.85 * page.height)
                    cropped_page = page.crop(size)
                    text = cropped_page.extract_text().replace("☒", "[CORRECTO]").replace("Nº", "número").replace("☐", "[INCORRECTO]")  # Extract text from the cropped page

                    # Extract sections based on page number
                    if data:
                        sec = section_identifier(text, data[-1]["sections"][-1])
                    else:
                        sec = section_identifier(text, "None")

                    if ant_text != "":
                        chunks = spliter(ant_text, text)  # Split text into chunks                   
                        data.append({"page": page.page_number -1, "chunks": chunks, "text": ant_text, "tables": ant_table,"sections": ant_sec})    
                        ant_text = text       
                        ant_table = cropped_page.extract_table()   
                        ant_sec = section_identifier(text, data[-1]["sections"][-1])
                    else: 
                        ant_text = text  
                        ant_table = cropped_page.extract_table()
                        ant_sec = sec
                    
                    if i == len(pdf.pages) - 1:
                        chunks = spliter(ant_text, "")
                        data.append({"page": page.page_number , "chunks": chunks , "text": ant_text, "tables": ant_table,"sections": ant_sec})
                    


        elif file_type == "zip":
            print("Compatibilidad con .zip aun no implementada")
            return None
        else:
            print(f"Tipo de archivo no soportado: {file_type}")
            return None

        # Save the extracted data to a JSON file
        json_file_path = os.path.join(output_dir, "temp_document.json")
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

        print(f"JSON file saved to: {json_file_path}")
        return json_file_path

    except requests.RequestException as e:
        print(f"Error al descargar el documento: {e}")
        return None
    except Exception as e:
        print(f"Error al convertir el documento a JSON: {e}")
        return None


def main():
    
    odt = "https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=a9af0a63-71e1-4f05-b71c-55e7a7e1c107"
    pdf = "https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=ed28a294-d582-4936-84d5-3364242e1f8b"
    docx = "https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=0cf6ca6f-add4-4d37-a48d-9e33e72e2adb"
    #text_file_path_odt = download_and_convert_to_text(odt, "odt")
    # print("odt", text_file_path_odt)
    download_to_json(pdf, "pdf")
    # print("pdf", text_file_path_pdf)
    # text_file_path_docx = download_and_convert_to_text(docx, "docx")
    #print("docx", text_file_path_docx)

if __name__ == "__main__":
    main()
