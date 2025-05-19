import os
import json
import sys
import openai
from dotenv import load_dotenv
from tqdm import tqdm 
from pydantic import BaseModel
import random

load_dotenv()

# Agregar el directorio padre al sys.path para importar document_handler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from document_handler import download_to_json  # Importar la función desde document_handler

class QuestionList(BaseModel):
    pregunta_1: str
    pregunta_2: str
    pregunta_3: str
    pregunta_4: str 
    pregunta_5: str
    pregunta_6: str
    pregunta_7: str
    pregunta_8: str
    pregunta_9: str
    pregunta_10: str

def generate_jsonCorpus(input_path, corpus_path, key):
    try:
        # Leer el archivo JSON de entrada
        with open(input_path, 'r', encoding='utf-8') as file:
            document_data = json.load(file)

        # Leer o inicializar el archivo de salida
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as file:
                output_data = json.load(file)
        else:
            # Crear un archivo vacío si no existe
            output_data = {}
            with open(corpus_path, 'w', encoding='utf-8') as file:
                json.dump(output_data, file, indent=4, ensure_ascii=False)

        # Agregar los datos bajo la clave correspondiente
        output_data[key] = document_data

        # Guardar los datos en el archivo JSON de salida
        with open(corpus_path, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, indent=4, ensure_ascii=False)

        print(f"Información guardada exitosamente en {corpus_path} bajo la clave '{key}'.")

    except Exception as e:
        print(f"Error generando corpus: {e}")

def generate_questions_from_chunks(corpus_path, output_path, ollama_url="https://api.poligpt.upv.es", model="llama3.3:70b", questions=10):
    try:
        # Leer el archivo JSON de entrada
        with open(corpus_path, 'r', encoding='utf-8') as file:
            corpus_data = json.load(file)

        # Diccionario para almacenar las preguntas generadas
        questions_data = {}

        # Configuración del cliente
        client = openai.OpenAI(api_key=os.getenv("UPV_API_KEY"), base_url=ollama_url)

        # Iterar sobre cada documento en el corpus con barra de progreso
        for doc_key, pages in tqdm(corpus_data.items(), desc="Procesando documentos", unit="documento"):
            questions_data[doc_key] = []
            index = 0

            # Iterar sobre cada página del documento con barra de progreso
            for page in tqdm(pages, desc=f"Procesando páginas de {doc_key}", unit="página", leave=False):
                # Obtener los chunks de la página
                chunks = page.get("chunks", [])
                for chunk in chunks:
                    # Crear el mensaje del sistema y del usuario
                    system_message = {
                        "role": "system",
                        "content": f"Eres un asistente que genera preguntas basadas en texto en formato JSON. "
                    }
                    chunk_message = {
                        "role": "user",
                        "content": f"""
                        Texto: {chunk}

                        Instrucciones:
                        1. Analiza los temas clave, hechos y conceptos en el texto proporcionado.
                        2. Genera diez preguntas que un usuario podría hacer para encontrar información en este texto.
                        3. Usa lenguaje natural y ocasionalmente incluye errores tipográficos o coloquialismos.
                        4. Asegúrate de que las preguntas estén relacionadas semánticamente con el contenido del texto.
                        5. Devuelve las preguntas en formato JSON.
                        """
                    }

                    # Llamar al cliente para generar preguntas
                    try:
                        completion = client.beta.chat.completions.parse(
                            model=model,
                            messages=[system_message, chunk_message],
                            response_format=QuestionList
                        )

                        # Extraer las preguntas de la respuesta
                        answer = completion.choices[0].message.parsed

                        # Convertir el objeto QuestionList a un diccionario
                        if isinstance(answer, QuestionList):
                            answer_dict = answer.model_dump()  # Convertir a diccionario
                        else:
                            raise ValueError("La respuesta no es del tipo esperado QuestionList")

                        # Agregar las preguntas al diccionario de datos
                        questions_data[doc_key].append({"index": index, "chunk": chunk, "questions": answer_dict})
                        index += 1

                    except Exception as e:
                        print(f"Error al generar preguntas para el chunk: {chunk}. Detalles: {e}")
                        continue

            # Guardar las preguntas generadas para el documento

        # Guardar las preguntas generadas en el archivo de salida
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(questions_data, file, indent=4, ensure_ascii=False)

        print(f"Preguntas generadas exitosamente y guardadas en {output_path}")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

def split_questions_train_val(questions_path, output_path, val_ratio=0.2, train_ratio=0.8, seed=42):
    """
    Divide el archivo questions.json en datos de entrenamiento y validación.
    Puedes especificar el porcentaje de validación y entrenamiento.
    """
    if abs((val_ratio + train_ratio) - 1.0) > 1e-6:
        raise ValueError("La suma de val_ratio y train_ratio debe ser 1.0")

    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data = {}
    val_data = {}

    random.seed(seed)

    for doc_code, questions_list in tqdm(data.items(), desc="Dividiendo documentos", unit="documento"):
        train_data[doc_code] = []
        val_data[doc_code] = []
        for item in tqdm(questions_list, desc=f"Chunks de {doc_code}", unit="chunk", leave=False):
            # item: {"chunk": ..., "questions": {...}}
            questions_dict = item.get("questions", {})
            questions_keys = list(questions_dict.keys())
            random.shuffle(questions_keys)
            total = len(questions_keys)
            train_count = int(total * train_ratio)
            train_questions = {k: questions_dict[k] for k in questions_keys[:train_count]}
            val_questions = {k: questions_dict[k] for k in questions_keys[train_count:]}
            # Añadir a los datos de entrenamiento y validación solo si hay preguntas
            if train_questions:
                train_data[doc_code].append({
                    "chunk": item.get("chunk"),
                    "questions": train_questions
                })
            if val_questions:
                val_data[doc_code].append({
                    "chunk": item.get("chunk"),
                    "questions": val_questions
                })

    with open(os.path.join(output_path, "training.json"), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_path, "validation.json"), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)

    print(f"Archivos generados:\nEntrenamiento: {os.path.join(output_path, 'training.json')}\nValidación: {os.path.join(output_path, 'validation.json')}")


def split_questions_train_val(questions_path, output_dir, doc_code, val_ratio=0.2, train_ratio=0.8, seed=42):
    """
    Exporta las preguntas de un documento en formato plano y las divide en entrenamiento y validación.
    Cada entrada tendrá: {"chunk", "question"}
    """
    if abs((val_ratio + train_ratio) - 1.0) > 1e-6:
        raise ValueError("La suma de val_ratio y train_ratio debe ser 1.0")

    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if doc_code not in data:
        print(f"Documento {doc_code} no encontrado en el archivo.")
        return

    flat_list = []
    for item in data[doc_code]:
        chunk = item.get("chunk")
        questions = item.get("questions", {})
        for q in questions.values():
            flat_list.append({
                "chunk": chunk,
                "question": q
            })

    # Mezclar y dividir en train/val
    random.seed(seed)
    random.shuffle(flat_list)
    total = len(flat_list)
    train_count = int(total * train_ratio)
    train_list = flat_list[:train_count]
    val_list = flat_list[train_count:]

    # Guardar archivos
    train_path = os.path.join(output_dir, f"{doc_code.replace('/', '_')}_train.json")
    val_path = os.path.join(output_dir, f"{doc_code.replace('/', '_')}_val.json")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_list, f, indent=4, ensure_ascii=False)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_list, f, indent=4, ensure_ascii=False)
    print(f"Archivos exportados:\nEntrenamiento: {train_path}\nValidación: {val_path}")

def merge_json_files(input_files, output_file):
    """
    Une varios archivos JSON (listas de diccionarios) en un solo archivo.
    """
    merged = []
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged.extend(data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)
    print(f"Archivo combinado guardado en: {output_file}")

if __name__ == "__main__":
    # Ruta al archivo de salida
    base_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(base_dir, "json_files", "Corpus.json"))
    input_path = os.path.abspath(os.path.join(base_dir, "..", "temp_files", "temp_document.json"))
    questions_path = os.path.abspath(os.path.join(base_dir, "json_files", "questions.json"))
    """
    # Array de tuplas (url, cod)
    documents = [
        ("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=0228702d-bab1-4757-8402-1454d0ba3cc2", "CMAYOR/2023/03Y05/19"),
        ("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=bef38c3d-cdd4-4f14-85c6-f798c5231334", "CMAYOR/2022/03Y03/101"),
        ("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=7e880529-03ca-45cc-b001-fbec307680bb", "CMAYOR/2022/03Y03/8 "),
        ("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=28f038af-faba-4985-bddd-efea11166bfd", "CMAYOR/2022/07Y10/44"),
        ("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=be0cf2e8-0713-42af-9250-6b69865c59b7", "CMAYOR/2023/07Y10/1"),
        ("https://contrataciondelestado.es/wps/wcm/connect/PLACE_es/Site/area/docAccCmpnt?srv=cmpnt&cmpntname=GetDocumentsById&source=library&DocumentIdParam=69c6ec11-3826-4b71-becc-d187156635cf","CMAYOR/2021/07Y10/134"),
    ]

    # Ejecutar download_to_json() para cada (url, cod)
    
    for url, cod in documents:
        download_to_json(url)  # Descargar el documento
        generate_jsonCorpus(input_path, corpus_path, cod)  # Guardar en el JSON

    generate_questions_from_chunks(corpus_path, questions_path)  # Generar preguntas
    """
    train_files = [
        r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\json_files\CMAYOR_2022_03Y03_8 _val.json",
        r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\json_files\CMAYOR_2022_03Y03_101_val.json",
        r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\json_files\CMAYOR_2022_07Y10_44_val.json",
        r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\json_files\CMAYOR_2023_03Y05_19_val.json",
        r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\json_files\CMAYOR_2023_07Y10_1_val.json"
    ]

    merge_json_files(train_files, corpus_path)

    split_questions_train_val(questions_path, base_dir + "/json_files", "CMAYOR/2023/07Y10/1")
