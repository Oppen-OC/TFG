from dotenv import load_dotenv
import openai
import os
import chromadb
import json
import logging
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel
from typing import List
import re
from .db_manager import DBManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from nltk.stem import WordNetLemmatizer


load_dotenv()

class Prompt(BaseModel):
    prompt_key: str
    prompt: str
    response: str
    score: float
    index: List[int]

class RAGResult(BaseModel):
    prompts: List[Prompt]

class RAG:
    # Configuración del logger específico para el RAG y archivos temporales
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Ruta base del archivo actual
    logs_dir = os.path.join(base_dir, 'logs')  # Ruta al directorio de logs
    temp_files_dir = os.path.join(base_dir, 'temp_files')

    # Ruta del archivo de log
    log_file_path = os.path.join(logs_dir, 'rag.log')
    rag_logger = logging.getLogger("rag_logger")
    rag_logger.setLevel(logging.INFO)
    rag_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    rag_handler.setLevel(logging.INFO)
    rag_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    rag_handler.setFormatter(rag_formatter)
    rag_logger.addHandler(rag_handler)
    rag_logger.propagate = False

    # Archivo JSON con los chunks procesados
    file_path = os.path.join(base_dir, 'temp_files', 'temp_document.json')  

    with open(file_path, 'r', encoding='utf-8') as f:
        file = json.load(f)
        
    # Archivo JSON donde se guardan las instrucciones para el RAG
    jsonFile = os.path.join(base_dir, 'JSON', 'Strings.json') 

    with open(jsonFile, 'r', encoding='utf-8') as f:
        strings = json.load(f)

    def __init__(self, 
                 embedding_model = "nomic-embed-text",
                 base_url = "https://api.poligpt.upv.es",
                 api_key = os.getenv("UPV_API_KEY"),
                 llm_model = "llama3.3:70b",
                 min_score = 0.60
                 ):
        
        # Cliente de OpenAI
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        strings = RAG.strings

        self.prompts = strings["Prompts"]["Ajuntament"]  
        self.semantics = strings["Prompts"]["Semantic"] 
        self.format = strings["Prompts"]["Ajuntament_formato"]
        self.sections = strings["Prompts"]["Common_sections"]
        
        self.llm_model = llm_model                  # Modelo de LLM a usar  
        self.min_score = min_score                  # Umbral mínimo de score para considerar un chunk como relevante
        self.embedding_model = embedding_model      # Modelo de embedding a usar
        self.client = client                        # Cliente de OpenAI

        self.logger = RAG.rag_logger            # Logger para el RAG
        self.file = RAG.file                    # Ruta al archivo JSON con los chunks
        # self.jsonFile = RAG.jsonFile          # Ruta al archivo JSON con las instrucciones
        self.chroma_path = RAG.temp_files_dir + "\chromadb"        # Ruta al índice chroma
        self.log_path = RAG.log_file_path       # Ruta al archivo de log

        self.chunks = []          # Lista para almacenar los chunks de texto, extraido del JSON
        self.metadata = []        # Lista para almacenar la página y seccioens de un chunk
        self.prompt_count = 0     # Contador de prompts, ayuda a llevar la cuenta
        self.key = ""             # Clave del prompt actual, valor auxiliar

        try:
            self.client.delete_collection(name='licitaciones_collection')
        except Exception:
            pass  # Si no existe, ignora el error        

        self.deleteLog()
        self.logger.info("Objeto RAG inicializado.")
        print("Objeto RAG inicializado.")

        self.load_chunks()
        self.embed_json()

    def normalize_chunk(self, chunk):
        """
        Normaliza un chunk de texto:
        - Convierte a minúsculas.
        - Reemplaza caracteres especiales.
        - Lematiza las palabras usando Stanza.
        """
        # Convertir a minúsculas
        chunk = chunk.lower()

        # Reemplazos básicos
        replacements = {
            "\n": " ", 
            "\t": " ",  
        }

        for old, new in replacements.items():
            chunk = chunk.replace(old, new)

        # Reemplazar múltiples espacios por uno solo
        chunk = re.sub(r"\s+", " ", chunk).strip()

        # Lematizar cada palabra del chunk
        chunk = " ".join(WordNetLemmatizer().lemmatize(word) for word in chunk.split())

        return chunk

    def process_response(self, response):        
        response = " ".join(response.split())
        response = response.replace("[SELECCIONADO]", "")
        if "no hay información" in response.lower() or "no se menciona" in response.lower():
            return "" 
        
        return response

    def semantic_search(self, query):
        self.logger.info(f"Performing semantic search for query: {query}")
        keywords = self.semantics[self.key]
        max_count = 0
        index = -1
        best_chunk = None
        for i, chunk in enumerate(self.chunks):
            # Convertir el chunk a minúsculas para coincidencias insensibles a mayúsculas/minúsculas
            chunk_lower = chunk.lower()
            # Contar las apariciones de cada palabra clave en el chunk
            count = sum(chunk_lower.count(keyword.lower()) for keyword in keywords if keyword.lower() != "[seleccionado]")
            
            # Actualizar el mejor chunk si este tiene más coincidencias
            if count > max_count:
                max_count = count
                best_chunk = chunk
                index = i

        if best_chunk is None or index == -1:
            self.logger.warning("No matching chunk found on semantic_search.")
            return "", 0, []
        
        score = self.reranking(0.5, index)
        if score < self.min_score:
            self.logger.warning(f"Score {score} for index {index} is below the minimum threshold. Las keywords eran: {keywords}")
            return "", 0, []
        
        return self.generate_response_with_context(query, [best_chunk]), score, [index]

    def reranking(self, score, indx):
        sections = sections = self.metadata[indx][1]
        chunk = self.chunks[indx]

        # Contar cuántos strings de interés están presentes en el chunk
        interesting_strings = self.semantics[self.key]
        chunk = chunk.lower()
        matches = sum(1 for string in interesting_strings if string in chunk)

        # Verificar si la única coincidencia es "[seleccionado]"
        if matches == 1 and ("[seleccionado]" in chunk.lower()) and ("[seleccionado]" in self.key):
            return 0.0
        
        # Calcular el nuevo valor usando una función sigmoide
        if matches > 0:
            # Normalizar el número de coincidencias
            normalized_matches = matches / len(interesting_strings)
            # Introducir un peso dinámico basado en la importancia de las coincidencias
            weight = len(interesting_strings) / (matches + 1)  # Evitar división por cero
            new_score = score + (normalized_matches * weight) * (1 - score)
        else:
            return 0.0  # Si no hay coincidencias, el score es 0

        # Si se encuentra en la sección esperada aumenta su puntaje
        expected_sections = self.sections[self.key]
        intersection = set(sections) & set(expected_sections)

        if intersection:
            new_score = new_score + 0.3 * (1 - new_score)  # Aumentar el score si hay coincidencias
        
        else:
            new_score = new_score - 0.1 * (1 - new_score)

        # Asegurar que el score no exceda 1
        new_score = min(new_score, 1.0)

        return new_score

    def retrieve_similar_chunks(self, query):
        # Generar el embedding para la consulta usando la API de OpenAI
        try:
            response = self.client.embeddings.create(
                input=query,
                model=self.embedding_model,  # Modelo especificado en la inicialización
            )
            query_embedding = response.data[0].embedding  # Chroma espera una lista de floats
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            return [], 0, []

        # Conectar a la colección de ChromaDB
        chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        collection_name = "rag_chunks"
        collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        # Buscar los chunks más similares en ChromaDB
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                include=["documents", "distances", "ids"]
            )
        except Exception as e:
            self.logger.error(f"Error querying ChromaDB: {e}")
            return [], 0, []

        # Procesar resultados
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        # Convertir distancias a scores (cuanto menor la distancia, mayor el score)
        first_scores = [1 - d for d in distances]

        similar_chunks = []
        aux_score = []
        aux_indices = []

        for i, (doc, score, id_str) in enumerate(zip(documents, first_scores, ids)):
            # Extraer el índice original del id (asumiendo formato "chunk_{i}")
            try:
                indx = int(id_str.split("_")[-1])
            except Exception:
                indx = i  # fallback
            score = self.reranking(score, indx)
            if score >= self.min_score:
                similar_chunks.append(doc)
                aux_score.append(score)
                aux_indices.append(indx)

        # Crear un zip de similar_chunks y aux_score
        zipped_chunks = list(zip(similar_chunks, aux_score))

        if not zipped_chunks:
            return [], 0, []

        # Ordenar la lista de mayor a menor con respecto al score
        zipped_chunks = sorted(zipped_chunks, key=lambda x: x[1], reverse=True)

        similar_chunks, aux_score = zip(*zipped_chunks)

        if len(similar_chunks) > 3:
            similar_chunks = similar_chunks[:3]
            aux_score = aux_score[:3]
            aux_indices = aux_indices[:3]

        weights = [1 / (i + 1) for i in range(len(aux_score))]  # Higher weight for earlier elements
        weighted_sum = sum(w * d for w, d in zip(weights, aux_score))
        score = weighted_sum / sum(weights)  # Normalize by the sum of weights

        # Log de los resultados
        self.logger.info(f"Retrieved chunks: {aux_indices}, with scores: {aux_score} and a total score of: {score}.")

        return similar_chunks, score, aux_indices

    def generate_response_with_context(self, query, similar_chunks):
        context = " ".join(similar_chunks)

        # Truncar el contexto si es demasiado largo
        max_context_tokens = 3000  # Ajusta según el modelo
        context_tokens = context.split()
        if len(context_tokens) > max_context_tokens:
            context = " ".join(context_tokens[:max_context_tokens])
            self.logger.warning("Context truncated to fit token limit.")

        system_message = {
            "role": "system",
            "content": """
                Eres un asistente de IA que responde preguntas sobre un documento.
                Responde ÚNICAMENTE con la respuesta en el formato dado, nada de texto introductorio.
                Toda la información extraida debe provenir del contexto proporcionado.
                En caso de no tener una respuesta clara, responde con "No hay información".
            """
        }

        user_message = {
            "role": "user",
            "content": f"""
            Contexto: 
            {context}\n

            Pregunta: 
            {query}\n

            Formato respuesta: 
            {self.format[self.key]}\n

            Respuesta:
            """
        }
        # self.logger.info(f"\nPrompt: {user_message}\n")
        self.logger.info(f"\nPrompt enviado para el parametro: {self.key}\n")

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[system_message, user_message],
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()
        answer = self.process_response(answer)

        return answer
    
    def QNA_RAG(self, query, top_k=3):
        try:
            # Generar el embedding para la consulta
            response = self.client.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            query_embedding = response.data[0].embedding  # Chroma espera una lista de floats

            # Conectar a la colección de ChromaDB
            chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            collection_name = "rag_chunks"
            collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

            # Buscar los chunks más similares en ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "ids"]
            )

            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            # Calcular las puntuaciones basadas en las distancias
            scores = [1 - distance for distance in distances]

            # Recuperar los chunks correspondientes
            similar_chunks = [(documents[i], scores[i], int(ids[i].split("_")[-1])) for i in range(len(documents)) if i < len(documents)]

            # Ordenar por puntuación descendente
            similar_chunks = sorted(similar_chunks, key=lambda x: x[1], reverse=True)

            self.logger.info(f"Chunks similares encontrados: {similar_chunks}")

        except Exception as e:
            self.logger.error(f"Error al buscar chunks similares: {e}")
            return ""

        system_message = {
            "role": "system",
            "content": "Eres un asistente RAG, respondes preguntas de un usuario en base al contexto otorgado."
        }

        user_message = {
            "role": "user",
            "content": f"""
            Contexto: {similar_chunks}\n

            Pregunta: {query}\n

            Respuesta:
            """
        }

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[system_message, user_message],
        )

        return response.choices[0].message.content.strip()

    def rag_workflow(self, query, max_attempts=3):
        attempt = 0
        aux_query = query

        while attempt < max_attempts:
            self.prompt_count += 1

            if attempt > 0:
                aux_query = self.reformulate_query(aux_query)

            similar_chunks, score, index = self.retrieve_similar_chunks(aux_query)

            if similar_chunks:
                return self.generate_response_with_context(aux_query, similar_chunks), score, index

            attempt += 1
            self.logger.warning(f"Attempt {attempt} failed for query: {aux_query}")

        # Si no encuentra resultados después de max_attempts, devuelve valores predeterminados
        self.logger.warning(f"All {max_attempts} attempts failed for query: {aux_query}")

        return self.semantic_search(query)

    def reformulate_query(self, prompt):
        try:
            # Reformula el prompt usando la API de OpenAI
            response = self.client.chat.completions.create(
                model=self.llm_model,
            messages=[
                {
                    "role": "user",  # Asegúrate de incluir el campo 'role'
                    "content": f"""
                    Reformula el siguiente prompt y responde únicamente con el prompt reformulado: {prompt},
                    procura que el nuevo promp tenga las siguintes palabras clave: {self.semantics[self.key]}
                    """
                }
                ],
            )
            new_prompt = response.choices[0].message.content.strip()
            return new_prompt
        except Exception as e:
            self.logger.error(f"Error reformulating query: {e}")
            return prompt  # Devuelve el prompt original si ocurre un error

    def embed_json(self):
        try:
            # Inicializa el cliente y la colección de ChromaDB
            chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            collection_name = "rag_chunks"
            # Borra la colección si existe para evitar duplicados
            try:
                chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass
            collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

            # Genera embeddings y añade a la colección
            for i, chunk in enumerate(tqdm(self.chunks, desc="Generating embeddings", unit="chunk")):
                try:
                    normalized_chunk = self.normalize_chunk(chunk)
                    response = self.client.embeddings.create(
                        input=normalized_chunk,
                        model=self.embedding_model
                    )
                    self.logger.info(f"created embedding for chunk {i}: {normalized_chunk}")
                    embedding = response.data[0].embedding
                    # Añade el chunk y su embedding a la colección
                    collection.add(
                        documents=[normalized_chunk],
                        embeddings=[embedding],
                        ids=[f"chunk_{i}"]
                    )
                except Exception as e:
                    self.logger.error(f"Error generating embedding for chunk {i}: {e}")
                    continue

            self.logger.info(f"Added {len(self.chunks)} chunks to ChromaDB collection '{collection_name}'.")

        except Exception as e:
            self.logger.error(f"Error embedding JSON file with ChromaDB: {e}")

    def load_chunks(self):
        # Debug log to check the state of self.file
        if not self.file:
            self.logger.error("self.file is None or empty.")
            return

        # Use self.file directly as it already contains the parsed JSON data
        data = self.file
        self.logger.info(f"Loaded JSON data with {len(data)} pages.")

        # Extract chunks from the JSON file
        for i, page in enumerate(data):
            chunks = page.get("chunks", [])
            meta = (page.get("page", []), page.get("sections", []))
            for chunk in chunks:
                if chunk:
                    self.chunks.append(chunk)  # Extract "chunks" field
                    self.metadata.append(meta) # Extract "metadata" field

            if "tables" in page:
                table = page.get("tables", [])
                meta = page.get("page", []), page.get("sections", [])
                if table:
                    zipped = list(zip(*table)) 
                    table_string = ""
                    for row in zipped:
                        if row:
                            cleaned_row = [cell if cell is not None else "" for cell in row]  # Replace None with ""
                            table_string += ", ".join(cleaned_row) + " "
                    self.chunks.append(table_string)  # Extract "chunks" field
                    self.metadata.append(meta)  # Extract "metadata" field

                    table_string = ""
                    for column in table:
                        cleaned_column = [cell if cell is not None else "" for cell in column]  # Replace None with ""
                        table_string += ", ".join(cleaned_column) + " "
                        self.chunks.append(table_string)  # Extract "chunks" field
                        self.metadata.append(meta)  # Extract "metadata" field


    def deleteLog(self):
        try:
            if os.path.exists(self.log_path):
                # Abrir el archivo en modo escritura y reemplazar su contenido con nada
                with open(self.log_path, 'w', encoding='utf-8') as log_file:
                    log_file.write("")
                self.logger.info("Archivo reseteado.")
            else:
                print("El archivo de log no existe.")
        except Exception as e:
            print(f"Error al intentar limpiar el contenido del archivo de log: {e}")

    def get_prompt_keys(self):
        return list(self.prompts.keys())

rag_lock = Lock()

def process_prompt(rag_object, prompt_key, prompt):
    """
    Función auxiliar para procesar un prompt.
    """
    with rag_lock:  # Asegurar acceso exclusivo a rag_object
        rag_object.key = prompt_key
        response, score, index = rag_object.rag_workflow(prompt)

    # Validar que los valores sean strings
    if not isinstance(prompt, str):
        raise ValueError(f"El valor de 'prompt' no es un string: {prompt}")
    if not isinstance(response, str):
        raise ValueError(f"El valor de 'response' no es un string: {response}")

    return Prompt(prompt_key=prompt_key, prompt=prompt, response=response, score=score, index=index), response

def main(cod):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'temp_files', 'output.json')

    db = DBManager()
    db.openConnection()

    rag_object = RAG(embedding_model=r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\models\linear_adapter_V3_E30_B32_L0.003.pth")

    results = []
    results_db = [cod]

    # Paralelizar el procesamiento de los prompts
    with ThreadPoolExecutor() as executor:
        # Crear tareas para cada prompt
        futures = {
            executor.submit(process_prompt, rag_object, prompt_key, prompt): prompt_key
            for prompt_key, prompt in rag_object.prompts.items()
        }

        # Usar tqdm para mostrar el progreso
        for future in tqdm(as_completed(futures), desc="Processing Prompts", unit="prompt", total=len(futures)):
            try:
                result, response = future.result()
                results.append(result)
                results_db.append(response)

            except Exception as e:
                print(f"Error processing prompt: {e}")

    db.insertDb("AnexoI", results_db)

    # Validar y guardar los resultados usando Pydantic
    rag_results = RAGResult(prompts=results)

    # Guardar los resultados en un archivo JSON
    with open(output_dir, "w", encoding="utf-8") as output_file:
        output_file.write(rag_results.model_dump_json(indent=4))

    print(f"Results saved to {output_dir}")

