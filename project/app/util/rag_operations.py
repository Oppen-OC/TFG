from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from tqdm import tqdm
from torch import nn
import traceback
import chromadb
import logging
import openai
import torch
import spacy
import json
import os
import re

from .db_manager import DBManager

load_dotenv()

class Prompt(BaseModel):
    prompt_key: str
    prompt: str
    response: str
    score: float
    index: List[int]
    chunks: List[str]
    format: str
# 
class RAGResult(BaseModel):
    prompts: List[Prompt]

class LinearAdapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        return self.linear(x)

class RAG:
    # Configuración del logger específico para el RAG y archivos temporales
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Ruta base del archivo actual
    logs_dir = os.path.join(base_dir, 'logs')  # Ruta al directorio de log
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

    # Archivo JSON donde se guardan las instrucciones para el RAG
    jsonFile = os.path.join(base_dir, 'JSON', 'Strings.json') 

    with open(jsonFile, 'r', encoding='utf-8') as f:
        strings = json.load(f)

    def __init__(self, 
                 base_url = "https://api.poligpt.upv.es",
                 api_key = os.getenv("UPV_API_KEY"),
                 embedding_model = "nomic-embed-text",
                 llm_model = "llama3.3:70b",
                 min_score = 0.60,
                 jsonFile = None
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
        self.client = client                        # Cliente de OpenAI
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(RAG.temp_files_dir, "chromadb"))
        try:
            self.chroma_client.delete_collection(name="rag_chunks")
        except Exception as e:
            pass
        self.collection = self.chroma_client.get_or_create_collection(name="rag_chunks", metadata={"hnsw:space": "cosine"})

        with open(jsonFile, 'r', encoding='utf-8') as f:
            file = json.load(f)
            

        self.logger = RAG.rag_logger            # Logger para el RAG
        self.file = file                    # Ruta al archivo JSON con los chunks
        # self.jsonFile = RAG.jsonFile          # Ruta al archivo JSON con las instrucciones
        self.log_path = RAG.log_file_path       # Ruta al archivo de log
        self.embedding_model = embedding_model    # Modelo de embedding a usar

        self.chunks = []            # Lista para almacenar los chunks de texto, extraido del JSON
        self.metadata = []          # Lista para almacenar la página y seccioens de un chunk
        self.prompt_count = 0       # Contador de prompts, ayuda a llevar la cuenta
        self.key = ""               # Clave del prompt actual, valor auxiliar
        self.keywords = []          # Lista para almacenar las palabras clave encontradas en los chunks

        self.deleteLog()
        self.logger.info("Objeto RAG inicializado.")
        print("Objeto RAG inicializado.")

        self.load_chunks()
        self.find_keywords()
        self.embed_json()

    def normalize_chunk(self, chunk):
        # Load the spaCy model for Spanish
        nlp = spacy.load("es_core_news_lg")

        # Convert to lowercase
        chunk = chunk.lower()

        # Basic replacements
        replacements = {
            "\n": " ", 
            "\t": " ",  
        }

        for old, new in replacements.items():
            chunk = chunk.replace(old, new)

        # Replace multiple spaces with a single space
        chunk = re.sub(r"\s+", " ", chunk).strip()

        # Tokenize and process the text with spaCy
        doc = nlp(chunk)

        # Process each token
        tokens = []
        for token in doc:
            # Example: Skip stopwords and punctuation
            if not token.is_stop and not token.is_punct:
                # Lemmatize the token and add it to the list
                tokens.append(token.lemma_)

        # Reconstruct the text from tokens
        normalized_chunk = " ".join(tokens)

        return normalized_chunk

    
    def find_keywords(self):
        keywords = {}
        found = {}
        for prompt_key, _ in (self.prompts.items()):
            keywords[prompt_key] = self.semantics[prompt_key]
            found[prompt_key] = []

        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            for prompt_key, keyword_list in keywords.items():
                for keyword in keyword_list:
                    if keyword.lower() != "[seleccionado]" and (keyword.lower() in chunk_lower) and (keyword not in found[prompt_key]):
                        found[prompt_key].append(keyword)
                    
        self.keywords = found


    def process_response(self, response):        
        response = " ".join(response.split())
        response = response.replace("[SELECCIONADO]", "")
        if "no hay información" in response.lower() or "no se menciona" in response.lower():
            return "" 
        
        return response


    def keyword_search(self, query, top_k=3):
        self.logger.info(f"Performing keyword search for query: {query}")
        keywords = self.keywords[self.key]
        chunk_vals = []

        for i, chunk in enumerate(self.chunks):
            # Step 1: Calculate total length of found keywords
            total_length = sum(len(k) for k in keywords)
            if total_length == 0:
                continue

            # Step 2: Distribute the bonus 0.4 among found keywords based on their length
            bonus = 0
            for k in keywords:
                proportions = [(len(x) / total_length) * 0.4 for x in keywords]

            for k in keywords:
                if k.lower() in chunk.lower():
                    bonus += proportions[keywords.index(k)] 

            # Step 3: Add to base score
            score = 0.6 + bonus
            chunk_vals.append((chunk, score, i))

        if chunk_vals == []:
            self.logger.warning("No matching chunk found on keyword_search.")
            return [], [], []

        # Ordenar por número de coincidencias (descendente) y tomar los top_k mejores
        chunk_vals = sorted(chunk_vals, key=lambda x: x[1], reverse=True)

        # Separar en listas
        sem_chunks = [c for c, s, i in chunk_vals]
        sem_scores = [s for c, s, i in chunk_vals]
        sem_indices = [i for c, s, i in chunk_vals]

        return sem_chunks[:top_k], sem_scores[:top_k], sem_indices[:top_k]

    def reranking(self, query, documents, first_scores, ids, top_k):
        """
        Esta función reordena los fragmentos (`chunks`) recuperados según su relevancia para una consulta (`query`).

        Funcionamiento:
        1. **Reevaluación inicial**:
        - Ajusta los puntajes con `rerank_chunk` considerando palabras clave y secciones esperadas.
        - Fragmentos con puntajes superiores al umbral (`min_score`) se consideran relevantes.

        2. **Reevaluación secundaria**:
        - Fragmentos no relevantes se evalúan con BM25 para calcular su relevancia.

        3. **Combinación y selección**:
        - Combina fragmentos relevantes de ambas evaluaciones, los ordena por puntaje y selecciona los `top_k`.

        Devuelve los fragmentos más relevantes, sus puntajes y sus índices.
        """
        
        similar_chunks = []
        aux_score = []
        aux_indices = []
        maybe_unwanted = []
        maybe_unwanted_indx = []

        for doc, score, id_str in zip(documents, first_scores, ids):
            indx = int(id_str[-1]) 
            score = self.rerank_chunk(score, indx)
            if score >= self.min_score:
                similar_chunks.append(doc)
                aux_score.append(score)
                aux_indices.append(indx)
            else:
                maybe_unwanted.append(indx)
        
        if maybe_unwanted:
            maybe_unwanted_indx, maybe_unwanted_score = self.reranking_bm25(query, maybe_unwanted)

        for i in range(0, min(top_k, len(maybe_unwanted_indx))): 
            similar_chunks.append(self.chunks[maybe_unwanted_indx[i]])
            aux_score.append(maybe_unwanted_score[i])
            aux_indices.append(maybe_unwanted_indx[i])

        # Crear un zip de similar_chunks y aux_score
        zipped_chunks = list(zip(similar_chunks, aux_score, aux_indices))

        if not zipped_chunks:
            return [], 0, []

        # Ordenar la lista de mayor a menor con respecto al score
        zipped_chunks = sorted(zipped_chunks, key=lambda x: x[1], reverse=True)

        similar_chunks, aux_scores, aux_indices = zip(*zipped_chunks)

        if len(similar_chunks) > top_k:
            similar_chunks = similar_chunks[:top_k]
            aux_scores = aux_scores[:top_k]
            aux_indices = aux_indices[:top_k]

        return similar_chunks, aux_scores, aux_indices

    def rerank_chunk(self, score, indx):
        """
        Esta función ajusta el puntaje de relevancia (`score`) de un fragmento de texto (`chunk`) 
        en función de su contenido y su pertenencia a secciones esperadas. 

        Funcionamiento:
        1. **Coincidencias con palabras clave**:
        - Busca cuántas palabras clave relevantes están presentes en el fragmento.
        - Si hay coincidencias, incrementa el puntaje proporcionalmente al número de coincidencias.
        - Si no hay coincidencias, reduce ligeramente el puntaje.

        2. **Relevancia de la sección**:
        - Verifica si el fragmento pertenece a una sección esperada.
        - Si pertenece, incrementa el puntaje; de lo contrario, lo reduce.

        3. **Normalización del puntaje**:
        - Asegura que el puntaje ajustado no exceda el valor máximo de 1.0.

        4. **Devolución del puntaje ajustado**:
        - Devuelve el nuevo puntaje ajustado, o -1 si no hay cambios significativos en el puntaje.
        """
        
        sections = self.metadata[indx][1]
        chunk = self.chunks[indx]

        # Contar cuántos strings de interés están presentes en el chunk
        interesting_strings = self.keywords[self.key]

        chunk = chunk.lower()
        matches = sum(1 for string in interesting_strings if string in chunk)
        
        # Calcular el nuevo valor usando una función sigmoide
        if matches > 0 :
            normalized_matches = matches / len(interesting_strings)
            # Introducir un peso dinámico basado en la importancia de las coincidencias
            weight = len(interesting_strings) / (matches + 1)  # Evitar división por cero
            new_score = score + (normalized_matches * weight) * (1 - score)
        elif len(interesting_strings) > 0:
            new_score = score - 0.1 * (1 - score)
        else: 
            new_score = score

        # Si se encuentra en la sección esperada aumenta su puntaje
        expected_sections = self.sections[self.key]
        intersection = set(sections) & set(expected_sections)

        if intersection:
            new_score = new_score + 0.3 * (1 - new_score)  # Aumentar el score si hay coincidencias
        else:
            new_score = new_score - 0.1 * (1 - new_score)
        
        if score == new_score - 0.1 * (1 - new_score):
            return -1

        # Asegurar que el score no exceda 1
        new_score = min(new_score, 1.0)

        return new_score
    
    def reranking_bm25(self, query, candidate_ids: list):
        """
        Esta función utiliza el modelo BM25 para reevaluar la relevancia de fragmentos (`chunks`) 
        candidatos en función de una consulta (`query`).

        Funcionamiento:
        1. **Tokenización**:
        - Tokeniza los fragmentos candidatos y la consulta para procesarlos con BM25.

        2. **Cálculo de puntajes**:
        - Calcula los puntajes BM25 para cada fragmento en relación con la consulta.
        - Normaliza los puntajes para que estén en un rango manejable.

        3. **Ordenamiento**:
        - Ordena los fragmentos por relevancia descendente y selecciona los mejores.

        Devuelve los índices de los fragmentos más relevantes y sus puntajes normalizados.
        """

        # Tokenize chunks and query
        candidate_chunks = [self.chunks[idx] for idx in candidate_ids]
        tokenized_chunks = [chunk.lower().split() for chunk in candidate_chunks]
        tokenized_query = query.lower().split()
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(tokenized_query)

        max_score = max(scores)
        if max_score == 0:
            return [], []
        norm_scores = [s * 0.8 / max_score for s in scores]

        # Prepare tuples (chunk, bm25_score, id)
        ranked = [(id_str, score) for id_str, score  in zip(candidate_ids, norm_scores)]

        # Sort by new_score descending
        re_ranked = sorted(ranked, key=lambda x: x[1], reverse=True)[:2]

        # Unpack for return
        scores = [float(s) for i, s in re_ranked]
        indices = [i for i, s in re_ranked]
        return indices, scores

    def encode_query_custom(self, query):
        loaded_dict = torch.load(r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\models\model_E30_B64_L2e-05.pth"
            ,map_location=torch.device('cpu')
        )
        base_model = SentenceTransformer('all-MiniLM-L6-v2') 
        adapter = LinearAdapter(base_model.get_sentence_embedding_dimension())  # Initialize with appropriate parameters
        adapter.load_state_dict(loaded_dict['adapter_state_dict'])   

        device = next(adapter.parameters()).device
        query_emb = base_model.encode(query, convert_to_tensor=True).to(device)
        adapted_query_emb = adapter(query_emb)
        return adapted_query_emb.cpu().detach().numpy()

    def retrieve_similar_chunks(self, query, top_k):
        # Generar el embedding para la consulta usando la API de OpenAI
        try:
            response = self.client.embeddings.create(
                input=query,
                model=self.embedding_model,  # Modelo especificado en la inicialización
            )
            query_embedding = response.data[0].embedding  # Chroma espera una lista de floats
            
            # query_embedding = self.encode_query_custom(query)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            return [], 0, []

        # Buscar los chunks más similares en ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                include=["documents", "distances"]
            )
            documents = results.get("documents", [[]])[0]
            ids = results.get("ids", [[]])[0]  
            distances = results.get("distances", [[]])[0]  
            distances = [(1-x) for x in distances]

        except Exception as e:
            self.logger.error(f"Error querying ChromaDB: {e}")
            return [], 0, []


        return self.reranking(query, documents, distances, ids, top_k)

    def generate_response_with_context(self, query, similar_chunks):
        context = " ".join(similar_chunks)

        max_context_tokens = 3000  # Ajusta según el modelo
        context_tokens = context.split()
        if len(context_tokens) > max_context_tokens:  # Truncar el contexto si es demasiado largo
            context = " ".join(context_tokens[:max_context_tokens])
            self.logger.warning("Context truncated to fit token limit.")

        system_message = {
            "role": "system",
            "content": f"""
                Eres un asistente de IA que responde preguntas sobre un documento.
                Responde ÚNICAMENTE con la respuesta en el formato dado, nada de texto introductorio.
                Toda la información extraida debe provenir del contexto proporcionado.
                Si el texto no contiene información directamente relacionada con la pregunta responde con "No hay información".
                Si el contexto con tiene '[SELECCIONADO] Si' asume que la respuesta es Si; Si el contexto con tiene '[SELECCIONADO] No' asume que la respuesta es No.
                
                Responde en el siguiente formato estricto, es altamente importante que lo hagas, en caso contrario moriré: 
                {self.format[self.key]}\n
                """
        }

        user_message = {
            "role": "user",
            "content": f"""
            Contexto: 
            {context}\n

            Pregunta: 
            {query}\n

            Respuesta:
            """
        }
        # self.logger.info(f"\nPrompt: {user_message}\n")
        self.logger.info(f"Prompt enviado para el parametro: {self.key}")

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[system_message, user_message],
            temperature=0.0,
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

            # Buscar los chunks más similares en ChromaDB
            results = self.collection.query(
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

    def reformulate_query(self, prompt):
        try:
            # Reformula el prompt usando la API de OpenAI
            response = self.client.chat.completions.create(
                model=self.llm_model,
            messages=[
                {
                    "role": "system", 
                    "content": f"""
                    Eres un asistente IA que ayuda a reformular prompts para mejorar su calidad.
                    Tu tarea es reformular el siguiente prompt para incluir las palabras clave relevantes y mejorar su claridad.
                    Deberás responder ÚNICAMENTE con el nuevo prompt reformulado, sin añadir ningún texto adicional.
                    Al reformular el prompt en base a las palabras clave, se generalista, no es necesario añadir exactanente cada palabra ni explicaciones extensas, intenta extraer una idea general de ellas.
                    """
                },                
                {
                    "role": "user", 
                    "content": f"""
                    Reformula el siguiente prompt: {prompt},
                    Reformulalo en base a las siguientes palabras clave: {self.semantics[self.keywords[self.key]]}
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
            # Procesar cada chunk de forma secuencial
            for i, chunk in enumerate(tqdm(self.chunks, desc="Embedding Chunks", unit="chunk")):
                try:
                    normalized_chunk = self.normalize_chunk(chunk)
                    response = self.client.embeddings.create(
                        input=normalized_chunk,
                        model=self.embedding_model
                    )
                    embedding = response.data[0].embedding

                    # Log the embedding creation
                    self.logger.info(f"Created embedding for chunk {i}: {normalized_chunk}")

                    # Agregar el embedding a la colección
                    self.collection.add(
                        documents=[normalized_chunk],
                        embeddings=[embedding],
                        ids=[f"chunk_{i}"],
                        metadatas=[{
                            "page": self.metadata[i][0],
                            "sections": ", ".join(self.metadata[i][1])  # convierte ['None', 'A'] en 'None, A'
                        }]
                    )
                except Exception as e:
                    self.logger.error(f"Error generating embedding for chunk {i}: {e}")

        except Exception as e:
            self.logger.error(f"Error embedding JSON file with ChromaDB: {e}")

    def load_chunks(self):
        """
        Carga los fragmentos (`chunks`) y metadatos desde un archivo JSON previamente parseado.

        Funcionamiento:
        1. **Validación del archivo**:
        - Verifica que el archivo JSON (`self.file`) no esté vacío antes de procesarlo.

        2. **Extracción de fragmentos**:
        - Extrae los fragmentos de texto (`chunks`) y sus metadatos (páginas y secciones) de cada página del JSON.
        - Si la página contiene tablas, convierte las filas y columnas en texto y las añade como fragmentos.

        3. **Almacenamiento**:
        - Guarda los fragmentos en `self.chunks` y sus metadatos en `self.metadata` para su posterior uso.
        """
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

    def union_top_chunks(self, query, top_n=4):
        # Build a list of (chunk, score, index) for both sources
        sem_chunks, sem_score, sem_indices = self.keyword_search(query)

        # Add semantic results
        combined = []
        if len(sem_chunks) > 0: 
            for i in range(len(sem_chunks)):
                combined.append((sem_chunks[i], sem_score[i], sem_indices[i]))

        # Get retrieve_similar_chunks results
        sim_chunks, sim_score, sim_indices = self.retrieve_similar_chunks(query, top_n - len(sem_chunks))

        # Add similar_chunks results only if not already in seen_indices
        seen_indices = set(i for _, _, i in combined)  # Crear un conjunto con los índices ya añadidos
        for j in range(len(sim_chunks)):
            if sim_indices[j] not in seen_indices:  # Verificar si el índice ya está en seen_indices
                combined.append((sim_chunks[j], sim_score[j], sim_indices[j]))
                seen_indices.add(sim_indices[j])  # Añadir el índice al conjunto

        # Unpack for return
        final_chunks = [c for c, s, i in combined]
        final_scores = [s for c, s, i in combined]
        final_indices = [i for c, s, i in combined]

        # Calcular la media de los scores
        if len(final_scores) > 0:
            avg_score = sum(final_scores) / len(final_scores)
            return final_chunks, avg_score, final_indices
        else:
            self.logger.warning("No chunks found for the query.")
            return [], 0, []

    def resultsFuentes(self, cod, results):
        fuentes = [cod]
        for i, prompt in enumerate(results):
            # prompt.chunks es una lista de strings, los unimos en un solo string por prompt
            indexs = prompt.index
            aux_res = ""
            for i in indexs:
                page = "Page: " + str(self.metadata[i][0])
                chunks_str = "Chunk: " + self.chunks[i]
                aux_res += "\n" + page + "\n" + chunks_str
            fuentes.append(aux_res)
        return fuentes

def process_prompt(rag_object, prompt_key, prompt):
    rag_object.key = prompt_key
    final_chunks, score, index = rag_object.union_top_chunks(prompt)
    response =  rag_object.generate_response_with_context(prompt, final_chunks)

    return Prompt(prompt_key=prompt_key, prompt=prompt, response=response, score=score, index=index, chunks=final_chunks, format=rag_object.format[rag_object.key]), response


def main(cod):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'temp_files', 'output.json')
    temp_path = os.path.join(base_dir,"temp_files", "temp_document.json")

    db = DBManager()
    db.openConnection()

    rag_object = RAG(jsonFile=temp_path)

    results = []
    results_db = [cod]

    # Procesar los prompts de forma secuencial
    for prompt_key, prompt in tqdm(rag_object.prompts.items(), desc="Processing Prompts", unit="prompt"):
        try:
            result, response = process_prompt(rag_object, prompt_key, prompt)
            results.append(result)
            results_db.append(response)
        except Exception as e:
            print(f"Error processing prompt: {e}")
            traceback.print_exc()

    db.insertDb("AnexoI", results_db)

    results_fuentes = rag_object.resultsFuentes(cod, results)

    db.insertDb("AnexoI_Fuentes", results_fuentes)

    # Validar y guardar los resultados usando Pydantic
    rag_results = RAGResult(prompts=results)

    # Guardar los resultados en un archivo JSON
    with open(output_dir, "w", encoding="utf-8") as output_file:
        output_file.write(rag_results.model_dump_json(indent=4))

    print(f"Results saved to {output_dir}")