from dotenv import load_dotenv
import openai
import os
import faiss
import json
import logging
import numpy as np
from tqdm import tqdm
import numpy as np
import math
from pydantic import BaseModel
from typing import List, Optional
import re
from db_manager import DBManager

load_dotenv()

class Chunk(BaseModel):
    text: str
    score: Optional[float] = None

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
    rag_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    rag_handler.setLevel(logging.INFO)
    rag_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    rag_handler.setFormatter(rag_formatter)
    rag_logger.addHandler(rag_handler)
    rag_logger.propagate = False

    def __init__(self, 
                 file = None,
                 jsonFile = None,
                 embedding_model = "nomic-embed-text", #SentenceTransformer("all-MiniLM-L6-v2")  
                 client = openai.OpenAI(api_key=os.getenv("UPV_API_KEY"), base_url="https://api.poligpt.upv.es"),
                 min_score = 0.60
                 ):
        
        with open(jsonFile, 'r', encoding='utf-8') as f:
            strings = json.load(f)
        self.prompts = strings["Prompts"]["Ajuntament"]  
        self.semantics = strings["Prompts"]["Semantic"] 
        self.tokens = strings["Prompts"]["Max_tokens"]
        self.format = strings["Prompts"]["Ajuntament_formato"]
        self.sections = strings["Prompts"]["Common_sections"]
        
        self.key = ""
        self.file = file                # Ruta al archivo JSON con los chunks
        self.prompt_count = 0           # Contador de prompts, ayuda a llevar la cuenta
        self.faiss_path = RAG.temp_files_dir + "/embeddings_index.faiss"
        self.embedding_model = embedding_model
        self.client = client            # Cliente de OpenAI
        self.logger = RAG.rag_logger
        self.chunks = []
        self.metadata = []
        
        self.min_score = min_score

        if os.path.exists(self.faiss_path):
                os.remove(self.faiss_path)
                self.logger.info(f"Previous FAISS index deleted: {self.faiss_path}")

        self.logger.info("RAG object initialized.")
        print("RAG object initialized.")

    def normalize_chunk(self, chunk):
        """
        Normaliza un chunk de texto para facilitar su procesamiento.
        :param chunk: Texto del chunk a normalizar.
        :return: Texto normalizado.
        """
        # Convertir a minúsculas
        chunk = chunk.lower()

        # Reemplazar caracteres especiales o no deseados
        replacements = {
            "\n": " ",  # Reemplazar saltos de línea por espacios
            "\t": " ",  # Reemplazar tabulaciones por espacios
        }

        for old, new in replacements.items():
            chunk = chunk.replace(old, new)

        # Eliminar caracteres no alfanuméricos (excepto espacios y puntuación básica)
        # chunk = re.sub(r"[^a-z0-9áéíóúüñ.,;:!?()\[\] ]", "", chunk)

        # Reemplazar múltiples espacios por uno solo
        chunk = re.sub(r"\s+", " ", chunk).strip()

        return chunk

    def process_response(self, response):        
        response = " ".join(response.split())
        return response

    def semantic_search(self, query):
        self.logger.info(f"Performing semantic search for query: {query}")
        keywords = self.semantics[self.key]
        for i, chunk in enumerate(self.chunks):
            # Convertir el chunk a minúsculas para coincidencias insensibles a mayúsculas/minúsculas
            chunk_lower = chunk.lower()
            # Contar las apariciones de cada palabra clave en el chunk
            count = sum(chunk_lower.count(keyword.lower()) for keyword in keywords)
            
            max_count = 0
            best_chunk = None
            # Actualizar el mejor chunk si este tiene más coincidencias
            if count > max_count:
                max_count = count
                best_chunk = chunk
                index = i

        if best_chunk is None:
            self.logger.warning("No matching chunk found.")
            return "", 0, []
        
        score = self.reranking(0.5, index)
        return self.generate_response_with_context(query, [best_chunk]), score, [index]

    def reranking(self, score, indx):

        sections = sections = self.metadata[indx][1]
        chunk = self.chunks[indx]

        # Contar cuántos strings de interés están presentes en el chunk
        interesting_strings = self.semantics[self.key]
        chunk = chunk.lower()
        matches = sum(1 for string in interesting_strings if string in chunk)

        # Calcular el nuevo valor usando una función sigmoide
        if matches > 0:
            # Normalizar el número de coincidencias
            normalized_matches = matches / len(interesting_strings)
            sigmoid_value = 1 / (1 + math.exp(-12 * (normalized_matches - 0.5)))  # Ajusta el factor 12 para controlar la pendiente
            new_score = score + sigmoid_value * (1 - score)
        else:
            new_score = score -0.1 # Mantener el valor original si no hay coincidencias

        expected_sections = self.sections[self.key]
        intersection = set(sections) & set(expected_sections)

        if intersection:
            new_score = new_score + 0.1 * (1 - new_score)  # Aumentar el score si hay coincidencias
        
        else:
            new_score = new_score - 0.1 * (1 - new_score)

        # Asegurar que el score no exceda 1
        new_score = min(new_score, 1.0)

        return new_score

    #Devuelve los chunks de texto más similares a la consulta
    def retrieve_similar_chunks(self, query):
        # Generar el embedding para la consulta usando la API de OpenAI
        try:
            response = self.client.embeddings.create(
                input=query,
                model=self.embedding_model,  # Modelo especificado en la inicialización
            )
            query_embedding = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)  # Asegurar formato correcto
        
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            return [], 0, []

        # Cargar el índice FAISS desde el archivo
        index = faiss.read_index(self.faiss_path)

        # Buscar los chunks más similares en el índice FAISS
        k = min(10, index.ntotal)
        distances, indices = index.search(query_embedding, k=k)  # Buscar los 10 más similares

        # Convertir los resultados a listas
        indices = indices.flatten().tolist()
        distances = distances.flatten().tolist()

        # Usa la distancia para calcular un valor de similitud
        scores = [abs(distance - 1) for distance in distances]

        # Recuperar los chunks correspondientes
        similar_chunks = []
        aux_score = []
        aux_indices = []
        
        for i, indx in enumerate(indices):
            score = scores[i]
            score = self.reranking(score, indx)
            if score >= self.min_score:
                # Solo agregar si el score es mayor o igual a min_score
                similar_chunks.append(self.chunks[indx])
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
        self.logger.info(f"\nPrompt: {user_message}\n")

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model="llama3.3:70b",
            messages=[system_message, user_message],
        )

        answer = response.choices[0].message.content.strip()
        answer = self.process_response(answer)

        return answer

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
                model="llama3.3:70b",
            messages=[
                {
                    "role": "user",  # Asegúrate de incluir el campo 'role'
                    "content": f"Reformula el siguiente prompt y responde únicamente con el prompt reformulado: {prompt}"
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
            # Generate embeddings for each chunk
            embeddings = []
            for i, chunk in enumerate(self.chunks):
                try:
                    normalized_chunk = self.normalize_chunk(chunk)
                    response = self.client.embeddings.create(
                        input=normalized_chunk,
                        model=self.embedding_model
                    )
                    self.logger.info(f"created embedding for chunk {i}: {normalized_chunk}")
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)      
                    
                except Exception as e:
                    self.logger.error(f"Error generating embedding for chunk {i}: {e}")
                continue           

            self.logger.info(f"Generated: {len(embeddings)} embeddings")

            # Convert embeddings to a NumPy array
            embeddings_array = np.array(embeddings, dtype='float32')

            # Initialize a FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)

            # Save the FAISS index to disk
            faiss.write_index(index, self.faiss_path)
            self.logger.info(f"FAISS index saved to {self.faiss_path}")

        except Exception as e:
            self.logger.error(f"Error embedding JSON file: {e}")

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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    string_path = os.path.join(base_dir, 'JSON', 'Strings.json')
    output_dir = os.path.join(base_dir, 'temp_files', 'output.json')
    file_path = os.path.join(base_dir, 'temp_files', 'temp_document.json')  
    db = DBManager()
    db.openConnection()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        file = json.load(f)

    rag_object = RAG(file, string_path)
    rag_object.load_chunks()
    rag_object.embed_json()

    with open(string_path, 'r', encoding='utf-8') as f:
        strings_anexo = json.load(f)["Prompts"]

    results = []
    results_db = []

    # Agregar barra de progreso para los prompts
    for prompt_key, prompt in tqdm(rag_object.prompts.items(), desc="Processing Prompts", unit="prompt"):
        rag_object.key = prompt_key
        response, score, index = rag_object.rag_workflow(prompt)
        results.append(Prompt(prompt_key=prompt_key, prompt=prompt, response=response, score=score, index=index))
        results_db.append(response)

    db.insertDb("AnexoI", results_db)

    # Validar y guardar los resultados usando Pydantic
    rag_results = RAGResult(prompts=results)

    # Guardar los resultados en un archivo JSON
    with open(output_dir, "w", encoding="utf-8") as output_file:
        output_file.write(rag_results.model_dump_json(indent=4))

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()