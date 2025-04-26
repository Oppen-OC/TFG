from dotenv import load_dotenv
import openai
import os
import faiss
import json
import logging
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math

load_dotenv()

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
                 minimun_distance = 0.7
                 ):
        
        with open(jsonFile, 'r', encoding='utf-8') as f:
            strings = json.load(f)
        self.prompts = strings["Prompts"]["Ajuntament"]  
        self.semantics = strings["Prompts"]["Metadata"] 
        
        self.file = file
        self.prompt_count = 0
        self.metadata = []
        self.faiss_path = RAG.temp_files_dir + "/embeddings_index.faiss"
        self.embedding_model = embedding_model
        self.client = client
        self.logger = RAG.rag_logger
        self.chunks = []
        
        self.minimun_distance = minimun_distance

        if os.path.exists(self.faiss_path):
                os.remove(self.faiss_path)
                self.logger.info(f"Previous FAISS index deleted: {self.faiss_path}")

        self.logger.info("RAG object initialized.")
        print("RAG object initialized.")

    def reranking(self, chunk, distance):
        # Contar cuántos strings de interés están presentes en el chunk
        interesting_strings = self.semantics[self.prompt_count]
        chunk = chunk.lower()
        matches = sum(1 for string in interesting_strings if string in chunk)

        # Calcular el nuevo valor de distance usando una función sigmoide
        if matches > 0:
            # Normalizar el número de coincidencias
            normalized_matches = matches / len(interesting_strings)
            sigmoid_value = 1 / (1 + math.exp(-20 * (normalized_matches - 0.5)))  # Ajusta el factor 20 para controlar la pendiente
            new_distance = distance + sigmoid_value * (1 - distance)
            #print(f"Matches {matches}]: {distance} - {new_distance}")
        else:
            new_distance = distance -0.1 # Mantener el valor original si no hay coincidencias

        # Asegurar que el valor de distance no exceda 1
        new_distance = min(new_distance, 1.0)

        return new_distance

    #Devuelve los chunks de texto más similares a la consulta
    def retrieve_similar_chunks(self, query):
        # Generar el embedding para la consulta usando la API de OpenAI
        try:
            response = self.client.embeddings.create(
                input=query,
                model=self.embedding_model  # Modelo especificado en la inicialización
            )
            query_embedding = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)  # Asegurar formato correcto
        
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            return []

        # Cargar el índice FAISS desde el archivo
        index = faiss.read_index(self.faiss_path)

        # Buscar los chunks más similares en el índice FAISS
        distances, indices = index.search(query_embedding, k=10)  # Buscar los 5 más similares

        # Convertir los resultados a listas
        indices = indices.flatten().tolist()
        distances = distances.flatten().tolist()

        # Recuperar los chunks correspondientes
        similar_chunks = []
        aux_dist = []
        aux_indices = []
        
        for i, indx in enumerate(indices):
            chunk = self.chunks[indx]
            dist = distances[i]
            distance = self.reranking(chunk, dist)
            if distance >= self.minimun_distance:
                # Solo agregar si la distancia es mayor o igual a minimun_distance
                similar_chunks.append(chunk)
                aux_dist.append(distance)
                aux_indices.append(indx)

        # Crear un zip de similar_chunks y aux_dist
        zipped_chunks = list(zip(similar_chunks, aux_dist))

        # Ordenar la lista de mayor a menor con respecto a la distancia
        zipped_chunks = sorted(zipped_chunks, key=lambda x: x[1], reverse=True)

        # Separar los chunks y las distancias ordenadas
        similar_chunks, aux_dist = zip(*zipped_chunks)

        if len(similar_chunks) > 5:
            similar_chunks = similar_chunks[:5]
            aux_dist = aux_dist[:5]
        
        # Log de los resultados
        self.logger.info(f"Retrieved chunks: {aux_indices}, with distances: {aux_dist}.")

        return similar_chunks

    def generate_response_with_context(self, query, similar_chunks):
        context = " ".join(similar_chunks)

        prompt = f"""
        Eres un asistente de IA que responde preguntas sobre un documento, la informacion junto a [SELECCIONADO] te es de especial interes.
        Contexto:\n{context}\n
        Pregunta: {query}\n
        Respuesta:
        """

        self.logger.info(f"Prompt nº {self.prompt_count}: {prompt}")

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model="llama3.3:70b",
            messages=[{"role": "assistant",
                       "content": prompt
                       }],
        )

        answer = response.choices[0].message.content.strip()

        # Check if the answer is relevant or empty
        if not answer or "I'm sorry" in answer or "I don't know" in answer:
            return "N/A"

        return answer

    def rag_workflow(self, query):
        similar_chunks = self.retrieve_similar_chunks(query)
        self.prompt_count += 1
        if similar_chunks == []:
            print("Vacio")
            self.logger.info(f"No similar chunks found for query: {query}")
            return "N/A_1"
        
        return self.generate_response_with_context(query, similar_chunks)


    def embed_json(self):
        try:
            # Generate embeddings for each chunk
            embeddings = []
            for i, chunk in enumerate(self.chunks):
                try:
                    response = self.client.embeddings.create(
                        input=chunk,
                        model=self.embedding_model
                    )
                    self.logger.info(f"created embedding for chunk {i}: {chunk}")
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
                    table_string = " ".join([", ".join(row) for row in zipped])
                    self.chunks.append(table_string)  # Extract "chunks" field
                    self.metadata.append(meta)  # Extract "metadata" field




def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    string_path = os.path.join(base_dir, 'JSON', 'Strings.json')
    output_dir = os.path.join(base_dir, 'temp_files', 'output.json')  # Cambiar a .json
    file_path = os.path.join(base_dir, 'temp_files', 'temp_document.json')  
    
    with open(file_path, 'r', encoding='utf-8') as f:
        file = json.load(f)

    rag_object = RAG(file, string_path)
    rag_object.load_chunks()
    rag_object.embed_json()

    results = []  # Lista para almacenar los resultados

    for prompt_key in tqdm(rag_object.prompts, desc="Processing Prompts", unit="prompt", leave=False):
        prompt = rag_object.prompts[prompt_key]
        response = rag_object.rag_workflow(prompt)
        
        # Agregar el resultado al diccionario
        results.append({
            "prompt_key": prompt_key,
            "prompt": prompt,
            "response": response
        })

    # Guardar los resultados en un archivo JSON
    with open(output_dir, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()

