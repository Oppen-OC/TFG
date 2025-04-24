from dotenv import load_dotenv
import openai
import os
import faiss
import json
import logging
import numpy as np
from tqdm import tqdm

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
                 embedding_model = "nomic-embed-text", #SentenceTransformer("all-MiniLM-L6-v2")  
                 client = openai.OpenAI(api_key=os.getenv("UPV_API_KEY"), base_url="https://api.poligpt.upv.es"),
                 minimun_distance = 0.5
                 ):
        
        self.file = file
        self.faiss_path = RAG.temp_files_dir + "/embeddings_index.faiss"
        self.text_path = RAG.temp_files_dir + "/temp_document.txt"
        self.embeddings = []
        self.embedding_model = embedding_model
        self.client = client
        self.prompt = ""
        self.logger = RAG.rag_logger
        self.chunks = []
        self.tables = []
        self.metadata = []
        self.minimun_distance = minimun_distance

        if os.path.exists(self.faiss_path):
                os.remove(self.faiss_path)
                self.logger.info(f"Previous FAISS index deleted: {self.faiss_path}")

        self.logger.info("RAG object initialized.")
        print("RAG object initialized.")


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

        # Cargar el índice FAISS
        try:
            index = faiss.read_index(self.faiss_path)
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {e}")
            return []

        # Buscar los chunks más similares en el índice FAISS
        try:
            distances, indices = index.search(query_embedding, k=5)  # Buscar los 5 más similares
        except Exception as e:
            self.logger.error(f"Error querying FAISS index: {e}")
            return []

        # Convertir los resultados a listas
        indices = indices.flatten().tolist()
        distances = distances.flatten().tolist()

        # Filtrar resultados según la distancia mínima
        try:
            filtered_results = [(i, d) for i, d in zip(indices, distances) if d >= self.minimun_distance]
            if not filtered_results:
                self.logger.warning(f"No similar chunks found for query: {query}")
                return []
            indices, distances = zip(*filtered_results)
        except ValueError:
            self.logger.warning(f"No similar chunks found for query: {query}")
            return []

        # Recuperar los chunks correspondientes
        similar_chunks = [{"text": self.chunks[i], "distance": distances[j]} for j, i in enumerate(indices)]

        # Log de los resultados
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Retrieved chunks: {indices}, with distances: {distances}.")

        return similar_chunks

    def generate_response_with_context(self, query, similar_chunks):
        context = " ".join([str(rank['text']) for rank in similar_chunks])

        self.prompt = f"""
        Eres un asistente de IA que responde preguntas sobre un documento.
        Context:\n{context}\n
        Question: {query}\n
        Answer:
        """
        print("Procesando")

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model="llama3.3:70b",
            messages=[{"role": "assistant",
                       "content": self.prompt
                       }],
        )
        print("respondido")

        answer = response.choices[0].message.content.strip()

        # Check if the answer is relevant or empty
        if not answer or "I'm sorry" in answer or "I don't know" in answer:
            return "N/A"

        return answer

    def rag_workflow(self, query):
        similar_chunks = self.retrieve_similar_chunks(query)
        if similar_chunks == []:
            self.logger.info(f"No similar chunks found for query: {query}")
            return "N/A"
        
        return self.generate_response_with_context(query, similar_chunks)

    def embbed(self):
        # Generar embeddings para todos los chunks
        embeddings = []
        for i, d in enumerate(self.file["chunks"]):
            try:
                # Llamar a la API de OpenAI para generar el embedding
                response = self.client.embeddings.create(
                    input=d,
                    model=self.embedding_model  # Cambiar a "self.embedding_model" si es necesario
                )
                embedding = response.data[0].embedding  # Extraer el embedding
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error generating embedding for chunk {i}: {e}")
                continue

        # Convertir los embeddings a un array de NumPy
        embeddings_array = np.array(embeddings, dtype='float32')

        # Asegurarse de que el array sea bidimensional
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)

        # Inicializar el índice FAISS
        dimension = embeddings_array.shape[1]  # Dimensión de los embeddings
        index = faiss.IndexFlatL2(dimension)  # Distancia L2 (Euclidiana)

        # Agregar los embeddings al índice FAISS
        index.add(embeddings_array)

        # Guardar el índice FAISS en disco
        faiss.write_index(index, self.faiss_path)

        self.logger.info(f"Stored {len(self.chunks)} embeddings in FAISS index.")

    def embed_json(self):
        try:
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
                    self.chunks.append(chunk)  # Extract "chunks" field
                    self.metadata.append(meta) # Extract "metadata" field

            for i, page in enumerate(data):
                if "tables" in page:
                    table = page.get("tables", [])
                    meta = page.get("page", []), page.get("sections", [])
                    self.chunks.append(table)  # Extract "chunks" field
                    self.metadata.append(meta)  # Extract "metadata" field
            
            # Generate embeddings for each chunk
            embeddings = []
            for i, chunk in enumerate(self.chunks):
                input = ""
                try:
                    if isinstance(chunk, str):
                        input = chunk

                    elif chunk:
                        zipped = list(zip(*chunk)) 
                        input = "\n".join([", ".join(row) for row in zipped])
                    
                    if input != "":
                        response = self.client.embeddings.create(
                            input=input,
                            model=self.embedding_model
                        )
                        self.logger.info(f"created embedding for chunk {i}: {input}")
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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    string_path = os.path.join(base_dir, 'JSON', 'Strings.json')
    output_dir = os.path.join(base_dir, 'temp_files', 'output.txt')
    file_path = os.path.join(base_dir, 'temp_files', 'temp_document.json')

    with open(string_path, 'r', encoding='utf-8') as f:
        strings = json.load(f)
    prompts = strings["Prompts"]["Ajuntament"]    
    
    with open(file_path, 'r', encoding='utf-8') as f:
        file = json.load(f)

    rag_object = RAG(file)
    # rag_object.embbed()
    rag_object.embed_json()

    with open(output_dir, "w", encoding="utf-8") as output_file:
        for prompt_key in tqdm(prompts, desc="Processing Prompts", unit="prompt", leave = False):
            output_file.write(prompts[prompt_key] + "\n\n")
            output_file.write(rag_object.rag_workflow(prompts[prompt_key]) + "\n")
            output_file.write("--------------------------------------------\n")

if __name__ == "__main__":
    main()

