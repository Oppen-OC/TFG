from dotenv import load_dotenv
import openai
import os
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import logging
import numpy as np

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

    # Configurar el manejador de archivo para escribir logs en el directorio de logs
    rag_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    rag_handler.setLevel(logging.INFO)

    rag_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    rag_handler.setFormatter(rag_formatter)
    rag_logger.addHandler(rag_handler)
    rag_logger.propagate = False

    def __init__(self, 
                 batch_size=512,
                 overlap=128, 
                 embedding_model = SentenceTransformer("all-MiniLM-L6-v2"),  
                 client = openai.OpenAI( api_key=os.getenv("UPV_API_KEY"), base_url="https://api.poligpt.upv.es"),
                 minimun_distance = 0.9
                 ):
        
        self.batch_size = batch_size
        self.overlap = overlap
        self.faiss_path = RAG.temp_files_dir + "/embeddings_index.faiss"
        self.text_path = RAG.temp_files_dir + "/temp.txt"
        self.embeddings = []
        self.chunks = []
        self.chunk_pages = []
        self.embedding_model = embedding_model
        self.client = client
        self.prompt = ""
        self.logger = RAG.rag_logger
        self.promptCount = 0
        self.minimun_distance = minimun_distance

        self.logger.info("RAG object initialized.")
        print("RAG object initialized.")

    def textSplitter(self):
        try:
            with open(self.text_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if not text.strip():
                    self.logger.error("The text file is empty.")

            chunks = self.spliter(text, separator="###", size=self.batch_size, overlap=self.overlap)
            self.logger.info(f"Text split into {len(chunks)} chunks.")
            self.chunks = chunks

        except Exception as e:
            self.logger.error(f"Error reading the file or splitting text at line {e.__traceback__.tb_lineno}: {e}")

    def generate_embeddings(self):
        self.embeddings = self.embedding_model.encode(self.chunks, batch_size=1024)
        self.logger.info(f"Generated {len(self.embeddings)} embeddings.")
        print(f"Generated {len(self.embeddings)} embeddings.")

    def store_embeddings_in_faiss(self):

        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

        index.add(self.embeddings)
        faiss.write_index(index, "embeddings_index.faiss")

        self.logger.info(f"Stored {len(self.embeddings)} embeddings in FAISS index.")

    def retrieve_similar_chunks(self, query):

        self.promptCount += 1

        try:
            index = faiss.read_index(self.faiss_path)
        except Exception as e:
            print(f"Error reading FAISS index from {self.faiss_path}: {e}")
            return []

        # Encode the query using the model
        query_embedding = self.embedding_model.encode([query])

        # Search for similar chunks in the FAISS index
        distances, indices = index.search(query_embedding, k=5)

        # Convierte a la lista de indices en una lista de enteros
        indices = indices.flatten().tolist()
        distances = distances.flatten().tolist()
        try:
            indices, distances = zip(*[(i, d) for i, d in zip(indices, distances) if d >= self.minimun_distance])
        except ValueError:
            self.logger.warning(f"No similar chunks found for query: {query}")
            return []
  
        
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Retrieved chunks {indices}, with distances {distances}.")

        aux = ""
        for i in range(len(indices)):
            aux += " " + str(self.chunk_pages[indices[i]])

        self.logger.info(f"from pages: {aux}.")
        similar_chunks = [{"text": self.chunks[i], "distance": distances[j]} for j, i in enumerate(indices)]

        return similar_chunks

    def generate_response_with_context(self, query, similar_chunks):
        context = " ".join([rank['text'] for rank in similar_chunks])

        self.prompt = f"""
        Eres un asistente de IA que responde preguntas sobre un documento.
        Context:\n{context}\n
        Question: {query}\n
        Answer:
        """

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model="llama3.3:70b",
            messages=[{"role": "assistant",
                       "content": self.prompt
                       }],
        )

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


    def spliter(self, text, separator, size, overlap):
        pages = text.split(separator)
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        chunks = []
        current_page = 1

        for i, page in enumerate(pages):
            page_chunks = splitter.split_text(page)
            for j, chunk in enumerate(page_chunks):
                # If it's the first chunk of a page and overlap exists, include the last chunk of the previous page
                if j == 0 and i > 0 and overlap > 0:
                    if chunks:
                        previous_chunk = chunks[-1]
                        chunk = previous_chunk[-overlap:] + chunk
                self.chunk_pages.append(current_page)
            chunks.extend(page_chunks)
            current_page += 1

        return chunks

def main():
    with open('code/JSON/Strings.json', 'r', encoding='utf-8') as f:
        strings = json.load(f)
    prompts = strings["Prompts"]["Anexo"]

    reg_object = RAG()
    reg_object.textSplitter()
    reg_object.generate_embeddings()
    reg_object.store_embeddings_in_faiss()

    with open("output.txt", "w", encoding="utf-8") as output_file:
        for prompt in prompts:
            output_file.write(prompts[prompt] + "\n\n")
            output_file.write(reg_object.rag_workflow(prompts[prompt]) + "\n")
            output_file.write("--------------------------------------------\n")

if __name__ == "__main__":
    main()
