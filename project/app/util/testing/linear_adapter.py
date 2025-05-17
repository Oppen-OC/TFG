import chromadb
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class linearAdapter:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chromadb_path = os.path.abspath(os.path.join(base_dir, "..", "temp_files", "chromadb"))    

    def __init__(self, model_name='all-MiniLM-L6-v2',validation_data=None):
        self.model = SentenceTransformer(model_name)
        with open(os.path.abspath(os.path.join(linearAdapter.base_dir, "json_files", validation_data)), 'r', encoding='utf-8') as f:
            self.validation_data = json.load(f)
        client = chromadb.PersistentClient(path=linearAdapter.chromadb_path) # Create chroma client

        # Create collections for both our specific simulated RAG pipelines
        client.delete_collection(name='licitaciones_collection')
        self.licitaciones_collection = client.get_or_create_collection(name='licitaciones_collection', metadata={"hnsw:space": "cosine"})
        self.add_chunks()


    def add_chunks(self):
        licitacion_chunks = []
        for doc_code, page in self.validation_data.items():
            if doc_code == "CMAYOR/2023/03Y05/19":
                print(f"Adding chunks for document {doc_code}...")
                for item in page:
                    chunk = item["chunk"]
                    licitacion_chunks.append(chunk)
        print(f"Total chunks to add: {len(licitacion_chunks)}")

        # Añadir los chunks a la colección
        for i, chunk in enumerate(licitacion_chunks):
            self.licitaciones_collection.add(
                documents=[chunk],
                ids=[f"chunk_{i}"]
            )

    def retrieve_documents_embeddings(self, query_embedding, k=10):
        query_embedding_list = query_embedding.tolist()
        
        results = self.licitaciones_collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k)
        return results['documents'][0]

    # Define la posición media del chunk esperado en los resultados obtenidos
    def reciprocal_rank(self, retrieved_docs, ground_truth, k):
        try:
            rank = retrieved_docs.index(ground_truth) + 1
            return 1.0 / rank if rank <= k else 0.0
        except ValueError:
            return 0.0
    
    # Define la tasa de aciertos (hit rate) como 1 si el chunk esperado está entre los k primeros resultados, 0 en caso contrario
    def hit_rate(self,retrieved_docs, ground_truth, k):
        return 1.0 if ground_truth in retrieved_docs[:k] else 0.0
    

    def validate_embedding_model(self, k=10):
        hit_rates = []
        reciprocal_ranks = []
        print("Validando modelo de embedding...")
        
        for doc_code, data_points in tqdm(self.validation_data.items(), desc="Validando documentos", unit="documento"):
            if doc_code == "CMAYOR/2023/03Y05/19":
                for item in data_points:
                    chunk = item["chunk"]
                    questions = item["questions"]
                    for question in questions.values():
                        # Generar embedding para la pregunta
                        question_embedding = self.model.encode(question)

                        # Recuperar documentos usando el embedding
                        retrieved_docs = self.retrieve_documents_embeddings(question_embedding, k)

                        # Calcular métricas
                        hr = self.hit_rate(retrieved_docs, chunk, k)
                        rr = self.reciprocal_rank(retrieved_docs, chunk, k)

                        hit_rates.append(hr)
                        reciprocal_ranks.append(rr)

        # Calculate average metrics
        avg_hit_rate = np.mean(hit_rates)
        avg_reciprocal_rank = np.mean(reciprocal_ranks)
        
        return {
            'average_hit_rate': avg_hit_rate,
            'average_reciprocal_rank': avg_reciprocal_rank
        }
    
if __name__ == "__main__":
    print("Validando modelo de embedding...")
    la = linearAdapter(validation_data = "validation.json")
    results = la.validate_embedding_model()

    print(f"Average Hit Rate @10: {results['average_hit_rate']}")
    print(f"Mean Reciprocal Rank @10: {results['average_reciprocal_rank']*10}")