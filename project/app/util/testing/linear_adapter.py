import chromadb
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



class chromaTester:
    def __init__(self, base_model=None,validation_data=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        chromadb_path = os.path.abspath(os.path.join(base_dir, "..", "temp_files", "chromadb"))   

        self.model = base_model
        self.validation_data = validation_data
        client = chromadb.PersistentClient(path=chromadb_path) # Create chroma client

        # Create collections for both our specific simulated RAG pipelines
        client.delete_collection(name='licitaciones_collection')
        self.licitaciones_collection = client.get_or_create_collection(name='licitaciones_collection', metadata={"hnsw:space": "cosine"})
        self.add_chunks()
        self.validate()

    def validate(self):
        print("Validando modelo de embedding...")
        results = self.validate_embedding_model()

        print(f"Average Hit Rate @10: {results['average_hit_rate']}")
        print(f"Mean Reciprocal Rank @10: {results['average_reciprocal_rank']*10}")

    def add_chunks(self):
        licitacion_chunks = []
        for i, chunk in enumerate(self.validation_data):
            self.licitaciones_collection.add(
                documents=[chunk["chunk"]],
                ids=[f"chunk_{i}"]
            )
            licitacion_chunks.append(chunk["chunk"])
        print(f"Total chunks to add: {len(licitacion_chunks)}")

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
        
        for data_point in self.validation_data:
            question = data_point['question']
            ground_truth = data_point['chunk']
            
            # Generate embedding for the question
            question_embedding = self.model.encode(question)
            
            # Retrieve documents using the embedding
            retrieved_docs = self.retrieve_documents_embeddings(question_embedding, k)
            
            # Calculate metrics
            hr = self.hit_rate(retrieved_docs, ground_truth, k)
            rr = self.reciprocal_rank(retrieved_docs, ground_truth, k)
            
            hit_rates.append(hr)
            reciprocal_ranks.append(rr)
        
        # Calculate average metrics
        avg_hit_rate = np.mean(hit_rates)
        avg_reciprocal_rank = np.mean(reciprocal_ranks)
        
        return {
            'average_hit_rate': avg_hit_rate,
            'average_reciprocal_rank': avg_reciprocal_rank
        }
    
class uselessData:
    def __init__(self, useless_data_path = r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\randomData.pdf"):
        self.useless_data_path = useless_data_path
        self.useless_chunks = self.get_chunks()

    def get_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800,
            chunk_overlap=400,
        )

        useless_loader = PyPDFLoader(self.useless_data_path)
        useless_pages = useless_loader.load()

        useless_document = ""
        for i in range(len(useless_pages)):
            useless_document += useless_pages[i].page_content

        useless_chunks = text_splitter.split_text(useless_document)

        return useless_chunks
    
    def random_negative(self):
        random_sample = random.choice(self.useless_chunks)
        return random_sample

class LinearAdapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        return self.linear(x)
    
class TripletDataset(Dataset):
    def __init__(self, data, base_model, negative_sampler):
        self.data = data
        self.base_model = base_model
        self.negative_sampler = negative_sampler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['question']
        positive = item['chunk']
        negative = self.negative_sampler()
        
        query_emb = self.base_model.encode(query, convert_to_tensor=True)
        positive_emb = self.base_model.encode(positive, convert_to_tensor=True)
        negative_emb = self.base_model.encode(negative, convert_to_tensor=True)
        
        return query_emb, positive_emb, negative_emb

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def train_linear_adapter(base_model, train_data, negative_sampler, num_epochs=10, batch_size=32, 
                         learning_rate=2e-5, warmup_steps=100, max_grad_norm=1.0, margin=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the LinearAdapter
    adapter = LinearAdapter(base_model.get_sentence_embedding_dimension()).to(device)
    
    # Define loss function and optimizer
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = AdamW(adapter.parameters(), lr=learning_rate)
    
    # Create dataset and dataloader
    dataset = TripletDataset(train_data, base_model, negative_sampler)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate total number of training steps
    total_steps = len(dataloader) * num_epochs
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            query_emb, positive_emb, negative_emb = [x.to(device) for x in batch]
            
            # Forward pass
            adapted_query_emb = adapter(query_emb)
            
            # Compute loss
            loss = triplet_loss(adapted_query_emb, positive_emb, negative_emb)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(adapter.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return adapter


class AdapterModelTesting:
    def __init__(self, base_model, adapter_path, validation_data):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        chromadb_path = os.path.abspath(os.path.join(base_dir, "..", "temp_files", "chromadb")) 
        client = chromadb.PersistentClient(path=chromadb_path) # Create chroma client
        self.validation_data = validation_data
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.adapter = None
        client = chromadb.PersistentClient(path=chromadb_path) # Create chroma client

        # Create collections for both our specific simulated RAG pipelines
        client.delete_collection(name='licitaciones_collection')
        self.licitaciones_collection = client.get_or_create_collection(name='licitaciones_collection', metadata={"hnsw:space": "cosine"})

    def train_model(self, train_data, num_epochs):    
        # Define the kwargs dictionary
        adapter_kwargs = {
            'num_epochs': num_epochs,
            'batch_size': 32,
            'learning_rate': 0.003,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'margin': 1.0
        }

        useless_data = uselessData()
        random_negative = useless_data.random_negative

        # Train the adapter using the kwargs dictionary
        trained_adapter = train_linear_adapter(self.base_model, train_data, random_negative, **adapter_kwargs)

        # Create a dictionary to store both the adapter state_dict and the kwargs
        save_dict = {
            'adapter_state_dict': trained_adapter.state_dict(),
            'adapter_kwargs': adapter_kwargs
        }

        # Save the combined dictionary
        adapter_path = rf'D:\WINDOWS\Escritorio\TFG\project\app\util\testing\models\linear_adapter_{num_epochs}epoch.pth'
        torch.save(save_dict, adapter_path)
        
    # Function to encode query using the adapter
    def encode_query(self, query, base_model, adapter):
        device = next(self.adapter.parameters()).device
        query_emb = base_model.encode(query, convert_to_tensor=True).to(device)
        adapted_query_emb = adapter(query_emb)
        return adapted_query_emb.cpu().detach().numpy()

    def add_chunks(self):
        licitacion_chunks = []
        for i, chunk in enumerate(self.validation_data):
            self.licitaciones_collection.add(
                documents=[chunk["chunk"]],
                ids=[f"chunk_{i}"]
            )
            licitacion_chunks.append(chunk["chunk"])
        print(f"Total chunks to add: {len(licitacion_chunks)}")

    def test_adapter(self):
        # Later, loading and using the saved information
        loaded_dict = torch.load(self.adapter_path)

        # Recreate the adapter
        loaded_adapter = LinearAdapter(self.base_model.get_sentence_embedding_dimension())  # Initialize with appropriate parameters
        loaded_adapter.load_state_dict(loaded_dict['adapter_state_dict'])

        self.adapter = loaded_adapter

        # Access the training parameters
        training_params = loaded_dict['adapter_kwargs']

        print("Adapter loaded successfully.")
        print("Training parameters used:")
        for key, value in training_params.items():
            print(f"{key}: {value}")

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

    def retrieve_documents_embeddings(self, query_embedding, k=10):
        query_embedding_list = query_embedding.tolist()
        
        results = self.licitaciones_collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k)
        return results['documents'][0]

    def evaluate_adapter(self, k=10):
        hit_rates = []
        reciprocal_ranks = []
        
        for data_point in self.validation_data:
            question = data_point['question']
            ground_truth = data_point['chunk']
            
            # Generate embedding for the question
            question_embedding = self.encode_query(question, self.base_model, self.adapter)
            # Retrieve documents using the embedding
            retrieved_docs = self.retrieve_documents_embeddings(question_embedding, k)
            
            # Calculate metrics
            hr = self.hit_rate(retrieved_docs, ground_truth, k)
            rr = self.reciprocal_rank(retrieved_docs, ground_truth, k)
            
            hit_rates.append(hr)
            reciprocal_ranks.append(rr)
        
        # Calculate average metrics
        avg_hit_rate = np.mean(hit_rates)
        avg_reciprocal_rank = np.mean(reciprocal_ranks)
        
        return {
            'average_hit_rate': avg_hit_rate,
            'average_reciprocal_rank': avg_reciprocal_rank
        }
    def evaluate(self):
        self.add_chunks()
        self.test_adapter()
        results = self.evaluate_adapter()
        print(f"Average Hit Rate @10: {results['average_hit_rate']}")
        print(f"Mean Reciprocal Rank @10: {results['average_reciprocal_rank'] * 10}")

if __name__ == "__main__":

    base_model = SentenceTransformer('all-MiniLM-L6-v2')

    with open(r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\json_files\val_data_5f.json", 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    with open(r"D:\WINDOWS\Escritorio\TFG\project\app\util\testing\json_files\train_data_5f.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    adapter_path = r'D:\WINDOWS\Escritorio\TFG\project\app\util\testing\models\linear_adapter_V2_E30_B32_L0.003.pth'

    chromaTester(base_model, validation_data)

    am = AdapterModelTesting(base_model, adapter_path = adapter_path, validation_data = validation_data)
    # am.train_model(train_data, num_epochs=40)
    am.evaluate()