import chromadb

# Create chroma client
path = "/Users/alucek/Documents/Jupyter_Notebooks/ft_emb/chromadb"
client = chromadb.PersistentClient(path=path)

# Create collections for both our specific simulated RAG pipelines
apple_collection = client.get_or_create_collection(name='apple_collection', metadata={"hnsw:space": "cosine"})

print("hola")