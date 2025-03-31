from dotenv import load_dotenv
from openai import OpenAI
import os
from langchain_core.documents import Document
from sklearn.cluster import KMeans
import numpy as np
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search

load_dotenv()

def load_file_to_document(file_path, chunk_size=500):
    """
    Carga un documento plano de texto a un objeto Document, divide su contenido
    en chunks de un tamaño indicado en el parametro de la función y le asigna un 
    index para poder ser referenciado en un futuro.
    """
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            i = 0
            while True:
                content = file.read(chunk_size)
                if not content:  # Stop when the file is fully read
                    break
                document = Document(
                    page_content=content,
                    metadata={"source": file_path, "chunk_index": i}
                )
                documents.append(document)
                i += 1
        return documents
    except Exception as e:
        print(f"Error reading the file: {e}")
        return []


client_upv = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.poligpt.upv.es",
)

client = OpenAI(
    api_key=os.getenv("OTRA"),
)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def main():
    try:
        print("Hola")
        client_local = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.poligpt.upv.es",
        )
        print(client_local)
        response = client_local.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': 'Hello, world!',
                }
            ],
            model='poligpt',
        )
        print(response)
        # Accede al atributo 'content' directamente
        return response.choices[0].message.content
    except Exception as e:
        print(e)

if __name__ == "__main__":
    print("HOla")
    # print(main())
    print(load_file_to_document("C:/Users/DALPDEL/Desktop/TFG/code/temp.txt")[-1])