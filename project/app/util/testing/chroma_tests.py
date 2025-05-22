from chunking_evaluation import BaseChunker, GeneralEvaluation
from chromadb.utils import embedding_functions

# Define a custom chunking class
class CustomChunker(BaseChunker):
    def split_text(self, text):
        # Custom chunking logic
        return [text[i:i+1200] for i in range(0, len(text), 1200)]

# Instantiate the custom chunker and evaluation
chunker = CustomChunker()
evaluation = GeneralEvaluation()

# Choose embedding function
default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="OPENAI_API_KEY",
    model_name="text-embedding-3-large"
)

# Evaluate the chunker
results = evaluation.run(chunker, default_ef)

print(results)
# {'iou_mean': 0.17715979570301696, 'iou_std': 0.10619791407460026, 
# 'recall_mean': 0.8091207841640163, 'recall_std': 0.3792297991952294}