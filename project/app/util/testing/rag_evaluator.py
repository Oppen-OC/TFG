import os
import json
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_operations import RAG

class RAGEvaluator:
    def __init__(self, output_path, test_data_path):
        with open(output_path, 'r', encoding='utf-8') as file:
            self.responses_file = json.load(file)
        self.test_data_path = test_data_path
        self.test_data = self.load_test_data()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo para embeddings

    def load_test_data(self):
        with open(self.test_data_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def evaluate(self):
        y_true = []  # Respuestas esperadas
        y_pred = []  # Respuestas generadas por RAG
        context_similarities = []  # Similitud de contexto
        keys = self.responses_file["prompts"]

        for i, item in enumerate(self.test_data):
            expected_response = item["response"]

            # Generar respuesta usando RAG
            generated_response = keys[i]["response"]

            # Comparar respuestas (puedes ajustar la lógica de comparación)
            y_true.append(expected_response)
            y_pred.append(generated_response)

            # Evaluar similitud de contexto
            if expected_response and generated_response:
                embedding_true = self.model.encode(expected_response, convert_to_tensor=True)
                embedding_pred = self.model.encode(generated_response, convert_to_tensor=True)
                similarity = util.cos_sim(embedding_true, embedding_pred).item()
                context_similarities.append(similarity)

        # Calcular métricas
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        avg_context_similarity = sum(context_similarities) / len(context_similarities) if context_similarities else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_context_similarity": avg_context_similarity
        }

def main():
    file_name = "CMAYOR/2023/03Y05/19.json"
    file_name = file_name.replace("/", "-")
    # Construir la ruta al archivo JSON de prueba
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_path = os.path.join(base_dir, "JSON", file_name)
    output_path = os.path.join(base_dir, "temp_files", "output.json")

    # Crear el evaluador
    evaluator = RAGEvaluator(output_path, test_data_path)

    # Evaluar el rendimiento
    metrics = evaluator.evaluate()

    # Mostrar los resultados
    print("Resultados de la evaluación:")
    print(f"Precisión: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1-Score: {metrics['f1_score']:.2f}")
    print(f"Similitud promedio de contexto: {metrics['avg_context_similarity']:.2f}")

if __name__ == "__main__":
    main()